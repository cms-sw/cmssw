/* $Id: EcalDccWeightBuilder.cc,v 1.9 2009/10/13 15:56:16 heltsley Exp $
 *
 * authors: Ph. Gras (CEA/Saclay), F. Cavallari (INFN/Roma)
 *          some code copied from CalibCalorimetry/EcalTPGTools code
 *          written by P. Paganini and F. Cavallari
 */

#define DB_WRITE_SUPPORT

#include "CalibCalorimetry/EcalSRTools/interface/EcalDccWeightBuilder.h"

#include <limits>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <TFile.h>
#include <TTree.h>

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#ifdef DB_WRITE_SUPPORT
#  include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#  include "OnlineDB/EcalCondDB/interface/ODWeightsDat.h"
#endif //DB_WRITE_SUPPORT defined

#include "CalibCalorimetry/EcalSRTools/src/PasswordReader.h"

using namespace std;
using namespace edm;

const double EcalDccWeightBuilder::weightScale_ = 1024.;


//TODO: handling case of weight encoding saturation: weights shall be downscaled to prevent saturation

EcalDccWeightBuilder::EcalDccWeightBuilder(edm::ParameterSet const& ps):
  dcc1stSample_(ps.getParameter<int>("dcc1stSample")),
  sampleToSkip_(ps.getParameter<int>("sampleToSkip")),
  nDccWeights_(ps.getParameter<int>("nDccWeights")),
  inputWeights_(ps.getParameter<vector<double> >("inputWeights")),
  mode_(ps.getParameter<string>("mode")),
  dccWeightsWithIntercalib_(ps.getParameter<bool>("dccWeightsWithIntercalib")),
  writeToDB_(ps.getParameter<bool>("writeToDB")),
  writeToAsciiFile_(ps.getParameter<bool>("writeToAsciiFile")),
  writeToRootFile_(ps.getParameter<bool>("writeToRootFile")),
  asciiOutputFileName_(ps.getParameter<string>("asciiOutputFileName")),
  rootOutputFileName_(ps.getParameter<string>("rootOutputFileName")),
  dbSid_(ps.getParameter<string>("dbSid")),
  dbUser_(ps.getParameter<string>("dbUser")),
  dbPassword_(ps.getUntrackedParameter<string>("dbPassword","")),
  dbTag_(ps.getParameter<string>("dbTag")),
  dbVersion_(ps.getParameter<int>("dbVersion")),
  sqlMode_(ps.getParameter<bool>("sqlMode")),
  calibMap_(emptyCalibMap_)
{
  if(mode_=="weightsFromConfig"){
    imode_ = WEIGHTS_FROM_CONFIG;
    if(inputWeights_.size()!=(unsigned)nDccWeights_){
      throw cms::Exception("Config")
        << "Inconsistent configuration. 'nDccWeights' parameters indicated "
        << nDccWeights_ << " weights while parameter 'inputWeights_' contains "
        << inputWeights_.size() << " weight values!\n";
    }
  } else if(mode_=="computeWeights"){
    imode_ = COMPUTE_WEIGHTS;
  } else{
    throw cms::Exception("Config")
      << "Invalid value ('" << mode_ << "') for parameter mode. "
      << "Valid values are: 'weightsFromConfig' and 'computeWeights'\n";
  }
}


void
EcalDccWeightBuilder::analyze(const edm::Event& event,
                              const edm::EventSetup& es){

  edm::ESHandle<EcalElectronicsMapping> handle;
  es.get<EcalMappingRcd>().get(handle);
  ecalElectronicsMap_ = handle.product();
  
  // Retrieval of intercalib constants
  if(dccWeightsWithIntercalib_){
    ESHandle<EcalIntercalibConstants> hIntercalib ;
    es.get<EcalIntercalibConstantsRcd>().get(hIntercalib) ;
    const EcalIntercalibConstants* intercalib = hIntercalib.product();
    calibMap_ = intercalib->getMap();
  }

  //gets geometry
  es.get<CaloGeometryRecord>().get(geom_);

  
  //computes the weights:
  computeAllWeights(dccWeightsWithIntercalib_);

  //Writing out weights.
  if(writeToAsciiFile_) writeWeightToAsciiFile();
  if(writeToRootFile_)  writeWeightToRootFile();
  if(writeToDB_)        writeWeightToDB();
}

void EcalDccWeightBuilder::computeAllWeights(bool withIntercalib){
  const int nw = nDccWeights_;
  int iSkip0_ = sampleToSkip_>=0?(sampleToSkip_-dcc1stSample_):-1;

  EcalSimParameterMap parameterMap;
  const vector<DetId>& ebDetIds
    = geom_->getValidDetIds(DetId::Ecal, EcalBarrel);

  //   cout << __FILE__ << ":" << __LINE__ << ": "
  //        <<  "Number of EB det IDs: " << ebDetIds.size() << "\n";
  
  const vector<DetId>& eeDetIds
    = geom_->getValidDetIds(DetId::Ecal, EcalEndcap);

  //  cout << __FILE__ << ":" << __LINE__ << ": "
  //        <<  "Number of EE det IDs: " << eeDetIds.size() << "\n";
  
  
  vector<DetId> detIds(ebDetIds.size()+eeDetIds.size());
  copy(ebDetIds.begin(), ebDetIds.end(), detIds.begin());
  copy(eeDetIds.begin(), eeDetIds.end(), detIds.begin()+ebDetIds.size());
  
  vector<double> baseWeights(nw); //weight obtained from signal shape
  vector<double> w(nw); //weight*intercalib
  vector<int> W(nw);    //weight in hw encoding (integrer)
  double prevPhase = numeric_limits<double>::min();


  if(imode_==WEIGHTS_FROM_CONFIG){
    assert(inputWeights_.size()==baseWeights.size());
    copy(inputWeights_.begin(), inputWeights_.end(), baseWeights.begin());
  }
  
  for(vector<DetId>::const_iterator it = detIds.begin();
      it != detIds.end(); ++it){
    
    double phase = parameterMap.simParameters(*it).timePhase();
    int binOfMax = parameterMap.simParameters(*it).binOfMaximum();
    
#if 0
    //for debugging...
    cout << __FILE__ << ":" << __LINE__ << ": ";
    if(it->subdetId()==EcalBarrel){
      cout << "ieta = " << setw(4) << ((EBDetId)(*it)).ieta()
           << " iphi = " << setw(4) << ((EBDetId)(*it)).iphi() << " ";
    } else if(it->subdetId()==EcalEndcap){
      cout << "ix = " << setw(3) << ((EEDetId)(*it)).ix()
           << " iy = " << setw(3) << ((EEDetId)(*it)).iy()
           << " iz = " << setw(1) << ((EEDetId)(*it)).iy() << " ";
    } else{
      throw cms::Exception("EcalDccWeightBuilder")
        << "Bug found in " << __FILE__ << ":" << __LINE__ << ": "
        << "Got a detId which is neither tagged as ECAL Barrel "
        << "not ECAL endcap while looping on ECAL cell detIds\n";
    }
    cout << " -> phase: "  << phase << "\n";
    cout << " -> binOfMax: " << binOfMax << "\n";
#endif
    
    try{
      EBShape ebShape;
      EEShape eeShape;
      EcalShapeBase* pShape;      

      if(it->subdetId()==EcalBarrel){
	pShape = &ebShape;
      } else if(it->subdetId()==EcalEndcap){
	pShape = &eeShape;
      } else{
	throw cms::Exception("EcalDccWeightBuilder")
	  << "Bug found in " << __FILE__ << ":" << __LINE__ << ": "
	  << "Got a detId which is neither tagged as ECAL Barrel "
	  << "not ECAL endcap while looping on ECAL cell detIds\n";
      }
      
      if(phase!=prevPhase){
        if(imode_==COMPUTE_WEIGHTS){
	  if(it->subdetId()==EcalBarrel){
	    computeWeights(*pShape, binOfMax, phase, 
			   dcc1stSample_-1, nDccWeights_, iSkip0_,
			   baseWeights);
	  } 
	  prevPhase = phase;
	}
      }
      for(int i = 0; i < nw; ++i){
	w[i] = baseWeights[i];
	if(withIntercalib) w[i]*= intercalib(*it);
      }
      unbiasWeights(w, &W);
      encodedWeights_[*it] = W;
    } catch(std::exception& e){
      cout << __FILE__ << ":" << __LINE__ << ": ";
      if(it->subdetId()==EcalBarrel){
	cout << "ieta = " << setw(4) << ((EBDetId)(*it)).ieta()
	     << " iphi = " << setw(4) << ((EBDetId)(*it)).iphi() << " ";
      } else if(it->subdetId()==EcalEndcap){
	cout << "ix = " << setw(3) << ((EEDetId)(*it)).ix()
	     << " iy = " << setw(3) << ((EEDetId)(*it)).iy()
	     << " iz = " << setw(1) << ((EEDetId)(*it)).iy() << " ";
      } else{
	cout << "DetId " << (uint32_t) (*it);
      }
      cout <<  "phase: "  << phase << "\n";
      throw;
    }
  }
}

void
EcalDccWeightBuilder::computeWeights(const EcalShapeBase& shape,
				     int binOfMax,
				     double timePhase,
				     int iFirst,
				     int nWeights, int iSkip,
				     vector<double>& result){
  double sum2 = 0.;
  double sum = 0;
  result.resize(nWeights);

  int nActualWeights = 0;

  const double tzero = -(binOfMax-1)*25+timePhase + shape.timeToRise();//ns

  for(int i=0; i<nWeights; ++i){
    double t_ns = tzero+(iFirst+i)*25;
    double s = shape(t_ns);
    if(i==iSkip){
      continue;
    }
    result[i] = s;
    sum += s;
    sum2 += s*s;
    ++nActualWeights;
  }
  for(int i=0; i<nWeights; ++i){
    if(i==iSkip){
      result[i] = 0;
    } else{
      result[i] = (result[i]-sum/nActualWeights)/(sum2-sum*sum/nActualWeights);
    }
  }
}

int EcalDccWeightBuilder::encodeWeight(double w){
  return lround(w * weightScale_);
}

double EcalDccWeightBuilder::decodeWeight(int W){
  return ((double) W) / weightScale_;
}


template<class T>
void EcalDccWeightBuilder::sort(const std::vector<T>& a,
				std::vector<int>& s,
				bool decreasingOrder){
  //   cout << __FILE__ << ":" << __LINE__ << ": "
  //        << "sort input array:" ;
  //   for(unsigned i=0; i<a.size(); ++i){
  //     cout << "\t" << a[i];
  //   }
  //   cout << "\n";
  
  //performs a bubble sort: adjacent elements are successively swapped 2 by 2
  //until the list is finally sorted.
  bool changed = false;
  s.resize(a.size());
  for(unsigned i=0; i<a.size(); ++i) s[i] = i;
  if(a.size() == 0) return;
  do {
    changed = false;
    for(unsigned i = 0; i < a.size()-1; ++i){
      const int j = s[i];
      const int nextj = s[i+1];
      if((decreasingOrder && (a[j] < a[nextj]))
	 || (!decreasingOrder && (a[j] > a[nextj]))){
	std::swap(s[i], s[i+1]);
	changed = true;
      }
    }
  } while(changed);
  
  //   cout << __FILE__ << ":" << __LINE__ << ": "
  //        << "sorted list of indices:" ;
  //   for(unsigned i=0; i < s.size(); ++i){
  //     cout << "\t" << s[i];
  //   }
  //   cout << "\n";
}
  
void EcalDccWeightBuilder::unbiasWeights(std::vector<double>& weights,
					 std::vector<int>* encodedWeights){
  const unsigned nw = weights.size();
  
  //computes integer weights, weights residuals and weight sum residual:
  vector<double> dw(nw); //weight residuals due to interger encoding
  vector<int> W(nw); //integer weights
  int wsum = 0;
  for(unsigned i = 0; i < nw; ++i){
    W[i] = encodeWeight(weights[i]);
    dw[i] = decodeWeight(W[i]) - weights[i];
    wsum += W[i];
  }

//   cout << __FILE__ << ":" << __LINE__ << ": "
//        <<  "weights before bias correction: ";
//   for(unsigned i=0; i<weights.size(); ++i){
//     const double w = weights[i];
//     cout << "\t" << encodeWeight(w) << "(" << w << ", dw = " << dw[i] << ")";
//   }
//   cout << "\t sum: " << wsum << "\n";
  
  //sorts weight residuals in decreasing order:
  vector<int> iw(nw);
  sort(dw, iw, true);

  //compensates weight sum residual by adding or substracting 1 to weights
  //starting from:
  // 1) the weight with the minimal signed residual if the correction
  // is positive (wsum<0)
  // 2) the weight with the maximal signed residual if the correction
  // is negative (wsum>0)
  int wsumSign = wsum>0?1:-1;
  int i = wsum>0?0:(nw-1);
  while(wsum!=0){
    W[iw[i]] -= wsumSign;
    wsum -= wsumSign;
    i += wsumSign;
    if(i<0 || i>=(int)nw){ //recompute the residuals if a second iteration is
      // needed (in principle, it is not expected with usual input weights), : 
      for(unsigned i = 0; i < nw; ++i){
	dw[i] = decodeWeight(W[i]) - weights[i];
	sort(dw, iw, true);
      }
    }
    if(i<0) i = nw-1;
    if(i>=(int)nw) i = 0;
  }

//   cout << __FILE__ << ":" << __LINE__ << ": "
//        <<  "weights after bias correction: ";
//   for(unsigned i=0; i<weights.size(); ++i){
//     cout << "\t" << W[i] << "(" << decodeWeight(W[i]) << ", dw = "
// 	 << (decodeWeight(W[i])-weights[i]) << ")";
//   }
//   cout << "\n";
  
  //copy result
  if(encodedWeights!=0) encodedWeights->resize(nw);
  for(unsigned i = 0; i < nw; ++i){
    weights[i] = decodeWeight(W[i]);
    if(encodedWeights) (*encodedWeights)[i] = W[i];
  }
}

double EcalDccWeightBuilder::intercalib(const DetId& detId){
  // get current intercalibration coeff
  double coef;
  EcalIntercalibConstantMap::const_iterator itCalib
    = calibMap_.find(detId.rawId());
  if(itCalib != calibMap_.end()){
    coef = (*itCalib);
  } else{
    coef = 1.;
    std::cout << (uint32_t) detId
	      << " not found in EcalIntercalibConstantMap"<<std::endl ;
  }
#if 0
  cout << __FILE__ << ":" << __LINE__ << ": ";
  if(detId.subdetId()==EcalBarrel){
    cout <<  "ieta = " << ((EBDetId)detId).ieta()
	 << " iphi = " << ((EBDetId)detId).iphi();
  } else{
    cout << "ix = " << ((EEDetId)detId).ix()
	 << " iy = " << ((EEDetId)detId).iy()
	 << " iz = " << ((EEDetId)detId).zside();
  }
  cout << " coef = " <<  coef << "\n";
#endif
  return coef;
}

void EcalDccWeightBuilder::writeWeightToAsciiFile(){
  string fName = asciiOutputFileName_.size()!=0?
    asciiOutputFileName_.c_str()
    :"dccWeights.txt";
  ofstream file(fName.c_str());
  if(!file.good()){
    throw cms::Exception("Output")
      << "Failed to open file '"
      << fName
      << "'for writing DCC weights\n";
  }

  const char* comment = sqlMode_?"-- ":"# ";
  
  file << comment << "List of weights for amplitude estimation to be used in DCC for\n"
       << comment << "zero suppresssion.\n\n";
  if(!sqlMode_){
    file << comment << "Note: RU: trigger tower in EB, supercrystal in EE\n"
	 << comment << "      xtl: crystal electronic channel id in RU, from 1 to 25\n\n"
	 << comment << " DetId    SM  FED RU xtl weights[0..5]...\n";
  }
  
  if(sqlMode_){
    file << "variable recid number;\n"
      "exec select COND2CONF_INFO_SQ.NextVal into :recid from DUAL;\n"
      "insert into weights_info (rec_id,tag,version) values (:recid,'"
	 << dbTag_ << "'," << dbVersion_ << ");\n";
    file << "\n" << comment
	 << "index of first sample used in the weighting sum\n"
      "begin\n"
      "  for fedid in " << ecalDccFedIdMin << ".." << ecalDccFedIdMax
	 << " loop\n"
      "    insert into dcc_weightsample_dat (rec_id, logic_id, sample_id, \n"
      "    weight_number)\n"
      "    values(:recid,fedid," << dcc1stSample_ << ",1);\n"
      "  end loop;\n"
      "end;\n"
      "/\n";
  } else{
    file << "1st DCC sample: " << dcc1stSample_ << "\n";
  }

  file << "\n" << comment << "list of weights per crystal channel\n";
  
  for(map<DetId, std::vector<int32_t> >::const_iterator it
	= encodedWeights_.begin();
      it !=  encodedWeights_.end();
      ++it){
    const DetId& detId = it->first;
    
    int fedId;
    int smId;
    int ruId;
    int xtalId;
    
    //detId ->  fedId, smId, ruId, xtalId
    dbId(detId, fedId, smId, ruId, xtalId);

    char delim = sqlMode_?',':' ';

    if(sqlMode_) file << "-- detId " << detId.rawId() << "\n"
		      << "insert into dcc_weights_dat(rec_id,sm_id,fed_id,"
		   "tt_id, cry_id,\n"
		   "weight_0,weight_1,weight_2,weight_3,weight_4,weight_5) \n"
		   "values ("
		   ":recid";
    
    const vector<int>& weights = it->second;
    if(!sqlMode_) file << setw(10) << detId.rawId();
    file << delim << setw(2) << smId;
    file << delim << setw(3) << fedId;
    file << delim << setw(2) << ruId;
    file << delim << setw(2) << xtalId;
      
    for(unsigned i=0; i<weights.size(); ++i){
      file << delim << setw(5) << weights[i];
    }
    if(sqlMode_) file << ");";
    file << "\n";
  }
  if(!file.good()){
    throw cms::Exception("Output") << "Error while writing DCC weights to '"
				   << fName << "' file.";
  }
}
void EcalDccWeightBuilder::writeWeightToRootFile(){
  string fName = rootOutputFileName_.size()!=0?
    rootOutputFileName_.c_str()
    :"dccWeights.root";
  TFile file(fName.c_str(), "RECREATE");
  if(file.IsZombie()){
    throw cms::Exception("Output")
      << "Failed to open file '"
      << fName
      << "'for writing DCC weights\n";
  }
  TTree t("dccWeights", "Weights for DCC ZS filter");
  const int nWeightMax = 20; //normally n_weights = 6. A different might be used
  //                           used for test purposes.
  struct {
    Int_t detId;
    Int_t fedId;
    Int_t smId;
    Int_t ruId;
    Int_t xtalId;
    Int_t n_weights;
    Int_t weights[nWeightMax];
  } buf;
  t.Branch("weights", &buf,
	   "rawDetId/I:"
	   "feId/I:"
	   "smSlotId/I:"
	   "ruId/I:"
	   "xtalInRuId/I:"
	   "n_weights/I:"
	   "weights[n_weights]/I");
  for(map<DetId, std::vector<int32_t> >::const_iterator it
	= encodedWeights_.begin();
      it !=  encodedWeights_.end();
      ++it){
    buf.detId = it->first.rawId();
    buf.n_weights = it->second.size();

    //detId ->  fedId, smId, ruId, xtalId
    dbId(buf.detId, buf.fedId, buf.smId, buf.ruId, buf.xtalId);

    if(buf.n_weights>nWeightMax){
      throw cms::Exception("EcalDccWeight")
	<< "Number of weights (" << buf.n_weights
	<< ") for DetId " << buf.detId
	<< " exceeded maximum limit (" << nWeightMax
	<< ") of root output format. ";
    }
    copy(it->second.begin(), it->second.end(), buf.weights);
    t.Fill();
  }
  t.Write();
  file.Close();
}

#ifndef DB_WRITE_SUPPORT
void EcalDccWeightBuilder::writeWeightToDB(){
  throw cms::Exception("DccWeight")
    << "Code was compiled without support for writing dcc weights directly "
    " into configuration DB. Configurable writeToDB must be set to False. "
    "sqlMode can be used to produce an SQL*PLUS script to fill the DB\n";
}
#else //DB_WRITE_SUPPORT defined
void EcalDccWeightBuilder::writeWeightToDB(){
  cout << "going to write to the online DB "<<dbSid_<<" user "<<dbUser_<<endl;;
  EcalCondDBInterface* econn;

  try {
    cout << "Making connection..." << flush;
    const string& filePrefix = string("file:");
    if(dbPassword_.find(filePrefix)==0){ //password must be read for a file
      string fileName = dbPassword_.substr(filePrefix.size());
      //substitute dbPassword_ value by the password read from the file
      PasswordReader pr;
      pr.readPassword(fileName, dbUser_, dbPassword_);
    }

    //     cout << __FILE__ << ":" << __LINE__ << ": "
    //           <<  "Password: " << dbPassword_ << "\n";

    econn = new EcalCondDBInterface( dbSid_, dbUser_, dbPassword_ );
    cout << "Done." << endl;
  } catch (runtime_error &e) {
    cerr << e.what() << endl;
    exit(-1);
  }
  
  ODFEWeightsInfo weight_info;
  weight_info.setConfigTag(dbTag_);
  weight_info.setVersion(dbVersion_);
  cout << "Inserting in DB..." << endl;

  econn->insertConfigSet(&weight_info);
  
  int weight_id=weight_info.getId();
  cout << "WeightInfo inserted with ID "<< weight_id<< endl;

  vector<ODWeightsDat> datadel;
  datadel.reserve(encodedWeights_.size());

  vector<ODWeightsSamplesDat> dcc1stSampleConfig(nDccs);
  for(int i = ecalDccFedIdMin; i <= ecalDccFedIdMax; ++i){
    dcc1stSampleConfig[i].setId(weight_id);
    dcc1stSampleConfig[i].setFedId(601+i);
    dcc1stSampleConfig[i].setSampleId(dcc1stSample_);
    dcc1stSampleConfig[i].setWeightNumber(-1); //not used.
  }
  econn->insertConfigDataArraySet(dcc1stSampleConfig, &weight_info);
  
  for(map<DetId, std::vector<int32_t> >::const_iterator it
	= encodedWeights_.begin();
      it !=  encodedWeights_.end();
      ++it){
    const DetId& detId = it->first;
    const unsigned nWeights = 6;
    vector<int> weights(nWeights);

    for(unsigned i=0; i<weights.size(); ++i){
      //completing the weight vector with zeros in case it has
      //less than 6 elements:
      const vector<int>& w = it->second;
      weights[i] = i<w.size()?w[i]:0;
    }

    ODWeightsDat one_dat;
    one_dat.setId(weight_id);

    int fedId;
    int smId;
    int ruId;
    int xtalId;
    
    //detId ->  fedId, smId, ruId, xtalId
    dbId(detId, fedId, smId, ruId, xtalId);
      
    one_dat.setSMId(smId);
    one_dat.setFedId(fedId);
    one_dat.setTTId(ruId);
    one_dat.setCrystalId(xtalId);
    
    one_dat.setWeight0(weights[0]);
    one_dat.setWeight1(weights[1]);
    one_dat.setWeight2(weights[2]);
    one_dat.setWeight3(weights[3]);
    one_dat.setWeight4(weights[4]);
    one_dat.setWeight5(weights[5]);
    
    datadel.push_back(one_dat);
  }
  econn->insertConfigDataArraySet(datadel,&weight_info);
  std::cout<< " .. done insertion in DB "<< endl;
  delete econn;
  cout<< "closed DB connection ... done"  << endl;
}
#endif //DB_WRITE_SUPPORT not defined


void EcalDccWeightBuilder::dbId(const DetId& detId, int& fedId, int& smId,
				int& ruId,
				int& xtalId) const{
  const EcalElectronicsId& elecId
    = ecalElectronicsMap_->getElectronicsId(detId);
  
  fedId = 600 + elecId.dccId();
  ruId =  ecalElectronicsMap_->getElectronicsId(detId).towerId();
  
  if(detId.subdetId()==EcalBarrel) {
    smId=((EBDetId)detId).ism();
  } else{
    smId = 10000-fedId; //no SM in EE. Use some unique value to satisfy
    //              current DB PK constraints.
  }
  const int stripLength = 5; //1 strip = 5 crystals in a row
  xtalId = (elecId.stripId()-1)*stripLength  + elecId.xtalId();

#if 0
  cout << __FILE__ << ":" << __LINE__ << ": FED ID "
       <<  fedId << "\n";

  cout << __FILE__ << ":" << __LINE__ << ": SM logical ID "
       <<  smId << "\n";
  
  cout << __FILE__ << ":" << __LINE__ << ": RU ID (TT or SC): "
       <<  ruId << "\n";
  
  cout << __FILE__ << ":" << __LINE__ << ": strip:"
       <<  elecId.stripId() << "\n";
  
  cout << __FILE__ << ":" << __LINE__ << ": xtal in strip: "
       <<  elecId.xtalId() << "\n";

  cout << __FILE__ << ":" << __LINE__ << ": xtalId in RU: "
       <<  xtalId << "\n";
#endif
}
