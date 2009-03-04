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

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

using namespace std;
using namespace edm;

const double EcalDccWeightBuilder::weightScale_ = 1024.;


//TODO: handling case of weight encoding saturation: weights shall be downscaled to prevent saturation

EcalDccWeightBuilder::EcalDccWeightBuilder(edm::ParameterSet const& ps):
  //  intercalibMax_(numeric_limits<double>::min()),
  //  minIntercalibRescale_(ps.getParameter<double>("minIntercalibRescale")),
  //maxIntercalibRescale_(ps.getParameter<double>("maxIntercalibRescale")),
  dcc1stSample_(ps.getParameter<int>("dcc1stSample")),
  sampleToSkip_(ps.getParameter<int>("sampleToSkip")),
  nDccWeights_(ps.getParameter<int>("nDccWeights")),
  dccWeightsWithIntercalib_(ps.getParameter<bool>("dccWeightsWithIntercalib")),
  writeToDB_(ps.getParameter<bool>("writeToDB")),
  writeToAsciiFile_(ps.getParameter<bool>("writeToAsciiFile")),
  writeToRootFile_(ps.getParameter<bool>("writeToRootFile")),
  asciiOutputFileName_(ps.getParameter<string>("asciiOutputFileName")),
  rootOutputFileName_(ps.getParameter<string>("rootOutputFileName")),
  calibMap_(emptyCalibMap_)
{}


void
EcalDccWeightBuilder::analyze(const edm::Event& event,
                              const edm::EventSetup& es){
  // Retrieval of intercalib constants
  if(dccWeightsWithIntercalib_){
    ESHandle<EcalIntercalibConstants> hIntercalib ;
    es.get<EcalIntercalibConstantsRcd>().get(hIntercalib) ;
    const EcalIntercalibConstants* intercalib = hIntercalib.product();
    calibMap_ = intercalib->getMap();
  }

  cout << __FILE__ << ":" << __LINE__ << ": "
       <<  endl;
  
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
  for(vector<DetId>::const_iterator it = detIds.begin();
      it != detIds.end(); ++it){
    
    double phase = parameterMap.simParameters(*it).timePhase();

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
#endif
    
    try{
      EcalShape shape(phase);
      
      if(phase!=prevPhase){
        computeWeights(shape, dcc1stSample_-1, nDccWeights_, iSkip0_,
                       baseWeights);
        prevPhase = phase;
      }
      for(int i = 0; i < nw; ++i){
        w[i] = baseWeights[i];
        if(withIntercalib) w[i]*= intercalib(*it);
                           //* intercalibRescale() ;
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
EcalDccWeightBuilder::computeWeights(const EcalShape& shape, int iFirst,
                                     int nWeights, int iSkip,
                                     vector<double>& result){
   double sum2 = 0.;
   double sum = 0;
   result.resize(nWeights);

   int nActualWeights = 0;

   //TO FIX:
   const int binOfMax = 6;
   const double timePhase = 56.1;//ns
   const double tzero = -(binOfMax-1)*25+timePhase;//ns

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

  //sorts weight residuals by amplitude in increasing order:
  vector<int> iw(nw);
  for(unsigned i = 0; i < nw; ++i){
    //TO FIX!!!!!
    iw[i] = i;
  }

  //compensates weight sum residual by adding or substracting 1 to weights
  //starting from:
  // 1) the weight with the minimal signed residual if the correction
  // is positive (wsum<0)
  // 2) the weight with the maximal signed residual if the correction
  // is negative (wsum>0)
  int wsumSign = wsum>0?1:-1;
  int i = wsum>0?0:nw;
  while(wsum!=0){
    W[iw[i]] -= wsumSign;
    wsum -= wsumSign;
    i += wsumSign;
  }

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
  for(map<DetId, std::vector<int32_t> >::const_iterator it
        = encodedWeights_.begin();
      it !=  encodedWeights_.end();
      ++it){
    const DetId& detId = it->first;
    const vector<int>& weights = it->second;
    file << setw(10) << detId.rawId();
    for(unsigned i=0; i<weights.size(); ++i){
      file << " " << setw(5) << weights[i];
    }
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
    Int_t n_weights;
    Int_t weights[nWeightMax];
  } buf;
  t.Branch("weights", &buf, "rawDetId/I:n_weights/I:weights[n_weights]/I");
  for(map<DetId, std::vector<int32_t> >::const_iterator it
        = encodedWeights_.begin();
      it !=  encodedWeights_.end();
      ++it){
    buf.detId = it->first.rawId();
    buf.n_weights = it->second.size();
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

void EcalDccWeightBuilder::writeWeightToDB(){
  cout << "Database export not yet implemented!\n";
}
