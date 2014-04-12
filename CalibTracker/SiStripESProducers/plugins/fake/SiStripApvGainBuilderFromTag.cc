#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripApvGainBuilderFromTag.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

SiStripApvGainBuilderFromTag::SiStripApvGainBuilderFromTag( const edm::ParameterSet& iConfig ):
  fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))),
  printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug",1)),
  pset_(iConfig)
{}

void SiStripApvGainBuilderFromTag::analyze(const edm::Event& evt, const edm::EventSetup& iSetup)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
 
  //   unsigned int run=evt.id().run();

  std::string genMode = pset_.getParameter<std::string>("genMode");
  bool applyTuning = pset_.getParameter<bool>("applyTuning");

  double meanGain_=pset_.getParameter<double>("MeanGain");
  double sigmaGain_=pset_.getParameter<double>("SigmaGain");
  double minimumPosValue_=pset_.getParameter<double>("MinPositiveGain");

  uint32_t  printdebug_ = pset_.getUntrackedParameter<uint32_t>("printDebug", 5);

  //parameters for layer/disk level correction; not used if applyTuning=false
  std::map<int, std::vector<double> > correct;
  fillParameters(correct, "correct");

  // Read the gain from the given tag
  edm::ESHandle<SiStripApvGain> inputApvGain;
  iSetup.get<SiStripApvGainRcd>().get( inputApvGain );
  std::vector<uint32_t> inputDetIds;
  inputApvGain->getDetIds(inputDetIds);

  // Prepare the new object
  SiStripApvGain* obj = new SiStripApvGain();

  SiStripDetInfoFileReader reader(fp_.fullPath());

  uint32_t count = 0;
  const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo > DetInfos = reader.getAllData();
  for(std::map<uint32_t, SiStripDetInfoFileReader::DetInfo >::const_iterator it = DetInfos.begin(); it != DetInfos.end(); it++) {

    // Find if this DetId is in the input tag and if so how many are the Apvs for which it contains information
    SiStripApvGain::Range inputRange;
    size_t inputRangeSize = 0;
    if( find( inputDetIds.begin(), inputDetIds.end(), it->first ) != inputDetIds.end() ) {
      inputRange = inputApvGain->getRange(it->first);
      inputRangeSize = distance(inputRange.first, inputRange.second);
    }

    std::vector<float> theSiStripVector;
    for(unsigned short j=0; j<it->second.nApvs; j++){

      double gainValue = meanGain_;

      if( j < inputRangeSize ) {
        gainValue = inputApvGain->getApvGain(j, inputRange);
        // cout << "Gain = " << gainValue <<" from input tag for DetId = " << it->first << " and apv = " << j << endl;
      }
      // else {
      //   cout << "No gain in input tag for DetId = " << it->first << " and apv = " << j << " using value from cfg = " << gainValue << endl;
      // }


      // corrections at layer/disk level:
      uint32_t detId = it->first;
      std::pair<int, int> sl = subDetAndLayer(detId, tTopo);
      //unsigned short nApvs = it->second.nApvs;
      if (applyTuning) {
	double correction = correct[sl.first][sl.second];
	gainValue *= correction;
      }

      // smearing:
      if (genMode == "gaussian") {
	gainValue = CLHEP::RandGauss::shoot(gainValue, sigmaGain_);
	if(gainValue<=minimumPosValue_) gainValue=minimumPosValue_;
      }
      else if( genMode != "default" ) {
        LogDebug("SiStripApvGain") << "ERROR: wrong genMode specifier : " << genMode << ", please select one of \"default\" or \"gaussian\"" << std::endl;
        exit(1);
      }
	
      if (count<printdebug_) {
	edm::LogInfo("SiStripApvGainGeneratorFromTag") << "detid: " << it->first  << " Apv: " << j <<  " gain: " << gainValue  << std::endl; 	    
      }
      theSiStripVector.push_back(gainValue);
    }
    count++;
    SiStripApvGain::Range range(theSiStripVector.begin(),theSiStripVector.end());
    if ( ! obj->put(it->first,range) )
      edm::LogError("SiStripApvGainGeneratorFromTag")<<" detid already exists"<<std::endl;
  }
  
  //End now write data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  
  if( mydbservice.isAvailable() ){
    if( mydbservice->isNewTagRequest("SiStripApvGainRcd2") ){
      mydbservice->createNewIOV<SiStripApvGain>(obj,mydbservice->beginOfTime(),mydbservice->endOfTime(),"SiStripApvGainRcd2");      
    }
    else {
      mydbservice->appendSinceTime<SiStripApvGain>(obj,mydbservice->currentTime(),"SiStripApvGainRcd2");      
    }
  }
  else {
    edm::LogError("SiStripApvGainBuilderFromTag")<<"Service is unavailable"<<std::endl;
  }
}
     
std::pair<int, int> SiStripApvGainBuilderFromTag::subDetAndLayer(const uint32_t detId, const TrackerTopology* tTopo) const
{
  int layerId = 0;

  StripSubdetector subid(detId);
  int subId = subid.subdetId();

  if( subId == int(StripSubdetector::TIB)) {
    layerId = tTopo->tibLayer(detId) - 1;
  }
  else if(subId == int(StripSubdetector::TOB)) {
    layerId = tTopo->tobLayer(detId) - 1;
  }
  else if(subId == int(StripSubdetector::TID)) {
    layerId = tTopo->tidRing(detId) - 1;
  }
  if(subId == int(StripSubdetector::TEC)) {
    layerId = tTopo->tecRing(detId) - 1;
  }
  return std::make_pair(subId, layerId);
}

void SiStripApvGainBuilderFromTag::fillParameters(std::map<int, std::vector<double> > & mapToFill, const std::string & parameterName) const
{
  int layersTIB = 4;
  int ringsTID = 3;
  int layersTOB = 6;
  int ringsTEC = 7;

  fillSubDetParameter( mapToFill, pset_.getParameter<std::vector<double> >(parameterName+"TIB"), int(StripSubdetector::TIB), layersTIB );
  fillSubDetParameter( mapToFill, pset_.getParameter<std::vector<double> >(parameterName+"TID"), int(StripSubdetector::TID), ringsTID );
  fillSubDetParameter( mapToFill, pset_.getParameter<std::vector<double> >(parameterName+"TOB"), int(StripSubdetector::TOB), layersTOB );
  fillSubDetParameter( mapToFill, pset_.getParameter<std::vector<double> >(parameterName+"TEC"), int(StripSubdetector::TEC), ringsTEC );
}

void SiStripApvGainBuilderFromTag::fillSubDetParameter(std::map<int, std::vector<double> > & mapToFill, const std::vector<double> & v, const int subDet, const unsigned short layers) const
{
  if( v.size() == layers ) {
    mapToFill.insert(std::make_pair( subDet, v ));
  }
  else if( v.size() == 1 ) {
    std::vector<double> parV(layers, v[0]);
    mapToFill.insert(std::make_pair( subDet, parV ));
  }
  else {
    throw cms::Exception("Configuration") << "ERROR: number of parameters for subDet " << subDet << " are " << v.size() << ". They must be either 1 or " << layers << std::endl;
  }
}
