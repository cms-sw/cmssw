#include "CalibTracker/SiStripChannelGain/plugins/SiStripGainFromAsciiFile.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h" 
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

#include <iostream>
#include <fstream>


SiStripGainFromAsciiFile::SiStripGainFromAsciiFile(const edm::ParameterSet& iConfig) : ConditionDBWriter<SiStripApvGain>::ConditionDBWriter<SiStripApvGain>(iConfig){

  edm::LogInfo("SiStripGainFromAsciiFile::SiStripGainFromAsciiFile");

  detid_apvs_.clear();

  Asciifilename_=iConfig.getParameter<std::string>("InputFileName");
  referenceValue_ = iConfig.getParameter<double>("referenceValue");
}


SiStripGainFromAsciiFile::~SiStripGainFromAsciiFile(){
  edm::LogInfo("SiStripGainFromAsciiFile::~SiStripGainFromAsciiFile");
}

void SiStripGainFromAsciiFile::algoAnalyze(const edm::Event & event, const edm::EventSetup& iSetup){
  
  edm::LogInfo("SiStripGainFromAsciiFile") <<"SiStripGainFromAsciiFile::getNewObject called"<<std::endl;

  obj = new SiStripApvGain();

  uint32_t detid;
  FibersGain FG;

  std::ifstream infile;
  infile.open (Asciifilename_.c_str());
  if (infile.is_open()){
    while (infile.good()){
      infile >> detid >> FG.fiber[0] >> FG.fiber[1] >> FG.fiber[2];
      edm::LogInfo("SiStripGainFromAsciiFile" ) << detid << " " <<  FG.fiber[0] << " " <<  FG.fiber[1] << " " <<  FG.fiber[2] << std::endl;
      GainsMap.insert(std::pair<unsigned int,FibersGain>(detid,FG));
    }
    infile.close();
  } else  {
    edm::LogError("SiStripGainFromAsciiFile")<< "Error opening file";
  }
  
  edm::ESHandle<SiStripDetCabling> DetCabling;
  
  iSetup.get<SiStripDetCablingRcd>().get( DetCabling );
  edm::LogInfo("SiStripGainFromAsciiFile" ) << "algoAnalyze - got cabling "<<std::endl;
  
  std::vector<uint32_t> vdetId_;
  DetCabling->addActiveDetectorsRawIds(vdetId_);
  
  short nApvPair;

  for(std::vector<uint32_t>::const_iterator it = vdetId_.begin(); it != vdetId_.end(); ++it){

    if (DetId(*it).det()!=DetId::Tracker)
      continue;

    nApvPair=DetCabling->nApvPairs(*it);
    
    edm::LogInfo("SiStripGainFromAsciiFile" ) << "Looking at detid " << *it << " nApvPairs  " << nApvPair << std::endl;

    __gnu_cxx::hash_map< unsigned int,FibersGain>::const_iterator iter=GainsMap.find(*it);

    if (iter!=GainsMap.end()){
      FG = iter->second;
      edm::LogInfo("SiStripGainFromAsciiFile" )<< *it << " " <<  FG.fiber[0] << " " <<  FG.fiber[1] << " " <<  FG.fiber[2] << std::endl;
    }
    else {
      edm::LogInfo("SiStripGainFromAsciiFile" )<< "Hard reset for detid " << *it << std::endl;
      FG.hard_reset(referenceValue_);
    }
    
    std::vector<float> DetGainsVector;
    
    if (nApvPair==2){
      DetGainsVector.push_back(FG.fiber[0]/referenceValue_);  
      DetGainsVector.push_back(FG.fiber[2]/referenceValue_);  
    } else if (nApvPair==3){   		   		   		  
      DetGainsVector.push_back(FG.fiber[0]/referenceValue_);  
      DetGainsVector.push_back(FG.fiber[1]/referenceValue_);  
      DetGainsVector.push_back(FG.fiber[2]/referenceValue_);  
    } else {
      edm::LogError("SiStripGainFromAsciiFile") << " ERROR for detid " << *it << " not expected number of APV pairs " << nApvPair <<std::endl;
    }
    
    SiStripApvGain::Range range(DetGainsVector.begin(),DetGainsVector.end());
    if ( ! obj->put(*it,range) )
      edm::LogError("SiStripGainCalculator")<<"[SiStripGainCalculator::beginJob] detid already exists"<<std::endl;
  }
}



