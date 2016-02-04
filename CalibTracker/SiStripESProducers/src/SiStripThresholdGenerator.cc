#include "CalibTracker/SiStripESProducers/interface/SiStripThresholdGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <boost/cstdint.hpp>
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

SiStripThresholdGenerator::SiStripThresholdGenerator(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg):
  SiStripCondObjBuilderBase<SiStripThreshold>::SiStripCondObjBuilderBase(iConfig)
{
  edm::LogInfo("SiStripThresholdGenerator") <<  "[SiStripThresholdGenerator::SiStripThresholdGenerator]";
}


SiStripThresholdGenerator::~SiStripThresholdGenerator() { 
  edm::LogInfo("SiStripThresholdGenerator") <<  "[SiStripThresholdGenerator::~SiStripThresholdGenerator]";
}


void SiStripThresholdGenerator::createObject(){
    
  obj_ = new SiStripThreshold();

  edm::FileInPath fp_ = _pset.getParameter<edm::FileInPath>("file");
  float lTh_ = _pset.getParameter<double>("LowTh");
  float hTh_ = _pset.getParameter<double>("HighTh");
  float cTh_ = _pset.getParameter<double>("ClusTh");

  SiStripDetInfoFileReader reader(fp_.fullPath());
 
  const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo > DetInfos  = reader.getAllData();
  
  for(std::map<uint32_t, SiStripDetInfoFileReader::DetInfo >::const_iterator it = DetInfos.begin(); it != DetInfos.end(); it++){    
    //Generate Thresholds for det detid
    SiStripThreshold::Container theSiStripVector;   
    uint16_t strip=0;

    obj_->setData(strip,lTh_,hTh_,cTh_,theSiStripVector);
    LogDebug("SiStripThresholdFakeESSource::produce") <<"detid: "  << it->first << " \t"
						      << "firstStrip: " << strip << " \t" << theSiStripVector.back().getFirstStrip() << " \t"
						      << "lTh: " << lTh_       << " \t" << theSiStripVector.back().getLth() << " \t"
						      << "hTh: " << hTh_       << " \t" << theSiStripVector.back().getHth() << " \t"
						      << "FirstStrip_and_Hth: " << theSiStripVector.back().FirstStrip_and_Hth << " \t"
						      << std::endl; 	    
    
    if ( ! obj_->put(it->first,theSiStripVector) )
      edm::LogError("SiStripThresholdFakeESSource::produce ")<<" detid already exists"<<std::endl;
  }
  
  std::stringstream ss;
  obj_->printDebug(ss);
  LogDebug("SiStripThresholdFakeESSource::produce") <<"printDebug\n" << ss.str() << std::endl;

}
