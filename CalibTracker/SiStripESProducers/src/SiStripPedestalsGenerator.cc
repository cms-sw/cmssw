#include "CalibTracker/SiStripESProducers/interface/SiStripPedestalsGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <boost/cstdint.hpp>
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

SiStripPedestalsGenerator::SiStripPedestalsGenerator(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg):
  SiStripCondObjBuilderBase<SiStripPedestals>::SiStripCondObjBuilderBase(iConfig)
{
  edm::LogInfo("SiStripPedestalsGenerator") <<  "[SiStripPedestalsGenerator::SiStripPedestalsGenerator]";
}


SiStripPedestalsGenerator::~SiStripPedestalsGenerator() { 
  edm::LogInfo("SiStripPedestalsGenerator") <<  "[SiStripPedestalsGenerator::~SiStripPedestalsGenerator]";
}


void SiStripPedestalsGenerator::createObject(){
    
  obj_ = new SiStripPedestals();

  uint32_t PedestalValue_ = _pset.getParameter<uint32_t>("PedestalsValue");  
  edm::FileInPath fp_ = _pset.getParameter<edm::FileInPath>("file");
  uint32_t  printdebug_ = _pset.getUntrackedParameter<uint32_t>("printDebug", 5);
  uint32_t count=0;

  SiStripDetInfoFileReader reader(fp_.fullPath());

  const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo > DetInfos  = reader.getAllData();
  
  for(std::map<uint32_t, SiStripDetInfoFileReader::DetInfo >::const_iterator it = DetInfos.begin(); it != DetInfos.end(); it++){    
    //Generate Noises for det detid
    SiStripPedestals::InputVector theSiStripVector;
    for(unsigned short j=0; j<128*it->second.nApvs; j++){
  
      if (count<printdebug_) {
	edm::LogInfo("SiStripPedestalsFakeESSource::makePedestals(): ") << "detid: " << it->first  << " strip: " << j <<  " ped: " << PedestalValue_  << std::endl; 	    
      }
      obj_->setData(PedestalValue_,theSiStripVector);
    }
    count++;
    if ( ! obj_->put(it->first, theSiStripVector) )
      edm::LogError("SiStripPedestalsFakeESSource::produce ")<<" detid already exists"<<std::endl;
  }
}
