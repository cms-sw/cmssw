#include "CalibTracker/SiStripESProducers/interface/SiStripBaseDelayGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <boost/cstdint.hpp>
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

SiStripBaseDelayGenerator::SiStripBaseDelayGenerator(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg):
  SiStripCondObjBuilderBase<SiStripBaseDelay>::SiStripCondObjBuilderBase(iConfig)
{
  edm::LogInfo("SiStripBaseDelayGenerator") << "[SiStripBaseDelayGenerator::SiStripBaseDelayGenerator]";
}

SiStripBaseDelayGenerator::~SiStripBaseDelayGenerator()
{
  edm::LogInfo("SiStripBaseDelayGenerator") << "[SiStripBaseDelayGenerator::~SiStripBaseDelayGenerator]";
}

void SiStripBaseDelayGenerator::createObject()
{
  obj_ = new SiStripBaseDelay();

  // Read the full list of detIds
  edm::FileInPath fp_ = _pset.getParameter<edm::FileInPath>("file");
  uint16_t coarseDelay = _pset.getParameter<uint32_t>("CoarseDelay");
  uint16_t fineDelay = _pset.getParameter<uint32_t>("FineDelay");
  SiStripDetInfoFileReader reader(fp_.fullPath());
  const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo > detInfos = reader.getAllData();
  if( !detInfos.empty() ) {
    std::map<uint32_t, SiStripDetInfoFileReader::DetInfo>::const_iterator it = detInfos.begin();
    for( ; it != detInfos.end(); ++it ) {
      obj_->put(it->first, coarseDelay, fineDelay);
    }
  }
  else {
    edm::LogError("SiStripBaseDelayGenerator") << "Error: detInfo map is empty." << std::endl;
  }
}
