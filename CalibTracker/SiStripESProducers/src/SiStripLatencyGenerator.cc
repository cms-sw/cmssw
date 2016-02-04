#include "CalibTracker/SiStripESProducers/interface/SiStripLatencyGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <boost/cstdint.hpp>
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

SiStripLatencyGenerator::SiStripLatencyGenerator(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg):
  SiStripCondObjBuilderBase<SiStripLatency>::SiStripCondObjBuilderBase(iConfig)
{
  edm::LogInfo("SiStripLatencyGenerator") << "[SiStripLatencyGenerator::SiStripLatencyGenerator]";
}

SiStripLatencyGenerator::~SiStripLatencyGenerator()
{
  edm::LogInfo("SiStripLatencyGenerator") << "[SiStripLatencyGenerator::~SiStripLatencyGenerator]";
}

void SiStripLatencyGenerator::createObject()
{
  obj_ = new SiStripLatency();

  // Read the full list of detIds
  edm::FileInPath fp_ = _pset.getParameter<edm::FileInPath>("file");
  SiStripDetInfoFileReader reader(fp_.fullPath());
  const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo > detInfos = reader.getAllData();
  // Take the last detId. Since the map is sorted it will be the biggest value
  if( !detInfos.empty() ) {
    // Set the apv number as 6, the highest possible
    edm::LogInfo("SiStripLatencyGenerator") << "detId = " << detInfos.rbegin()->first << " apv = " << 6
                                            << " latency = " << _pset.getParameter<uint32_t>("latency")
                                            << " mode = " << _pset.getParameter<uint32_t>("mode") << std::endl;
    obj_->put(detInfos.rbegin()->first, 6, _pset.getParameter<uint32_t>("latency"), _pset.getParameter<uint32_t>("mode") );

    // Call this method to collapse all consecutive detIdAndApvs with the same latency and mode to a single entry
    obj_->compress();
  }
  else {
    edm::LogError("SiStripLatencyGenerator") << "Error: detInfo map is empty. Cannot get the last detId." << std::endl;
  }
}
