#include "sstream"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/L1TObjects/interface/L1TMuonOverlapFwVersion.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapFwVersionRcd.h"

#include "L1Trigger/L1TMuonOverlap/plugins/L1TMuonOverlapFwVersionESProducer.h"

L1TMuonOverlapFwVersionESProducer::L1TMuonOverlapFwVersionESProducer(const edm::ParameterSet& theConfig) {
  setWhatProduced(this, &L1TMuonOverlapFwVersionESProducer::produceFwVersion);

  unsigned algoV = theConfig.getParameter<unsigned>("algoVersion");
  unsigned layersV = theConfig.getParameter<unsigned>("layersVersion");
  unsigned patternsV = theConfig.getParameter<unsigned>("patternsVersion");
  std::string sDate = theConfig.getParameter<std::string>("synthDate");
  params.setAlgoVersion(algoV);
  params.setLayersVersion(layersV);
  params.setPatternsVersion(patternsV);
  params.setSynthDate(sDate);
}

L1TMuonOverlapFwVersionESProducer::~L1TMuonOverlapFwVersionESProducer() {}

L1TMuonOverlapFwVersionESProducer::ReturnType L1TMuonOverlapFwVersionESProducer::produceFwVersion(
    const L1TMuonOverlapFwVersionRcd& iRecord) {
  return std::make_unique<L1TMuonOverlapFwVersion>(params);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonOverlapFwVersionESProducer);
