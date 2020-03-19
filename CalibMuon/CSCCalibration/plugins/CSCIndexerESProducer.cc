#include "FWCore/Framework/interface/ModuleFactory.h"

#include "CalibMuon/CSCCalibration/interface/CSCIndexerESProducer.h"
#include "CalibMuon/CSCCalibration/interface/CSCIndexerFactory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

CSCIndexerESProducer::CSCIndexerESProducer(const edm::ParameterSet &pset) {
  algoName = pset.getParameter<std::string>("AlgoName");

  LogTrace("CSCIndexerESProducer") << " will produce: " << algoName;

  setWhatProduced(this);
}

CSCIndexerESProducer::~CSCIndexerESProducer() {}

CSCIndexerESProducer::BSP_TYPE CSCIndexerESProducer::produce(const CSCIndexerRecord &) {
  LogTrace("CSCIndexerESProducer") << " producing: " << algoName;

  return CSCIndexerESProducer::BSP_TYPE(CSCIndexerFactory::get()->create(algoName));
}

// define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(CSCIndexerESProducer);
