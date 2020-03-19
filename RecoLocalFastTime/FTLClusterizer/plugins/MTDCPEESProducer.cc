#include "RecoLocalFastTime/Records/interface/MTDCPERecord.h"
#include "RecoLocalFastTime/FTLClusterizer/interface/MTDClusterParameterEstimator.h"
#include "RecoLocalFastTime/FTLClusterizer/interface/MTDCPEBase.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <memory>

using namespace edm;

class MTDCPEESProducer : public edm::ESProducer {
public:
  MTDCPEESProducer(const edm::ParameterSet& p);
  ~MTDCPEESProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<MTDClusterParameterEstimator> produce(const MTDCPERecord&);

private:
  edm::ParameterSet pset_;
  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> ddToken_;
};

MTDCPEESProducer::MTDCPEESProducer(const edm::ParameterSet& p) {
  pset_ = p;
  setWhatProduced(this, "MTDCPEBase").setConsumes(ddToken_);
}

// Configuration descriptions
void MTDCPEESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("MTDCPEESProducer", desc);
}

std::unique_ptr<MTDClusterParameterEstimator> MTDCPEESProducer::produce(const MTDCPERecord& iRecord) {
  return std::make_unique<MTDCPEBase>(pset_, iRecord.get(ddToken_));
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_EVENTSETUP_MODULE(MTDCPEESProducer);
