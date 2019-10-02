
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "RecoLocalTracker/Records/interface/TkPhase2OTCPERecord.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/SiPhase2VectorHitBuilder/interface/VectorHitBuilderEDProducer.h"
#include <memory>

class SiPhase2RecHitMatcherESProducer : public edm::ESProducer {
public:
  SiPhase2RecHitMatcherESProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  std::shared_ptr<VectorHitBuilderEDProducer> produce(const TkPhase2OTCPERecord&);

private:
  std::string name_;
  std::shared_ptr<VectorHitBuilderEDProducer> matcher_;
  edm::ParameterSet pset_;
};

SiPhase2RecHitMatcherESProducer::SiPhase2RecHitMatcherESProducer(const edm::ParameterSet& p) {
  name_ = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this, name_);
}

std::shared_ptr<VectorHitBuilderEDProducer> SiPhase2RecHitMatcherESProducer::produce(
    const TkPhase2OTCPERecord& iRecord) {
  if (name_ == "SiPhase2VectorHitMatcher") {
    matcher_ = std::make_shared<VectorHitBuilderEDProducer>(pset_);

    edm::ESHandle<TrackerGeometry> tGeomHandle;
    edm::ESHandle<TrackerTopology> tTopoHandle;

    iRecord.getRecord<TrackerDigiGeometryRecord>().get(tGeomHandle);
    iRecord.getRecord<TrackerTopologyRcd>().get(tTopoHandle);

    matcher_->algo()->initTkGeom(tGeomHandle);
    matcher_->algo()->initTkTopo(tTopoHandle);
  }
  return matcher_;
}

void SiPhase2RecHitMatcherESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("offlinestubs", "vectorHits");
  desc.add<int>("maxVectorHits", 999999999);
  desc.add<std::string>("Algorithm", "VectorHitBuilderAlgorithm");
  desc.add<std::string>("ComponentName", "SiPhase2VectorHitMatcher");
  desc.add<edm::ESInputTag>("CPE", edm::ESInputTag("phase2StripCPEESProducer", "Phase2StripCPE"));
  desc.add<std::vector<double>>("BarrelCut",
                                {
                                    0.0,
                                    0.05,
                                    0.06,
                                    0.08,
                                    0.09,
                                    0.12,
                                    0.2,
                                });
  desc.add<std::string>("Phase2CPE_name", "Phase2StripCPE");
  desc.add<std::string>("Clusters", "siPhase2Clusters");
  desc.add<int>("maxVectorHitsInAStack", 999);
  desc.add<std::vector<double>>("EndcapCut",
                                {
                                    0.0,
                                    0.1,
                                    0.1,
                                    0.1,
                                    0.1,
                                    0.1,
                                });
  descriptions.add("SiPhase2RecHitMatcherESProducer", desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(SiPhase2RecHitMatcherESProducer);
