#include "PhysicsTools/CandAlgos/plugins/EventShapeVarsProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/CandUtils/interface/Thrust.h"

#include <vector>
#include <memory>

EventShapeVarsProducer::EventShapeVarsProducer(const edm::ParameterSet& cfg) {
  srcToken_ = consumes<edm::View<reco::Candidate>>(cfg.getParameter<edm::InputTag>("src"));
  r_ = cfg.exists("r") ? cfg.getParameter<double>("r") : 2.;
  fwmax_ = cfg.exists("fwmax") ? cfg.getParameter<unsigned>("fwmax") : 0;

  produces<double>("thrust");
  //produces<double>("oblateness");
  produces<double>("isotropy");
  produces<double>("circularity");
  produces<double>("sphericity");
  produces<double>("aplanarity");
  produces<double>("C");
  produces<double>("D");
  if (fwmax_ > 0)
    produces<std::vector<double>>("FWmoments");
}

void put(edm::Event& evt, double value, const char* instanceName) {
  evt.put(std::make_unique<double>(value), instanceName);
}

void EventShapeVarsProducer::produce(edm::Event& evt, const edm::EventSetup&) {
  edm::Handle<edm::View<reco::Candidate>> objects;
  evt.getByToken(srcToken_, objects);

  Thrust thrustAlgo(objects->begin(), objects->end());
  put(evt, thrustAlgo.thrust(), "thrust");
  //put(evt, thrustAlgo.oblateness(), "oblateness");

  EventShapeVariables eventShapeVarsAlgo(*objects);
  eventShapeVarsAlgo.set_r(r_);
  put(evt, eventShapeVarsAlgo.isotropy(), "isotropy");
  put(evt, eventShapeVarsAlgo.circularity(), "circularity");
  put(evt, eventShapeVarsAlgo.sphericity(), "sphericity");
  put(evt, eventShapeVarsAlgo.aplanarity(), "aplanarity");
  put(evt, eventShapeVarsAlgo.C(), "C");
  put(evt, eventShapeVarsAlgo.D(), "D");
  if (fwmax_ > 0) {
    eventShapeVarsAlgo.setFWmax(fwmax_);
    auto vfw = std::make_unique<std::vector<double>>(eventShapeVarsAlgo.getFWmoments());
    evt.put(std::move(vfw), "FWmoments");
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(EventShapeVarsProducer);
