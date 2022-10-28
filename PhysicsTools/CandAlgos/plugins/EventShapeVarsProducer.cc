/** \class EventShapeVarsProducer
 *
 * Produce set of event shape variables.
 * The values of different event shape variables are stored as doubles in the event.
 * They can be retrieved with InputTags like "moduleName::instanceName", where moduleName corresponds to
 * "eventShapeVarsProducer" per default and instance name specifies an individual event shape variable
 * which you wish to retrieve from the event:
 *
 *  - thrust
 *  - oblateness
 *  - isotropy
 *  - circularity
 *  - sphericity
 *  - aplanarity
 *  - C
 *  - D
 *  - Fox-Wolfram moments
 *
 *  See https://arxiv.org/pdf/hep-ph/0603175v2.pdf#page=524
 *  for an explanation of sphericity, aplanarity, the quantities C and D, thrust, oblateness, Fox-Wolfram moments.
 *
 * \author Christian Veelken, UC Davis
 *
 *
 *
 */

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "PhysicsTools/CandUtils/interface/EventShapeVariables.h"
#include "PhysicsTools/CandUtils/interface/Thrust.h"

#include <memory>
#include <vector>

class EventShapeVarsProducer : public edm::global::EDProducer<> {
public:
  explicit EventShapeVarsProducer(const edm::ParameterSet&);

private:
  edm::EDGetTokenT<edm::View<reco::Candidate>> srcToken_;
  double r_;
  unsigned fwmax_;

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
};

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

void EventShapeVarsProducer::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup&) const {
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
