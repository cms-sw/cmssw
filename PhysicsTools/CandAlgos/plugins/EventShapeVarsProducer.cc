#include "PhysicsTools/CandAlgos/plugins/EventShapeVarsProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"

#include "PhysicsTools/CandUtils/interface/EventShapeVariables.h"
#include "PhysicsTools/CandUtils/interface/Thrust.h"

EventShapeVarsProducer::EventShapeVarsProducer(const edm::ParameterSet& cfg)
{
  src_ = cfg.getParameter<edm::InputTag>("src");
  r_ = cfg.exists("r") ? cfg.getParameter<double>("r") : 2.;

  produces<double>("thrust");
  //produces<double>("oblateness");
  produces<double>("isotropy");
  produces<double>("circularity");
  produces<double>("sphericity");
  produces<double>("aplanarity");
  produces<double>("C");
  produces<double>("D");
  
}

void put(edm::Event& evt, double value, const char* instanceName)
{
  std::auto_ptr<double> eventShapeVarPtr(new double(value));
  evt.put(eventShapeVarPtr, instanceName);
}

void EventShapeVarsProducer::produce(edm::Event& evt, const edm::EventSetup&) 
{ 
  //std::cout << "<EventShapeVarsProducer::produce>:" << std::endl;

  edm::Handle<edm::View<reco::Candidate> > objects;
  evt.getByLabel(src_, objects);

  Thrust thrustAlgo(objects->begin(), objects->end());
  put(evt, thrustAlgo.thrust(), "thrust");
  //put(evt, thrustAlgo.oblateness(), "oblateness");
  
  EventShapeVariables eventShapeVarsAlgo(*objects);
  put(evt, eventShapeVarsAlgo.isotropy(), "isotropy");
  put(evt, eventShapeVarsAlgo.circularity(), "circularity");
  put(evt, eventShapeVarsAlgo.sphericity(r_), "sphericity");
  put(evt, eventShapeVarsAlgo.aplanarity(r_), "aplanarity");
  put(evt, eventShapeVarsAlgo.C(r_), "C");
  put(evt, eventShapeVarsAlgo.D(r_), "D");
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(EventShapeVarsProducer);
