#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "FastSimDataFormats/NuclearInteractions/interface/FSimVertexTypeFwd.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Provenance/interface/EventID.h"

#include "FastSimulation/EventProducer/interface/FamosProducer.h"
#include "FastSimulation/EventProducer/interface/FamosManager.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/KineParticleFilter.h"
#include "FastSimulation/Calorimetry/interface/CalorimetryManager.h"
#include "FastSimulation/TrajectoryManager/interface/TrajectoryManager.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "HepMC/GenVertex.h"
#include "HepMC/GenEvent.h"

#include <iostream>
#include <memory>
#include <vector>

FamosProducer::FamosProducer(edm::ParameterSet const& p) {
  produces<edm::SimTrackContainer>();
  produces<edm::SimVertexContainer>();
  produces<FSimVertexTypeCollection>("VertexTypes");
  produces<edm::PSimHitContainer>("TrackerHits");
  produces<edm::PCaloHitContainer>("EcalHitsEB");
  produces<edm::PCaloHitContainer>("EcalHitsEE");
  produces<edm::PCaloHitContainer>("EcalHitsES");
  produces<edm::PCaloHitContainer>("HcalHits");
  // Temporary facility to allow for the crossing frame to work...
  simulateMuons = p.getParameter<bool>("SimulateMuons");
  if (simulateMuons)
    produces<edm::SimTrackContainer>("MuonSimTracks");

  // hepmc event from signal event
  edm::InputTag sourceLabel = p.getParameter<edm::InputTag>("SourceLabel");
  sourceToken = consumes<edm::HepMCProduct>(sourceLabel);

  // famos manager
  famosManager_ = new FamosManager(p);
}

FamosProducer::~FamosProducer() { delete famosManager_; }

void FamosProducer::beginRun(edm::Run const& run, const edm::EventSetup& es) {
  famosManager_->setupGeometryAndField(run, es);
}

void FamosProducer::produce(edm::Event& iEvent, const edm::EventSetup& es) {
  using namespace edm;

  RandomEngineAndDistribution random(iEvent.streamID());

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  es.get<TrackerTopologyRcd>().get(tTopoHand);
  const TrackerTopology* tTopo = tTopoHand.product();

  // get the signal event
  Handle<HepMCProduct> theHepMCProduct;
  iEvent.getByToken(sourceToken, theHepMCProduct);
  const HepMC::GenEvent* myGenEvent = theHepMCProduct->GetEvent();

  // do the simulation
  famosManager_->reconstruct(myGenEvent, tTopo, &random);

  // get the hits, simtracks and simvertices and put in the event
  CalorimetryManager* calo = famosManager_->calorimetryManager();
  TrajectoryManager* tracker = famosManager_->trackerManager();

  std::unique_ptr<edm::SimTrackContainer> p1(new edm::SimTrackContainer);
  std::unique_ptr<edm::SimTrackContainer> m1(new edm::SimTrackContainer);
  std::unique_ptr<edm::SimVertexContainer> p2(new edm::SimVertexContainer);
  std::unique_ptr<FSimVertexTypeCollection> v1(new FSimVertexTypeCollection);
  std::unique_ptr<edm::PSimHitContainer> p3(new edm::PSimHitContainer);
  std::unique_ptr<edm::PCaloHitContainer> p4(new edm::PCaloHitContainer);
  std::unique_ptr<edm::PCaloHitContainer> p5(new edm::PCaloHitContainer);
  std::unique_ptr<edm::PCaloHitContainer> p6(new edm::PCaloHitContainer);
  std::unique_ptr<edm::PCaloHitContainer> p7(new edm::PCaloHitContainer);

  FSimEvent* fevt = famosManager_->simEvent();
  fevt->load(*p1, *m1);
  fevt->load(*p2);
  fevt->load(*v1);
  tracker->loadSimHits(*p3);

  if (calo) {
    calo->loadFromEcalBarrel(*p4);
    calo->loadFromEcalEndcap(*p5);
    calo->loadFromPreshower(*p6);
    calo->loadFromHcal(*p7);
    calo->loadMuonSimTracks(*m1);
  }

  if (simulateMuons)
    iEvent.put(std::move(m1), "MuonSimTracks");
  iEvent.put(std::move(p1));
  iEvent.put(std::move(p2));
  iEvent.put(std::move(p3), "TrackerHits");
  iEvent.put(std::move(v1), "VertexTypes");
  iEvent.put(std::move(p4), "EcalHitsEB");
  iEvent.put(std::move(p5), "EcalHitsEE");
  iEvent.put(std::move(p6), "EcalHitsES");
  iEvent.put(std::move(p7), "HcalHits");
}

DEFINE_FWK_MODULE(FamosProducer);
