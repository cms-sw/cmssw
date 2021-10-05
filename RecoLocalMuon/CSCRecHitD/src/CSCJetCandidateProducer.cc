#include <RecoLocalMuon/CSCRecHitD/src/CSCJetCandidateProducer.h>

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Utilities/interface/Exception.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include "Geometry/Records/interface/MuonGeometryRecord.h"

CSCJetCandidateProducer::CSCJetCandidateProducer(const edm::ParameterSet& ps) {
  cscRechitInputToken_ = consumes<CSCRecHit2DCollection>(edm::InputTag("csc2DRecHits")),

  // register what this produces
      produces<CSCJetCandidateCollection>();
}

CSCJetCandidateProducer::~CSCJetCandidateProducer() {}

void CSCJetCandidateProducer::produce(edm::Event& ev, const edm::EventSetup& setup) {
  LogTrace("CSCRecHit") << "[CSCJetCandidateProducer] starting event ";

  edm::ESHandle<CSCGeometry> cscG;
  edm::Handle<CSCRecHit2DCollection> cscRechits;

  setup.get<MuonGeometryRecord>().get(cscG);
  ev.getByToken(cscRechitInputToken_, cscRechits);

  // Create empty collection of rechits
  auto oc = std::make_unique<CSCJetCandidateCollection>();

  // Put collection in event
  LogTrace("CSCRecHit") << "[CSCJetCandidateProducer] putting collection of " << oc->size() << " rechits into event.";

  for (const CSCRecHit2D& cscRechit : *cscRechits) {
    LocalPoint cscRecHitLocalPosition = cscRechit.localPosition();
    CSCDetId cscdetid = cscRechit.cscDetId();
    int endcap = CSCDetId::endcap(cscdetid) == 1 ? 1 : -1;
    const CSCChamber* cscchamber = cscG->chamber(cscdetid);
    if (cscchamber) {
      GlobalPoint globalPosition = cscchamber->toGlobal(cscRecHitLocalPosition);

      float x = globalPosition.x();
      float y = globalPosition.y();
      float z = globalPosition.z();
      double phi = globalPosition.phi();
      double eta = globalPosition.eta();
      float tpeak = cscRechit.tpeak();
      float wireTime = cscRechit.wireTime();
      int quality = cscRechit.quality();
      int chamber = endcap * (CSCDetId::station(cscdetid) * 10 + CSCDetId::ring(cscdetid));
      int station = endcap * CSCDetId::station(cscdetid);
      int nStrips = cscRechit.nStrips();
      int hitWire = cscRechit.hitWire();
      int wgroupsBX = cscRechit.wgroupsBX();
      int nWireGroups = cscRechit.nWireGroups();

      reco::CSCJetCandidate rh(
          phi, eta, x, y, z, tpeak, wireTime, quality, chamber, station, nStrips, hitWire, wgroupsBX, nWireGroups);
      oc->push_back(rh);
    }
  }
  ev.put(std::move(oc));
}

//define this as a plug-in
DEFINE_FWK_MODULE(CSCJetCandidateProducer);
