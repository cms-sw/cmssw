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

CSCJetCandidateProducer::CSCJetCandidateProducer(const edm::ParameterSet& ps)
{
  cscRechitInputToken_ = consumes<CSCRecHit2DCollection>(edm::InputTag("csc2DRecHits")),

  // register what this produces
  produces<CSCJetCandidateCollection>();
}

CSCJetCandidateProducer::~CSCJetCandidateProducer() {
}

void CSCJetCandidateProducer::produce(edm::Event& ev, const edm::EventSetup& setup) {
  LogTrace("CSCRecHit") << "[CSCJetCandidateProducer] starting event ";

  edm::Handle<CSCRecHit2DCollection> cscRechits;

  ev.getByToken(cscRechitInputToken_,cscRechits);

  // Create empty collection of rechits
  auto oc = std::make_unique<CSCJetCandidateCollection>();

  // Put collection in event
  LogTrace("CSCRecHit") << "[CSCJetCandidateProducer] putting collection of " << oc->size() << " rechits into event.";

  for (const CSCRecHit2D cscRechit : *cscRechits) {
    LocalPoint  cscRecHitLocalPosition       = cscRechit.localPosition();
    CSCDetId cscdetid = cscRechit.cscDetId();
    cscRechitsDetId[ncscRechits] = CSCDetId::rawIdMaker(CSCDetId::endcap(cscdetid), CSCDetId::station(cscdetid), CSCDetId::ring(cscdetid), CSCDetId::chamber(cscdetid), CSCDetId::layer(cscdetid));
    int endcap = CSCDetId::endcap(cscdetid) == 1 ? 1 : -1;
    const CSCChamber* cscchamber = cscG->chamber(cscdetid);
    if (cscchamber) {
        GlobalPoint globalPosition = cscchamber->toGlobal(cscRecHitLocalPosition);

        double x = globalPosition.x();
        double y = globalPosition.y();
        double z = globalPosition.z();
        double phi = globalPosition.phi();
        double eta = globalPosition.eta();
        double tpeak    = cscRechit.tpeak();
        double wireTime = cscRechit.wireTime();

        CSCJetCandidate rh = (phi, eta, x ,y,z,tpeak,wireTime);
        oc.push_back(rh)
    }
  }
  ev.put(std::move(oc));
}


//define this as a plug-in
DEFINE_FWK_MODULE(CSCJetCandidateProducer);
