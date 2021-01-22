#include <iostream>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "HLTDisplacedmumuVtxProducer.h"

using namespace edm;
using namespace reco;
using namespace std;
using namespace trigger;
//
// constructors and destructor
//
HLTDisplacedmumuVtxProducer::HLTDisplacedmumuVtxProducer(const edm::ParameterSet& iConfig)
    : srcTag_(iConfig.getParameter<edm::InputTag>("Src")),
      srcToken_(consumes<reco::RecoChargedCandidateCollection>(srcTag_)),
      previousCandTag_(iConfig.getParameter<edm::InputTag>("PreviousCandTag")),
      previousCandToken_(consumes<trigger::TriggerFilterObjectWithRefs>(previousCandTag_)),
      matchToPrevious_(iConfig.getParameter<bool>("matchToPrevious")),
      maxEta_(iConfig.getParameter<double>("MaxEta")),
      minPt_(iConfig.getParameter<double>("MinPt")),
      minPtPair_(iConfig.getParameter<double>("MinPtPair")),
      minInvMass_(iConfig.getParameter<double>("MinInvMass")),
      maxInvMass_(iConfig.getParameter<double>("MaxInvMass")),
      chargeOpt_(iConfig.getParameter<int>("ChargeOpt")) {
  produces<VertexCollection>();
}

HLTDisplacedmumuVtxProducer::~HLTDisplacedmumuVtxProducer() = default;

void HLTDisplacedmumuVtxProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("Src", edm::InputTag("hltL3MuonCandidates"));
  desc.add<edm::InputTag>("PreviousCandTag", edm::InputTag(""));
  desc.add<bool>("matchToPrevious", true);
  desc.add<double>("MaxEta", 2.5);
  desc.add<double>("MinPt", 0.0);
  desc.add<double>("MinPtPair", 0.0);
  desc.add<double>("MinInvMass", 1.0);
  desc.add<double>("MaxInvMass", 20.0);
  desc.add<int>("ChargeOpt", -1);
  descriptions.add("hltDisplacedmumuVtxProducer", desc);
}

// ------------ method called on each new Event  ------------
void HLTDisplacedmumuVtxProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  double const MuMass = 0.106;
  double const MuMass2 = MuMass * MuMass;

  // get hold of muon trks
  Handle<RecoChargedCandidateCollection> mucands;
  iEvent.getByToken(srcToken_, mucands);

  //get the transient track builder:
  edm::ESHandle<TransientTrackBuilder> theB;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", theB);

  std::unique_ptr<VertexCollection> vertexCollection(new VertexCollection());

  // look at all mucands,  check cuts and make vertices
  double e1, e2;
  Particle::LorentzVector p, p1, p2;

  RecoChargedCandidateCollection::const_iterator cand1;
  RecoChargedCandidateCollection::const_iterator cand2;

  // get the objects passing the previous filter
  Handle<TriggerFilterObjectWithRefs> previousCands;
  if (matchToPrevious_)
    iEvent.getByToken(previousCandToken_, previousCands);

  vector<RecoChargedCandidateRef> vPrevCands;
  if (matchToPrevious_)
    previousCands->getObjects(TriggerMuon, vPrevCands);

  for (cand1 = mucands->begin(); cand1 != mucands->end(); cand1++) {
    TrackRef tk1 = cand1->get<TrackRef>();
    LogDebug("HLTDisplacedMumuFilter") << " 1st muon in loop: q*pt= " << cand1->charge() * cand1->pt()
                                       << ", eta= " << cand1->eta() << ", hits= " << tk1->numberOfValidHits();

    //first check if this muon passed the previous filter
    if (matchToPrevious_ && !checkPreviousCand(tk1, vPrevCands))
      continue;

    // cuts
    if (fabs(cand1->eta()) > maxEta_)
      continue;
    if (cand1->pt() < minPt_)
      continue;

    cand2 = cand1;
    cand2++;
    for (; cand2 != mucands->end(); cand2++) {
      TrackRef tk2 = cand2->get<TrackRef>();

      // eta cut
      LogDebug("HLTDisplacedmumuVtxProducer")
          << " 2nd muon in loop: q*pt= " << cand2->charge() * cand2->pt() << ", eta= " << cand2->eta()
          << ", hits= " << tk2->numberOfValidHits() << ", d0= " << tk2->d0();
      //first check if this muon passed the previous filter
      if (matchToPrevious_ && !checkPreviousCand(tk2, vPrevCands))
        continue;

      // cuts
      if (fabs(cand2->eta()) > maxEta_)
        continue;
      if (cand2->pt() < minPt_)
        continue;

      // opposite sign or same sign
      if (chargeOpt_ < 0) {
        if (cand1->charge() * cand2->charge() > 0)
          continue;
      } else if (chargeOpt_ > 0) {
        if (cand1->charge() * cand2->charge() < 0)
          continue;
      }

      // Combined dimuon system
      e1 = sqrt(cand1->momentum().Mag2() + MuMass2);
      e2 = sqrt(cand2->momentum().Mag2() + MuMass2);
      p1 = Particle::LorentzVector(cand1->px(), cand1->py(), cand1->pz(), e1);
      p2 = Particle::LorentzVector(cand2->px(), cand2->py(), cand2->pz(), e2);
      p = p1 + p2;

      if (p.pt() < minPtPair_)
        continue;

      double invmass = abs(p.mass());
      LogDebug("HLTDisplacedMumuFilter") << " ... 1-2 invmass= " << invmass;

      if (invmass < minInvMass_)
        continue;
      if (invmass > maxInvMass_)
        continue;

      // do the vertex fit
      vector<TransientTrack> t_tks;
      TransientTrack ttkp1 = (*theB).build(&tk1);
      TransientTrack ttkp2 = (*theB).build(&tk2);
      t_tks.push_back(ttkp1);
      t_tks.push_back(ttkp2);

      if (t_tks.size() != 2)
        continue;

      KalmanVertexFitter kvf;
      TransientVertex tv = kvf.vertex(t_tks);

      if (!tv.isValid())
        continue;

      Vertex vertex = tv;

      // put vertex in the event
      vertexCollection->push_back(vertex);
    }
  }
  iEvent.put(std::move(vertexCollection));
}

bool HLTDisplacedmumuVtxProducer::checkPreviousCand(const TrackRef& trackref,
                                                    const vector<RecoChargedCandidateRef>& refVect) const {
  bool ok = false;
  for (auto& i : refVect) {
    if (i->get<TrackRef>() == trackref) {
      ok = true;
      break;
    }
  }
  return ok;
}
