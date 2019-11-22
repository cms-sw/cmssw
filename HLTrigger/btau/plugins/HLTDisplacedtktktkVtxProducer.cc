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

#include "HLTDisplacedtktktkVtxProducer.h"

using namespace edm;
using namespace reco;
using namespace std;
using namespace trigger;
//
// constructors and destructor
//
HLTDisplacedtktktkVtxProducer::HLTDisplacedtktktkVtxProducer(const edm::ParameterSet& iConfig)
    : srcTag_(iConfig.getParameter<edm::InputTag>("Src")),
      srcToken_(consumes<reco::RecoChargedCandidateCollection>(srcTag_)),
      previousCandTag_(iConfig.getParameter<edm::InputTag>("PreviousCandTag")),
      previousCandToken_(consumes<trigger::TriggerFilterObjectWithRefs>(previousCandTag_)),
      maxEta_(iConfig.getParameter<double>("MaxEtaTk")),
      minPtTk1_(iConfig.getParameter<double>("MinPtResTk1")),
      minPtTk2_(iConfig.getParameter<double>("MinPtResTk2")),
      minPtTk3_(iConfig.getParameter<double>("MinPtThirdTk")),
      minPtRes_(iConfig.getParameter<double>("MinPtRes")),
      minPtTri_(iConfig.getParameter<double>("MinPtTri")),
      minInvMassRes_(iConfig.getParameter<double>("MinInvMassRes")),
      maxInvMassRes_(iConfig.getParameter<double>("MaxInvMassRes")),
      minInvMass_(iConfig.getParameter<double>("MinInvMass")),
      maxInvMass_(iConfig.getParameter<double>("MaxInvMass")),
      massParticle1_(iConfig.getParameter<double>("massParticleRes1")),
      massParticle2_(iConfig.getParameter<double>("massParticleRes2")),
      massParticle3_(iConfig.getParameter<double>("massParticle3")),
      chargeOpt_(iConfig.getParameter<int>("ChargeOpt")),
      resOpt_(iConfig.getParameter<int>("ResOpt")),
      triggerTypeDaughters_(iConfig.getParameter<int>("triggerTypeDaughters"))

{
  produces<VertexCollection>();

  firstTrackMass = massParticle1_;
  secondTrackMass = massParticle2_;
  thirdTrackMass = massParticle3_;
  firstTrackPt = minPtTk1_;
  secondTrackPt = minPtTk2_;
  thirdTrackPt = minPtTk3_;
  if (resOpt_ <= 0 && massParticle1_ != massParticle2_) {
    if (massParticle1_ == massParticle3_) {
      std::swap(secondTrackMass, thirdTrackMass);
      std::swap(secondTrackPt, thirdTrackPt);
    }
    if (massParticle2_ == massParticle3_) {
      std::swap(firstTrackMass, thirdTrackMass);
      std::swap(firstTrackPt, thirdTrackPt);
    }
  }
  firstTrackMass2 = firstTrackMass * firstTrackMass;
  secondTrackMass2 = secondTrackMass * secondTrackMass;
  thirdTrackMass2 = thirdTrackMass * thirdTrackMass;
}

HLTDisplacedtktktkVtxProducer::~HLTDisplacedtktktkVtxProducer() = default;

void HLTDisplacedtktktkVtxProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("Src", edm::InputTag("hltL3MuonCandidates"));
  desc.add<edm::InputTag>("PreviousCandTag", edm::InputTag(""));
  desc.add<double>("MaxEtaTk", 2.5);
  desc.add<double>("MinPtResTk1", 0.0);
  desc.add<double>("MinPtResTk2", 0.0);
  desc.add<double>("MinPtThirdTk", 0.0);
  desc.add<double>("MinPtRes", 0.0);
  desc.add<double>("MinPtTri", 0.0);
  desc.add<double>("MinInvMassRes", 1.0);
  desc.add<double>("MaxInvMassRes", 20.0);
  desc.add<double>("MinInvMass", 1.0);
  desc.add<double>("MaxInvMass", 20.0);
  desc.add<double>("massParticleRes1", 0.4937);
  desc.add<double>("massParticleRes2", 0.4937);
  desc.add<double>("massParticle3", 0.1396);
  desc.add<int>("ChargeOpt", -1);
  desc.add<int>("ResOpt", 1);
  desc.add<int>("triggerTypeDaughters", 0);

  descriptions.add("hltDisplacedtktktkVtxProducer", desc);
}

// ------------ method called on each new Event  ------------
void HLTDisplacedtktktkVtxProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get hold of track trks
  Handle<RecoChargedCandidateCollection> trackcands;
  iEvent.getByToken(srcToken_, trackcands);
  if (trackcands->size() < 3)
    return;

  //get the transient track builder:
  edm::ESHandle<TransientTrackBuilder> theB;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", theB);

  std::unique_ptr<VertexCollection> vertexCollection(new VertexCollection());

  // look at all trackcands,  check cuts and make vertices
  double e1, e2, e3;
  Particle::LorentzVector p, pres;

  RecoChargedCandidateCollection::const_iterator cand1;
  RecoChargedCandidateCollection::const_iterator cand2;
  RecoChargedCandidateCollection::const_iterator cand3;

  // get the objects passing the previous filter
  Handle<TriggerFilterObjectWithRefs> previousCands;
  iEvent.getByToken(previousCandToken_, previousCands);

  vector<RecoChargedCandidateRef> vPrevCands;
  previousCands->getObjects(triggerTypeDaughters_, vPrevCands);

  std::vector<bool> candComp;
  for (cand1 = trackcands->begin(); cand1 != trackcands->end(); cand1++)
    candComp.push_back(checkPreviousCand(cand1->get<TrackRef>(), vPrevCands));

  for (cand1 = trackcands->begin(); cand1 != trackcands->end(); cand1++) {
    TrackRef tk1 = cand1->get<TrackRef>();
    LogDebug("HLTDisplacedtktktkVtxProducer") << " 1st track in loop: q*pt= " << cand1->charge() * cand1->pt()
                                              << ", eta= " << cand1->eta() << ", hits= " << tk1->numberOfValidHits();

    //first check if this track passed the previous filter
    if (!candComp[cand1 - trackcands->begin()])
      continue;
    // if( ! checkPreviousCand( tk1, vPrevCands) ) continue;

    // cuts
    if (std::abs(cand1->eta()) > maxEta_)
      continue;
    if (cand1->pt() < firstTrackPt)
      continue;

    cand2 = trackcands->begin();
    if (firstTrackMass == secondTrackMass) {
      cand2 = cand1 + 1;
    }

    for (; cand2 != trackcands->end(); cand2++) {
      TrackRef tk2 = cand2->get<TrackRef>();
      if (tk1 == tk2)
        continue;
      LogDebug("HLTDisplacedtktktkVtxProducer")
          << " 2nd track in loop: q*pt= " << cand2->charge() * cand2->pt() << ", eta= " << cand2->eta()
          << ", hits= " << tk2->numberOfValidHits() << ", d0= " << tk2->d0();

      //first check if this track passed the previous filter
      if (!candComp[cand2 - trackcands->begin()])
        continue;
      // if( ! checkPreviousCand( tk2, vPrevCands) ) continue;

      // cuts
      if (std::abs(cand2->eta()) > maxEta_)
        continue;
      if (cand2->pt() < secondTrackPt)
        continue;

      // opposite sign or same sign for resonance
      if (resOpt_ > 0) {
        if (chargeOpt_ < 0) {
          if (cand1->charge() * cand2->charge() > 0)
            continue;
        } else if (chargeOpt_ > 0) {
          if (cand1->charge() * cand2->charge() < 0)
            continue;
        }
      }

      //
      // Combined ditrack system
      e1 = sqrt(cand1->momentum().Mag2() + firstTrackMass2);
      e2 = sqrt(cand2->momentum().Mag2() + secondTrackMass2);
      pres = Particle::LorentzVector(cand1->px(), cand1->py(), cand1->pz(), e1) +
             Particle::LorentzVector(cand2->px(), cand2->py(), cand2->pz(), e2);

      if (resOpt_ > 0) {
        if (pres.pt() < minPtRes_)
          continue;
        double invmassRes = std::abs(pres.mass());
        LogDebug("HLTDisplacedtktktkVtxProducer") << " ... 1-2 invmass= " << invmassRes;
        if (invmassRes < minInvMassRes_)
          continue;
        if (invmassRes > maxInvMassRes_)
          continue;
      }

      cand3 = trackcands->begin();
      if (firstTrackMass == secondTrackMass && firstTrackMass == thirdTrackMass && resOpt_ <= 0) {
        cand3 = cand2 + 1;
      }

      for (; cand3 != trackcands->end(); cand3++) {
        TrackRef tk3 = cand3->get<TrackRef>();
        if (tk1 == tk3 || tk2 == tk3)
          continue;
        LogDebug("HLTDisplacedtktktkVtxProducer")
            << " 3rd track in loop: q*pt= " << cand3->charge() * cand3->pt() << ", eta= " << cand3->eta()
            << ", hits= " << tk3->numberOfValidHits();

        //first check if this track passed the previous filter
        if (!candComp[cand3 - trackcands->begin()])
          continue;
        // if( ! checkPreviousCand( tk3, vPrevCands) ) continue;

        // cuts
        if (std::abs(cand3->eta()) > maxEta_)
          continue;
        if (cand3->pt() < thirdTrackPt)
          continue;

        e3 = sqrt(cand3->momentum().Mag2() + thirdTrackMass2);
        p = Particle::LorentzVector(cand1->px(), cand1->py(), cand1->pz(), e1) +
            Particle::LorentzVector(cand2->px(), cand2->py(), cand2->pz(), e2) +
            Particle::LorentzVector(cand3->px(), cand3->py(), cand3->pz(), e3);

        if (p.pt() < minPtTri_)
          continue;
        double invmass = std::abs(p.mass());
        LogDebug("HLTDisplacedtktktkVtxProducer") << " ... 1-2-3 invmass= " << invmass;
        if (invmass < minInvMass_)
          continue;
        if (invmass > maxInvMass_)
          continue;

        // do the vertex fit
        vector<TransientTrack> t_tks;
        TransientTrack ttkp1 = (*theB).build(&tk1);
        TransientTrack ttkp2 = (*theB).build(&tk2);
        TransientTrack ttkp3 = (*theB).build(&tk3);

        t_tks.push_back(ttkp1);
        t_tks.push_back(ttkp2);
        t_tks.push_back(ttkp3);

        if (t_tks.size() != 3)
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
  }
  iEvent.put(std::move(vertexCollection));
}

bool HLTDisplacedtktktkVtxProducer::checkPreviousCand(const TrackRef& trackref,
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
