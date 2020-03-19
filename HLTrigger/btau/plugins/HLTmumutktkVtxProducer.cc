#include <algorithm>
#include <cmath>
#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

#include "HLTmumutktkVtxProducer.h"
#include <DataFormats/Math/interface/deltaR.h>
#include "TMath.h"

using namespace edm;
using namespace reco;
using namespace std;
using namespace trigger;

// ----------------------------------------------------------------------
HLTmumutktkVtxProducer::HLTmumutktkVtxProducer(const edm::ParameterSet& iConfig)
    : muCandTag_(iConfig.getParameter<edm::InputTag>("MuCand")),
      muCandToken_(consumes<reco::RecoChargedCandidateCollection>(muCandTag_)),
      trkCandTag_(iConfig.getParameter<edm::InputTag>("TrackCand")),
      trkCandToken_(consumes<reco::RecoChargedCandidateCollection>(trkCandTag_)),
      previousCandTag_(iConfig.getParameter<edm::InputTag>("PreviousCandTag")),
      previousCandToken_(consumes<trigger::TriggerFilterObjectWithRefs>(previousCandTag_)),
      mfName_(iConfig.getParameter<std::string>("SimpleMagneticField")),
      thirdTrackMass_(iConfig.getParameter<double>("ThirdTrackMass")),
      fourthTrackMass_(iConfig.getParameter<double>("FourthTrackMass")),
      maxEta_(iConfig.getParameter<double>("MaxEta")),
      minPt_(iConfig.getParameter<double>("MinPt")),
      minInvMass_(iConfig.getParameter<double>("MinInvMass")),
      maxInvMass_(iConfig.getParameter<double>("MaxInvMass")),
      minTrkTrkMass_(iConfig.getParameter<double>("MinTrkTrkMass")),
      maxTrkTrkMass_(iConfig.getParameter<double>("MaxTrkTrkMass")),
      minD0Significance_(iConfig.getParameter<double>("MinD0Significance")),
      oppositeSign_(iConfig.getParameter<bool>("OppositeSign")),
      overlapDR_(iConfig.getParameter<double>("OverlapDR")),
      beamSpotTag_(iConfig.getParameter<edm::InputTag>("BeamSpotTag")),
      beamSpotToken_(consumes<reco::BeamSpot>(beamSpotTag_)) {
  produces<VertexCollection>();
}

// ----------------------------------------------------------------------
HLTmumutktkVtxProducer::~HLTmumutktkVtxProducer() = default;

void HLTmumutktkVtxProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("MuCand", edm::InputTag("hltMuTracks"));
  desc.add<edm::InputTag>("TrackCand", edm::InputTag("hltMumukAllConeTracks"));
  desc.add<edm::InputTag>("PreviousCandTag", edm::InputTag("hltDisplacedmumuFilterDoubleMu4Jpsi"));
  desc.add<std::string>("SimpleMagneticField", "");
  desc.add<double>("ThirdTrackMass", 0.493677);
  desc.add<double>("FourthTrackMass", 0.493677);
  desc.add<double>("MaxEta", 2.5);
  desc.add<double>("MinPt", 0.0);
  desc.add<double>("MinInvMass", 0.0);
  desc.add<double>("MaxInvMass", 99999.);
  desc.add<double>("MinTrkTrkMass", 0.0);
  desc.add<double>("MaxTrkTrkMass", 99999.);
  desc.add<double>("MinD0Significance", 0.0);
  desc.add<bool>("OppositeSign", false);
  desc.add<double>("OverlapDR", 0.001);
  desc.add<edm::InputTag>("BeamSpotTag", edm::InputTag("hltOfflineBeamSpot"));
  descriptions.add("HLTmumutktkVtxProducer", desc);
}

// ----------------------------------------------------------------------
void HLTmumutktkVtxProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const double MuMass(0.106);
  const double MuMass2(MuMass * MuMass);
  const double thirdTrackMass2(thirdTrackMass_ * thirdTrackMass_);
  const double fourthTrackMass2(fourthTrackMass_ * fourthTrackMass_);

  // get hold of muon trks
  Handle<RecoChargedCandidateCollection> mucands;
  iEvent.getByToken(muCandToken_, mucands);

  //get the transient track builder:
  edm::ESHandle<TransientTrackBuilder> theB;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", theB);

  //get the beamspot position
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(beamSpotToken_, recoBeamSpotHandle);

  //get the b field
  ESHandle<MagneticField> bFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(mfName_, bFieldHandle);
  const MagneticField* magField = bFieldHandle.product();
  TSCBLBuilderNoMaterial blsBuilder;

  // get track candidates around displaced muons
  Handle<RecoChargedCandidateCollection> trkcands;
  iEvent.getByToken(trkCandToken_, trkcands);

  unique_ptr<VertexCollection> vertexCollection(new VertexCollection());

  // Ref to Candidate object to be recorded in filter object
  RecoChargedCandidateRef refMu1;
  RecoChargedCandidateRef refMu2;
  RecoChargedCandidateRef refTrk1;
  RecoChargedCandidateRef refTrk2;

  double e1, e2, e3_m3, e3_m4, e4_m3, e4_m4;
  Particle::LorentzVector p, pBar, p1, p2, p3_m3, p3_m4, p4_m3, p4_m4, p_m3m4, p_m4m3;

  if (mucands->size() < 2)
    return;
  if (trkcands->size() < 2)
    return;

  RecoChargedCandidateCollection::const_iterator mucand1;
  RecoChargedCandidateCollection::const_iterator mucand2;
  RecoChargedCandidateCollection::const_iterator trkcand1;
  RecoChargedCandidateCollection::const_iterator trkcand2;

  // get the objects passing the previous filter
  Handle<TriggerFilterObjectWithRefs> previousCands;
  iEvent.getByToken(previousCandToken_, previousCands);

  vector<RecoChargedCandidateRef> vPrevCands;
  previousCands->getObjects(TriggerMuon, vPrevCands);

  for (mucand1 = mucands->begin(); mucand1 != mucands->end(); ++mucand1) {
    TrackRef trk1 = mucand1->get<TrackRef>();
    LogDebug("HLTmumutktkVtxProducer") << " 1st muon: q*pt= " << trk1->charge() * trk1->pt() << ", eta= " << trk1->eta()
                                       << ", hits= " << trk1->numberOfValidHits();

    //first check if this muon passed the previous filter
    if (!checkPreviousCand(trk1, vPrevCands))
      continue;
    // eta and pt cut
    if (fabs(trk1->eta()) > maxEta_)
      continue;
    if (trk1->pt() < minPt_)
      continue;

    mucand2 = mucand1;
    ++mucand2;
    for (; mucand2 != mucands->end(); mucand2++) {
      TrackRef trk2 = mucand2->get<TrackRef>();
      if (overlap(trk1, trk2))
        continue;

      LogDebug("HLTDisplacedMumukFilter") << " 2nd muon: q*pt= " << trk2->charge() * trk2->pt()
                                          << ", eta= " << trk2->eta() << ", hits= " << trk2->numberOfValidHits();

      //first check if this muon passed the previous filter
      if (!checkPreviousCand(trk2, vPrevCands))
        continue;
      // eta and pt cut
      if (fabs(trk2->eta()) > maxEta_)
        continue;
      if (trk2->pt() < minPt_)
        continue;

      //loop on track collection - trk1
      for (trkcand1 = trkcands->begin(); trkcand1 != trkcands->end(); ++trkcand1) {
        TrackRef trk3 = trkcand1->get<TrackRef>();

        if (overlap(trk1, trk3))
          continue;
        if (overlap(trk2, trk3))
          continue;

        LogDebug("HLTDisplacedMumukFilter") << " 3rd track: q*pt= " << trk3->charge() * trk3->pt()
                                            << ", eta= " << trk3->eta() << ", hits= " << trk3->numberOfValidHits();

        // eta and pt cut
        if (fabs(trk3->eta()) > maxEta_)
          continue;
        if (trk3->pt() < minPt_)
          continue;

        FreeTrajectoryState InitialFTS_Trk3 = initialFreeState(*trk3, magField);
        TrajectoryStateClosestToBeamLine tscb_Trk3(blsBuilder(InitialFTS_Trk3, *recoBeamSpotHandle));
        double d0sigTrk3 = tscb_Trk3.transverseImpactParameter().significance();
        if (d0sigTrk3 < minD0Significance_)
          continue;

        //loop on track collection - trk2
        for (trkcand2 = trkcands->begin(); trkcand2 != trkcands->end(); ++trkcand2) {
          TrackRef trk4 = trkcand2->get<TrackRef>();

          if (oppositeSign_) {
            if (trk3->charge() * trk4->charge() != -1)
              continue;
          }
          if (overlap(trk1, trk4))
            continue;
          if (overlap(trk2, trk4))
            continue;
          if (overlap(trk3, trk4))
            continue;

          LogDebug("HLTDisplacedMumukFilter") << " 4th track: q*pt= " << trk4->charge() * trk4->pt()
                                              << ", eta= " << trk4->eta() << ", hits= " << trk4->numberOfValidHits();

          // eta and pt cut
          if (fabs(trk4->eta()) > maxEta_)
            continue;
          if (trk4->pt() < minPt_)
            continue;

          FreeTrajectoryState InitialFTS_Trk4 = initialFreeState(*trk4, magField);
          TrajectoryStateClosestToBeamLine tscb_Trk4(blsBuilder(InitialFTS_Trk4, *recoBeamSpotHandle));
          double d0sigTrk4 = tscb_Trk4.transverseImpactParameter().significance();
          if (d0sigTrk4 < minD0Significance_)
            continue;

          // Combined system
          e1 = sqrt(trk1->momentum().Mag2() + MuMass2);
          e2 = sqrt(trk2->momentum().Mag2() + MuMass2);
          e3_m3 = sqrt(trk3->momentum().Mag2() + thirdTrackMass2);
          e3_m4 = sqrt(trk3->momentum().Mag2() + fourthTrackMass2);
          e4_m3 = sqrt(trk4->momentum().Mag2() + thirdTrackMass2);
          e4_m4 = sqrt(trk4->momentum().Mag2() + fourthTrackMass2);

          p1 = Particle::LorentzVector(trk1->px(), trk1->py(), trk1->pz(), e1);
          p2 = Particle::LorentzVector(trk2->px(), trk2->py(), trk2->pz(), e2);
          p3_m3 = Particle::LorentzVector(trk3->px(), trk3->py(), trk3->pz(), e3_m3);
          p3_m4 = Particle::LorentzVector(trk3->px(), trk3->py(), trk3->pz(), e3_m4);
          p4_m3 = Particle::LorentzVector(trk4->px(), trk4->py(), trk4->pz(), e4_m3);
          p4_m4 = Particle::LorentzVector(trk4->px(), trk4->py(), trk4->pz(), e4_m4);

          p = p1 + p2 + p3_m3 + p4_m4;
          pBar = p1 + p2 + p3_m4 + p4_m3;
          p_m3m4 = p3_m3 + p4_m4;
          p_m4m3 = p3_m4 + p4_m3;

          //invariant mass cut
          if (!((p_m3m4.mass() > minTrkTrkMass_ && p_m3m4.mass() < maxTrkTrkMass_) ||
                (p_m4m3.mass() > minTrkTrkMass_ && p_m4m3.mass() < maxTrkTrkMass_)))
            continue;
          if (!((p.mass() > minInvMass_ && p.mass() < maxInvMass_) ||
                (pBar.mass() > minInvMass_ && pBar.mass() < maxInvMass_)))
            continue;

          // do the vertex fit
          vector<TransientTrack> t_tks;
          t_tks.push_back((*theB).build(&trk1));
          t_tks.push_back((*theB).build(&trk2));
          t_tks.push_back((*theB).build(&trk3));
          t_tks.push_back((*theB).build(&trk4));
          if (t_tks.size() != 4)
            continue;

          KalmanVertexFitter kvf;
          TransientVertex tv = kvf.vertex(t_tks);
          if (!tv.isValid())
            continue;
          Vertex vertex = tv;

          vertexCollection->push_back(vertex);
        }
      }
    }
  }
  iEvent.put(std::move(vertexCollection));
}

FreeTrajectoryState HLTmumutktkVtxProducer::initialFreeState(const reco::Track& tk, const MagneticField* field) {
  Basic3DVector<float> pos(tk.vertex());
  GlobalPoint gpos(pos);
  Basic3DVector<float> mom(tk.momentum());
  GlobalVector gmom(mom);
  GlobalTrajectoryParameters par(gpos, gmom, tk.charge(), field);
  CurvilinearTrajectoryError err(tk.covariance());
  return FreeTrajectoryState(par, err);
}

bool HLTmumutktkVtxProducer::overlap(const TrackRef& trackref1, const TrackRef& trackref2) {
  if (deltaR(trackref1->eta(), trackref1->phi(), trackref2->eta(), trackref2->phi()) < overlapDR_)
    return true;
  return false;
}

bool HLTmumutktkVtxProducer::checkPreviousCand(const TrackRef& trackref,
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
