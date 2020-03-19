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

#include "HLTmumutkVtxProducer.h"
#include <DataFormats/Math/interface/deltaR.h>

using namespace edm;
using namespace reco;
using namespace std;
using namespace trigger;

// ----------------------------------------------------------------------
HLTmumutkVtxProducer::HLTmumutkVtxProducer(const edm::ParameterSet& iConfig)
    : muCandTag_(iConfig.getParameter<edm::InputTag>("MuCand")),
      muCandToken_(consumes<reco::RecoChargedCandidateCollection>(muCandTag_)),
      trkCandTag_(iConfig.getParameter<edm::InputTag>("TrackCand")),
      trkCandToken_(consumes<reco::RecoChargedCandidateCollection>(trkCandTag_)),
      previousCandTag_(iConfig.getParameter<edm::InputTag>("PreviousCandTag")),
      previousCandToken_(consumes<trigger::TriggerFilterObjectWithRefs>(previousCandTag_)),
      mfName_(iConfig.getParameter<std::string>("SimpleMagneticField")),
      thirdTrackMass_(iConfig.getParameter<double>("ThirdTrackMass")),
      maxEta_(iConfig.getParameter<double>("MaxEta")),
      minPt_(iConfig.getParameter<double>("MinPt")),
      minInvMass_(iConfig.getParameter<double>("MinInvMass")),
      maxInvMass_(iConfig.getParameter<double>("MaxInvMass")),
      minD0Significance_(iConfig.getParameter<double>("MinD0Significance")),
      overlapDR_(iConfig.getParameter<double>("OverlapDR")),
      beamSpotTag_(iConfig.getParameter<edm::InputTag>("BeamSpotTag")),
      beamSpotToken_(consumes<reco::BeamSpot>(beamSpotTag_)) {
  produces<VertexCollection>();
}

// ----------------------------------------------------------------------
HLTmumutkVtxProducer::~HLTmumutkVtxProducer() = default;

void HLTmumutkVtxProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("MuCand", edm::InputTag("hltMuTracks"));
  desc.add<edm::InputTag>("TrackCand", edm::InputTag("hltMumukAllConeTracks"));
  desc.add<edm::InputTag>("PreviousCandTag", edm::InputTag("hltDisplacedmumuFilterDoubleMu4Jpsi"));
  desc.add<std::string>("SimpleMagneticField", "");
  desc.add<double>("ThirdTrackMass", 0.493677);
  desc.add<double>("MaxEta", 2.5);
  desc.add<double>("MinPt", 3.0);
  desc.add<double>("MinInvMass", 0.0);
  desc.add<double>("MaxInvMass", 99999.);
  desc.add<double>("MinD0Significance", 0.0);
  desc.add<double>("OverlapDR", 1.44e-4);
  desc.add<edm::InputTag>("BeamSpotTag", edm::InputTag("hltOfflineBeamSpot"));
  descriptions.add("HLTmumutkVtxProducer", desc);
}

// ----------------------------------------------------------------------
void HLTmumutkVtxProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const double MuMass(0.106);
  const double MuMass2(MuMass * MuMass);
  const double thirdTrackMass2(thirdTrackMass_ * thirdTrackMass_);

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
  RecoChargedCandidateRef refTrk;

  double e1, e2, e3;
  Particle::LorentzVector p, p1, p2, p3;

  if (mucands->size() < 2)
    return;
  if (trkcands->empty())
    return;

  RecoChargedCandidateCollection::const_iterator mucand1;
  RecoChargedCandidateCollection::const_iterator mucand2;
  RecoChargedCandidateCollection::const_iterator trkcand;

  // get the objects passing the previous filter
  Handle<TriggerFilterObjectWithRefs> previousCands;
  iEvent.getByToken(previousCandToken_, previousCands);

  vector<RecoChargedCandidateRef> vPrevCands;
  previousCands->getObjects(TriggerMuon, vPrevCands);

  for (mucand1 = mucands->begin(); mucand1 != mucands->end(); ++mucand1) {
    TrackRef trk1 = mucand1->get<TrackRef>();
    LogDebug("HLTmumutkVtxProducer") << " 1st muon: q*pt= " << trk1->charge() * trk1->pt() << ", eta= " << trk1->eta()
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

      //loop on track collection
      for (trkcand = trkcands->begin(); trkcand != trkcands->end(); ++trkcand) {
        TrackRef trk3 = trkcand->get<TrackRef>();
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

        // Combined system
        e1 = sqrt(trk1->momentum().Mag2() + MuMass2);
        e2 = sqrt(trk2->momentum().Mag2() + MuMass2);
        e3 = sqrt(trk3->momentum().Mag2() + thirdTrackMass2);

        p1 = Particle::LorentzVector(trk1->px(), trk1->py(), trk1->pz(), e1);
        p2 = Particle::LorentzVector(trk2->px(), trk2->py(), trk2->pz(), e2);
        p3 = Particle::LorentzVector(trk3->px(), trk3->py(), trk3->pz(), e3);

        p = p1 + p2 + p3;

        //invariant mass cut
        double invmass = abs(p.mass());
        LogDebug("HLTDisplacedMumukFilter") << " Invmass= " << invmass;
        if (invmass < minInvMass_)
          continue;
        if (invmass > maxInvMass_)
          continue;

        // do the vertex fit
        vector<TransientTrack> t_tks;
        t_tks.push_back((*theB).build(&trk1));
        t_tks.push_back((*theB).build(&trk2));
        t_tks.push_back((*theB).build(&trk3));
        if (t_tks.size() != 3)
          continue;

        FreeTrajectoryState InitialFTS = initialFreeState(*trk3, magField);
        TrajectoryStateClosestToBeamLine tscb(blsBuilder(InitialFTS, *recoBeamSpotHandle));
        double d0sig = tscb.transverseImpactParameter().significance();
        if (d0sig < minD0Significance_)
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

FreeTrajectoryState HLTmumutkVtxProducer::initialFreeState(const reco::Track& tk, const MagneticField* field) {
  Basic3DVector<float> pos(tk.vertex());
  GlobalPoint gpos(pos);
  Basic3DVector<float> mom(tk.momentum());
  GlobalVector gmom(mom);
  GlobalTrajectoryParameters par(gpos, gmom, tk.charge(), field);
  CurvilinearTrajectoryError err(tk.covariance());
  return FreeTrajectoryState(par, err);
}

bool HLTmumutkVtxProducer::overlap(const TrackRef& trackref1, const TrackRef& trackref2) {
  if (deltaR(trackref1->eta(), trackref1->phi(), trackref2->eta(), trackref2->phi()) < overlapDR_)
    return true;
  return false;
}

bool HLTmumutkVtxProducer::checkPreviousCand(const TrackRef& trackref,
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
