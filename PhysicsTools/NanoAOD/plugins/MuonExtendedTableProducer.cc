// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

class MuonExtendedTableProducer : public edm::global::EDProducer<> {
public:
  explicit MuonExtendedTableProducer(const edm::ParameterSet& iConfig)
      : name_(iConfig.getParameter<std::string>("name")),
        rhoTag_(consumes<double>(iConfig.getParameter<edm::InputTag>("rho"))),
        muonTag_(consumes<std::vector<pat::Muon>>(iConfig.getParameter<edm::InputTag>("muons"))),
        vtxTag_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertex"))),
        bsTag_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamspot"))),
        jetTag_(consumes<std::vector<pat::Jet>>(iConfig.getParameter<edm::InputTag>("jets"))),
        jetFatTag_(consumes<std::vector<pat::Jet>>(iConfig.getParameter<edm::InputTag>("jetsFat"))),
        jetSubTag_(consumes<std::vector<pat::Jet>>(iConfig.getParameter<edm::InputTag>("jetsSub"))),
        transientTrackBuilderToken_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))) {
    produces<nanoaod::FlatTable>();
  }

  ~MuonExtendedTableProducer() override {};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("rho")->setComment("input rho parameter");
    desc.add<edm::InputTag>("muons")->setComment("input muon collection");
    desc.add<edm::InputTag>("primaryVertex")->setComment("input primary vertex collection");
    desc.add<edm::InputTag>("beamspot")->setComment("input beamspot collection");
    desc.add<edm::InputTag>("jets")->setComment("input jet collection");
    desc.add<edm::InputTag>("jetsFat")->setComment("input fat jet collection");
    desc.add<edm::InputTag>("jetsSub")->setComment("input sub jet collection");
    desc.add<std::string>("name")->setComment("name of the muon nanoaod::FlatTable we are extending");
    descriptions.add("muonTable", desc);
  }

private:
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  float getPFIso(const pat::Muon& muon) const;
  int findMatchedJet(const reco::Candidate& lepton, const std::vector<pat::Jet>& jets) const;
  void fillLeptonJetVariables(const reco::Muon* mu,
                              const std::vector<pat::Jet>& jets,
                              const reco::Vertex& vertex,
                              const double rho,
                              std::vector<int>* jetIdx,
                              std::vector<float> relIso0p4,
                              std::vector<float>* jetPtRatio,
                              std::vector<float>* jetPtRel,
                              std::vector<int>* jetSelectedChargedMultiplicity) const;

  std::string name_;
  edm::EDGetTokenT<double> rhoTag_;
  edm::EDGetTokenT<std::vector<pat::Muon>> muonTag_;
  edm::EDGetTokenT<reco::VertexCollection> vtxTag_;
  edm::EDGetTokenT<reco::BeamSpot> bsTag_;
  edm::EDGetTokenT<std::vector<reco::Track>> generalTrackTag_;
  edm::EDGetTokenT<std::vector<pat::Jet>> jetTag_;
  edm::EDGetTokenT<std::vector<pat::Jet>> jetFatTag_;
  edm::EDGetTokenT<std::vector<pat::Jet>> jetSubTag_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transientTrackBuilderToken_;
};

void MuonExtendedTableProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const double& rho = iEvent.get(rhoTag_);
  edm::Handle<std::vector<pat::Muon>> muonHandle;
  iEvent.getByToken(muonTag_, muonHandle);
  const auto& muons = *muonHandle;
  const reco::VertexCollection& primaryVertices = iEvent.get(vtxTag_);
  const reco::BeamSpot& beamspots = iEvent.get(bsTag_);
  const std::vector<pat::Jet>& jets = iEvent.get(jetTag_);
  const std::vector<pat::Jet>& jetFat = iEvent.get(jetFatTag_);
  const std::vector<pat::Jet>& jetSub = iEvent.get(jetSubTag_);

  const auto& pv = primaryVertices.at(0);
  GlobalPoint primaryVertex(pv.x(), pv.y(), pv.z());

  const auto& bs = beamspots.position();
  GlobalPoint beamSpot(bs.x(), bs.y(), bs.z());
  reco::Vertex beamSpotVertex(beamspots.position(), beamspots.covariance3D());

  const TransientTrackBuilder& builder = iSetup.getData(transientTrackBuilderToken_);

  unsigned int nMuons = muons.size();

  std::vector<float> idx, charge, trkPt, trkPtErr;
  std::vector<int> numberInnerHitsMissing;
  std::vector<float> relIso0p4;
  std::vector<float> jetPtRatio, jetPtRel;
  std::vector<int> jetIdx;
  std::vector<int> jetFatIdx, jetSubIdx;
  std::vector<int> jetSelectedChargedMultiplicity;

  std::vector<float> dzPV, dzPVErr, dxyPVTraj, dxyPVTrajErr, dxyPVSigned, dxyPVSignedErr, ip3DPVSigned, ip3DPVSignedErr;

  std::vector<float> trkNumPlanes, trkNumHits, trkNumDTHits, trkNumCSCHits, trkNumPixelHits(nMuons, -1),
      trkNumTrkLayers(nMuons, -1), normChi2;
  std::vector<float> outerEta(nMuons, -5), outerPhi(nMuons, -5);

  for (unsigned int i = 0; i < nMuons; i++) {
    const pat::Muon& muon = muons[i];
    const pat::MuonRef muonRef(muonHandle, i);

    reco::TrackRef trackRef;

    if (muon.isGlobalMuon()) {
      trackRef = muon.combinedMuon();
    } else if (muon.isStandAloneMuon()) {
      trackRef = muon.standAloneMuon();
    } else {
      trackRef = muon.tunePMuonBestTrack();
    }

    idx.push_back(i);

    const auto& track = trackRef.get();
    reco::TransientTrack transientTrack = builder.build(track);

    charge.push_back(muon.charge());

    trkPt.push_back(track->pt());
    trkPtErr.push_back(track->ptError());

    relIso0p4.push_back(getPFIso(muon));

    fillLeptonJetVariables(
        &muon, jets, pv, rho, &jetIdx, relIso0p4, &jetPtRatio, &jetPtRel, &jetSelectedChargedMultiplicity);

    const reco::Candidate* mu_cand = dynamic_cast<const reco::Candidate*>(&muon);
    jetFatIdx.push_back(findMatchedJet(*mu_cand, jetFat));
    jetSubIdx.push_back(findMatchedJet(*mu_cand, jetSub));

    numberInnerHitsMissing.push_back(!muon.innerTrack().isNull() ? muon.innerTrack()->hitPattern().numberOfLostHits(
                                                                       reco::HitPattern::MISSING_INNER_HITS)
                                                                 : 0);

    dzPV.push_back(track->dz(pv.position()));
    dzPVErr.push_back(std::hypot(track->dzError(), pv.zError()));
    TrajectoryStateClosestToPoint trajectoryPV = transientTrack.trajectoryStateClosestToPoint(primaryVertex);
    dxyPVTraj.push_back(trajectoryPV.perigeeParameters().transverseImpactParameter());
    dxyPVTrajErr.push_back(trajectoryPV.perigeeError().transverseImpactParameterError());
    GlobalVector muonRefTrackDir(muon.px(), muon.py(), muon.pz());
    dxyPVSigned.push_back(IPTools::signedTransverseImpactParameter(transientTrack, muonRefTrackDir, pv).second.value());
    dxyPVSignedErr.push_back(
        IPTools::signedTransverseImpactParameter(transientTrack, muonRefTrackDir, pv).second.error());

    ip3DPVSigned.push_back(
        IPTools::signedImpactParameter3D(transientTrack, muonRefTrackDir, beamSpotVertex).second.value());
    ip3DPVSignedErr.push_back(
        IPTools::signedImpactParameter3D(transientTrack, muonRefTrackDir, beamSpotVertex).second.error());

    trkNumPlanes.push_back(track->hitPattern().muonStationsWithValidHits());
    trkNumHits.push_back(track->hitPattern().numberOfValidMuonHits());
    trkNumDTHits.push_back(track->hitPattern().numberOfValidMuonDTHits());
    trkNumCSCHits.push_back(track->hitPattern().numberOfValidMuonCSCHits());

    normChi2.push_back(track->normalizedChi2());

    // Cannot get outer track for tracker muons
    if (track->extra().isNonnull() && track->extra().isAvailable() && track->outerOk()) {
      outerEta[i] = track->outerEta();
      outerPhi[i] = track->outerPhi();
    }
  }

  auto tab = std::make_unique<nanoaod::FlatTable>(nMuons, name_, false, true);
  tab->addColumn<float>("idx", idx, "EXOnanoAOD muon index");

  tab->addColumn<float>("trkPt", trkPt, "");
  tab->addColumn<float>("trkPtErr", trkPtErr, "");

  tab->addColumn<float>("relIso0p4", relIso0p4, "");
  tab->addColumn<float>("jetPtRatio", jetPtRatio, "");
  tab->addColumn<float>("jetPtRel", jetPtRel, "");
  tab->addColumn<int>("jetSelectedChargedMultiplicity", jetSelectedChargedMultiplicity, "");
  tab->addColumn<int>("jetIdx", jetIdx, "");
  tab->addColumn<int>("jetFatIdx", jetFatIdx, "");
  tab->addColumn<int>("jetSubIdx", jetSubIdx, "");

  tab->addColumn<int>("numberInnerHitsMissing", numberInnerHitsMissing, "");

  tab->addColumn<float>("dzPV", dzPV, "");
  tab->addColumn<float>("dzPVErr", dzPVErr, "");
  tab->addColumn<float>("dxyPVTraj", dxyPVTraj, "");
  tab->addColumn<float>("dxyPVTrajErr", dxyPVTrajErr, "");
  tab->addColumn<float>("dxyPVSigned", dxyPVSigned, "");
  tab->addColumn<float>("dxyPVSignedErr", dxyPVSignedErr, "");
  tab->addColumn<float>("ip3DPVSigned", ip3DPVSigned, "");
  tab->addColumn<float>("ip3DPVSignedErr", ip3DPVSignedErr, "");

  tab->addColumn<float>("trkNumPlanes", trkNumPlanes, "");
  tab->addColumn<float>("trkNumHits", trkNumHits, "");
  tab->addColumn<float>("trkNumDTHits", trkNumDTHits, "");
  tab->addColumn<float>("trkNumCSCHits", trkNumCSCHits, "");
  tab->addColumn<float>("normChi2", normChi2, "");
  tab->addColumn<float>("trkNumPixelHits", trkNumPixelHits, "");
  tab->addColumn<float>("trkNumTrkLayers", trkNumTrkLayers, "");

  tab->addColumn<float>("outerEta", outerEta, "");
  tab->addColumn<float>("outerPhi", outerPhi, "");

  iEvent.put(std::move(tab));
}

float MuonExtendedTableProducer::getPFIso(const pat::Muon& muon) const {
  return (muon.pfIsolationR04().sumChargedHadronPt +
          std::max(0.,
                   muon.pfIsolationR04().sumNeutralHadronEt + muon.pfIsolationR04().sumPhotonEt -
                       0.5 * muon.pfIsolationR04().sumPUPt)) /
         muon.pt();
}

template <typename T1, typename T2>
bool isSourceCandidatePtrMatch(const T1& lhs, const T2& rhs) {
  for (size_t lhsIndex = 0; lhsIndex < lhs.numberOfSourceCandidatePtrs(); ++lhsIndex) {
    auto lhsSourcePtr = lhs.sourceCandidatePtr(lhsIndex);
    for (size_t rhsIndex = 0; rhsIndex < rhs.numberOfSourceCandidatePtrs(); ++rhsIndex) {
      auto rhsSourcePtr = rhs.sourceCandidatePtr(rhsIndex);
      if (lhsSourcePtr == rhsSourcePtr) {
        return true;
      }
    }
  }

  return false;
}

int MuonExtendedTableProducer::findMatchedJet(const reco::Candidate& lepton, const std::vector<pat::Jet>& jets) const {
  int iJet = -1;

  unsigned int nJets = jets.size();

  for (unsigned int i = 0; i < nJets; i++) {
    const pat::Jet& jet = jets[i];
    if (isSourceCandidatePtrMatch(lepton, jet)) {
      return i;
    }
  }

  return iJet;
}

void MuonExtendedTableProducer::fillLeptonJetVariables(const reco::Muon* mu,
                                                       const std::vector<pat::Jet>& jets,
                                                       const reco::Vertex& vertex,
                                                       const double rho,
                                                       std::vector<int>* jetIdx,
                                                       std::vector<float> relIso0p4,
                                                       std::vector<float>* jetPtRatio,
                                                       std::vector<float>* jetPtRel,
                                                       std::vector<int>* jetSelectedChargedMultiplicity) const {
  const reco::Candidate* cand = dynamic_cast<const reco::Candidate*>(mu);
  int matchedJetIdx = findMatchedJet(*cand, jets);

  jetIdx->push_back(matchedJetIdx);

  if (matchedJetIdx < 0) {
    float ptRatio = (1. / (1. + relIso0p4.back()));
    jetPtRatio->push_back(ptRatio);
    jetPtRel->push_back(0);
    jetSelectedChargedMultiplicity->push_back(0);
  } else {
    const pat::Jet& jet = jets[matchedJetIdx];
    auto rawJetP4 = jet.correctedP4("Uncorrected");
    auto leptonP4 = cand->p4();

    bool leptonEqualsJet = ((rawJetP4 - leptonP4).P() < 1e-4);

    if (leptonEqualsJet) {
      jetPtRatio->push_back(1);
      jetPtRel->push_back(0);
      jetSelectedChargedMultiplicity->push_back(0);
    } else {
      auto L1JetP4 = jet.correctedP4("L1FastJet");
      double L2L3JEC = jet.pt() / L1JetP4.pt();
      auto lepAwareJetP4 = (L1JetP4 - leptonP4) * L2L3JEC + leptonP4;

      float ptRatio = cand->pt() / lepAwareJetP4.pt();
      float ptRel = leptonP4.Vect().Cross((lepAwareJetP4 - leptonP4).Vect().Unit()).R();
      jetPtRatio->push_back(ptRatio);
      jetPtRel->push_back(ptRel);
      jetSelectedChargedMultiplicity->push_back(0);

      for (const auto& daughterPtr : jet.daughterPtrVector()) {
        const pat::PackedCandidate& daughter = *((const pat::PackedCandidate*)daughterPtr.get());

        if (daughter.charge() == 0)
          continue;
        if (daughter.fromPV() < 2)
          continue;
        if (reco::deltaR(daughter, *cand) > 0.4)
          continue;
        if (!daughter.hasTrackDetails())
          continue;

        auto daughterTrack = daughter.pseudoTrack();

        if (daughterTrack.pt() <= 1)
          continue;
        if (daughterTrack.hitPattern().numberOfValidHits() < 8)
          continue;
        if (daughterTrack.hitPattern().numberOfValidPixelHits() < 2)
          continue;
        if (daughterTrack.normalizedChi2() >= 5)
          continue;
        if (std::abs(daughterTrack.dz(vertex.position())) >= 17)
          continue;
        if (std::abs(daughterTrack.dxy(vertex.position())) >= 0.2)
          continue;
        ++jetSelectedChargedMultiplicity->back();
      }
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
//define this as a plug-in
DEFINE_FWK_MODULE(MuonExtendedTableProducer);
