// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

class ElectronExtendedTableProducer : public edm::global::EDProducer<> {
public:
  explicit ElectronExtendedTableProducer(const edm::ParameterSet& iConfig)
      : name_(iConfig.getParameter<std::string>("name")),
        rhoTag_(consumes<double>(iConfig.getParameter<edm::InputTag>("rho"))),
        electronTag_(consumes<std::vector<pat::Electron>>(iConfig.getParameter<edm::InputTag>("electrons"))),
        vtxTag_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertex"))),
        jetTag_(consumes<std::vector<pat::Jet>>(iConfig.getParameter<edm::InputTag>("jets"))),
        jetFatTag_(consumes<std::vector<pat::Jet>>(iConfig.getParameter<edm::InputTag>("jetsFat"))),
        jetSubTag_(consumes<std::vector<pat::Jet>>(iConfig.getParameter<edm::InputTag>("jetsSub"))) {
    produces<nanoaod::FlatTable>();
  }

  ~ElectronExtendedTableProducer() override {};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("rho")->setComment("input rho parameter");
    desc.add<edm::InputTag>("electrons")->setComment("input electron collection");
    desc.add<edm::InputTag>("primaryVertex")->setComment("input primary vertex collection");
    desc.add<edm::InputTag>("jets")->setComment("input jet collection");
    desc.add<edm::InputTag>("jetsFat")->setComment("input fat jet collection");
    desc.add<edm::InputTag>("jetsSub")->setComment("input sub jet collection");
    desc.add<std::string>("name")->setComment("name of the electron nanoaod::FlatTable we are extending");
    descriptions.add("electronTable", desc);
  }

private:
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  float GetDEtaInSeed(const pat::Electron* el) const;
  float getPFIso(const pat::Electron& electron) const;
  int findMatchedJet(const reco::Candidate& lepton, const std::vector<pat::Jet>& jets) const;
  void fillLeptonJetVariables(const reco::GsfElectron* el,
                              const std::vector<pat::Jet>& jets,
                              const reco::Vertex& vertex,
                              const double rho,
                              std::vector<int>* jetIdx,
                              std::vector<float> relIso0p4,
                              std::vector<float>* jetPtRatio,
                              std::vector<float>* jetPtRel,
                              std::vector<int>* jrtSelectedChargedMultiplicity) const;

  std::string name_;
  edm::EDGetTokenT<double> rhoTag_;
  edm::EDGetTokenT<std::vector<pat::Electron>> electronTag_;
  edm::EDGetTokenT<reco::VertexCollection> vtxTag_;
  edm::EDGetTokenT<std::vector<pat::Jet>> jetTag_;
  edm::EDGetTokenT<std::vector<pat::Jet>> jetFatTag_;
  edm::EDGetTokenT<std::vector<pat::Jet>> jetSubTag_;
};

void ElectronExtendedTableProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const double rho = iEvent.get(rhoTag_);
  const std::vector<pat::Electron>& electrons = iEvent.get(electronTag_);
  const reco::VertexCollection& primaryVertices = iEvent.get(vtxTag_);
  const std::vector<pat::Jet>& jets = iEvent.get(jetTag_);
  const std::vector<pat::Jet>& jetFat = iEvent.get(jetFatTag_);
  const std::vector<pat::Jet>& jetSub = iEvent.get(jetSubTag_);

  const auto& pv = primaryVertices.at(0);
  GlobalPoint primaryVertex(pv.x(), pv.y(), pv.z());

  unsigned int nElectrons = electrons.size();

  std::vector<float> idx, charge;

  std::vector<float> relIso0p4;
  std::vector<float> jetPtRatio, jetPtRel;
  std::vector<int> jetIdx;
  std::vector<int> jetFatIdx, jetSubIdx;
  std::vector<int> jetSelectedChargedMultiplicity;
  std::vector<float> dxy, dz, IP3d, IP3dSig;

  std::vector<bool> isEB, isEE;
  std::vector<float> superClusterOverP, ecalEnergy, dEtaInSeed;
  std::vector<int> numberInnerHitsMissing, numberOfValidPixelHits, numberOfValidTrackerHits;
  std::vector<float> sigmaIetaIeta, deltaPhiSuperClusterTrack, deltaEtaSuperClusterTrack, eInvMinusPInv, hOverE;

  for (unsigned int i = 0; i < nElectrons; i++) {
    const pat::Electron& electron = electrons[i];

    idx.push_back(i);

    charge.push_back(electron.charge());

    isEB.push_back(electron.isEB());
    isEE.push_back(electron.isEE());
    superClusterOverP.push_back(electron.eSuperClusterOverP());
    ecalEnergy.push_back(electron.ecalEnergy());
    dEtaInSeed.push_back(std::abs(GetDEtaInSeed(&electron)));
    numberInnerHitsMissing.push_back(
        electron.gsfTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS));
    numberOfValidPixelHits.push_back(
        (!electron.gsfTrack().isNull()) ? electron.gsfTrack()->hitPattern().numberOfValidPixelHits() : 0);
    numberOfValidTrackerHits.push_back(
        (!electron.gsfTrack().isNull()) ? electron.gsfTrack()->hitPattern().numberOfValidTrackerHits() : 0);

    sigmaIetaIeta.push_back(electron.full5x5_sigmaIetaIeta());
    deltaPhiSuperClusterTrack.push_back(fabs(electron.deltaPhiSuperClusterTrackAtVtx()));
    deltaEtaSuperClusterTrack.push_back(fabs(electron.deltaEtaSuperClusterTrackAtVtx()));
    eInvMinusPInv.push_back((1.0 - electron.eSuperClusterOverP()) / electron.correctedEcalEnergy());
    hOverE.push_back(electron.hadronicOverEm());

    dxy.push_back(electron.dB(pat::Electron::PV2D));
    dz.push_back(electron.dB(pat::Electron::PVDZ));
    IP3d.push_back(electron.dB(pat::Electron::PV3D));
    IP3dSig.push_back(fabs(electron.dB(pat::Electron::PV3D) / electron.edB(pat::Electron::PV3D)));

    relIso0p4.push_back(getPFIso(electron));

    fillLeptonJetVariables(
        &electron, jets, pv, rho, &jetIdx, relIso0p4, &jetPtRatio, &jetPtRel, &jetSelectedChargedMultiplicity);

    const reco::Candidate* el_cand = dynamic_cast<const reco::Candidate*>(&electron);
    jetFatIdx.push_back(findMatchedJet(*el_cand, jetFat));
    jetSubIdx.push_back(findMatchedJet(*el_cand, jetSub));
  }

  auto tab = std::make_unique<nanoaod::FlatTable>(nElectrons, name_, false, true);
  tab->addColumn<float>("idx", idx, "LLPnanoAOD electron index");

  tab->addColumn<float>("dxy", dxy, "");
  tab->addColumn<float>("dz", dz, "");
  tab->addColumn<float>("IP3d", IP3d, "");
  tab->addColumn<float>("IP3dSig", IP3dSig, "");

  tab->addColumn<float>("relIso0p4", relIso0p4, "");
  tab->addColumn<float>("jetPtRatio", jetPtRatio, "");
  tab->addColumn<float>("jetPtRel", jetPtRel, "");
  tab->addColumn<int>("jetSelectedChargedMultiplicity", jetSelectedChargedMultiplicity, "");
  tab->addColumn<int>("jetIdx", jetIdx, "");
  tab->addColumn<int>("jetFatIdx", jetFatIdx, "");
  tab->addColumn<int>("jetSubIdx", jetSubIdx, "");

  tab->addColumn<bool>("isEB", isEB, "");
  tab->addColumn<bool>("isEE", isEE, "");
  tab->addColumn<float>("superClusterOverP", superClusterOverP, "");
  tab->addColumn<float>("ecalEnergy", ecalEnergy, "");
  tab->addColumn<float>("dEtaInSeed", dEtaInSeed, "");
  tab->addColumn<int>("numberInnerHitsMissing", numberInnerHitsMissing, "");
  tab->addColumn<int>("numberOfValidPixelHits", numberOfValidPixelHits, "");
  tab->addColumn<int>("numberOfValidTrackerHits", numberOfValidTrackerHits, "");
  tab->addColumn<float>("sigmaIetaIeta", sigmaIetaIeta, "");
  tab->addColumn<float>("deltaPhiSuperClusterTrack", deltaPhiSuperClusterTrack, "");
  tab->addColumn<float>("deltaEtaSuperClusterTrack", deltaEtaSuperClusterTrack, "");
  tab->addColumn<float>("eInvMinusPInv", eInvMinusPInv, "");
  tab->addColumn<float>("hOverE", hOverE, "");

  iEvent.put(std::move(tab));
}

float ElectronExtendedTableProducer::GetDEtaInSeed(const pat::Electron* el) const {
  if (el->superCluster().isNonnull() and el->superCluster()->seed().isNonnull())
    return el->deltaEtaSuperClusterTrackAtVtx() - el->superCluster()->eta() + el->superCluster()->seed()->eta();
  else
    return std::numeric_limits<float>::max();
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

int ElectronExtendedTableProducer::findMatchedJet(const reco::Candidate& lepton,
                                                  const std::vector<pat::Jet>& jets) const {
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

void ElectronExtendedTableProducer::fillLeptonJetVariables(const reco::GsfElectron* el,
                                                           const std::vector<pat::Jet>& jets,
                                                           const reco::Vertex& vertex,
                                                           const double rho,
                                                           std::vector<int>* jetIdx,
                                                           std::vector<float> relIso0p4,
                                                           std::vector<float>* jetPtRatio,
                                                           std::vector<float>* jetPtRel,
                                                           std::vector<int>* jetSelectedChargedMultiplicity) const {
  const reco::Candidate* cand = dynamic_cast<const reco::Candidate*>(el);
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

float ElectronExtendedTableProducer::getPFIso(const pat::Electron& electron) const {
  return electron.userFloat("PFIsoAll04") / electron.pt();
}

#include "FWCore/Framework/interface/MakerMacros.h"
//define this as a plug-in
DEFINE_FWK_MODULE(ElectronExtendedTableProducer);
