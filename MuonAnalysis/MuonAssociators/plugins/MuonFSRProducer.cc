// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

//
// class declaration
//

class MuonFSRProducer : public edm::global::EDProducer<> {
public:
  explicit MuonFSRProducer(const edm::ParameterSet& iConfig)
      :

        pfcands_{consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("packedPFCandidates"))},
        electrons_{consumes<pat::ElectronCollection>(iConfig.getParameter<edm::InputTag>("slimmedElectrons"))},
        muons_{consumes<edm::View<reco::Muon>>(iConfig.getParameter<edm::InputTag>("muons"))},
        ptCut(iConfig.getParameter<double>("muonPtMin")),
        etaCut(iConfig.getParameter<double>("muonEtaMax")),
        photonPtCut(iConfig.getParameter<double>("photonPtMin")),
        drEtCut(iConfig.getParameter<double>("deltaROverEt2Max")),
        isoCut(iConfig.getParameter<double>("isolation")) {
    produces<std::vector<pat::GenericParticle>>();
  }
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("packedPFCandidates", edm::InputTag("packedPFCandidates"))
        ->setComment("packed pf candidates where to look for photons");
    desc.add<edm::InputTag>("slimmedElectrons", edm::InputTag("slimmedElectrons"))
        ->setComment(
            "electrons to check for footprint, the electron collection must have proper linking with the "
            "packedCandidate collection");
    desc.add<edm::InputTag>("muons", edm::InputTag("slimmedMuons"))
        ->setComment("collection of muons to correct for FSR ");
    desc.add<double>("muonPtMin", 20.)->setComment("minimum pt of the muon to look for a near photon");
    desc.add<double>("muonEtaMax", 2.4)->setComment("max eta of the muon to look for a near photon");
    desc.add<double>("photonPtMin", 2.0)->setComment("minimum photon Pt");
    desc.add<double>("deltaROverEt2Max", 0.05)->setComment("max ratio of deltsR(mu,photon) over et2 of the photon");
    desc.add<double>("isolation", 2.0)->setComment("relative isolation cut");

    descriptions.addWithDefaultLabel(desc);
  }
  ~MuonFSRProducer() override {}

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  double computeRelativeIsolation(const pat::PackedCandidate& photon,
                                  const pat::PackedCandidateCollection& pfcands,
                                  const double& isoConeMax,
                                  const double& isoConeMin) const;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<pat::PackedCandidateCollection> pfcands_;
  const edm::EDGetTokenT<pat::ElectronCollection> electrons_;
  const edm::EDGetTokenT<edm::View<reco::Muon>> muons_;
  float ptCut;
  float etaCut;
  float photonPtCut;
  float drEtCut;
  float isoCut;
};

void MuonFSRProducer::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace std;

  edm::Handle<pat::PackedCandidateCollection> pfcands;
  iEvent.getByToken(pfcands_, pfcands);
  edm::Handle<edm::View<reco::Muon>> muons;
  iEvent.getByToken(muons_, muons);
  edm::Handle<pat::ElectronCollection> electrons;
  iEvent.getByToken(electrons_, electrons);

  auto fsrPhotons = std::make_unique<std::vector<pat::GenericParticle>>();
  // loop over all muons
  for (auto muon = muons->begin(); muon != muons->end(); ++muon) {
    int photonPosition = -1;
    double distance_metric_min = -1;
    // minimum muon pT
    if (muon->pt() < ptCut)
      continue;
    // maximum muon eta
    if (abs(muon->eta()) > etaCut)
      continue;

    // for each muon, loop over all pf cadidates
    for (auto iter_pf = pfcands->begin(); iter_pf != pfcands->end(); iter_pf++) {
      auto const& pc = *iter_pf;

      // consider only photons
      if (abs(pc.pdgId()) != 22)
        continue;
      // minimum pT cut
      if (pc.pt() < photonPtCut)
        continue;

      // eta requirements
      if (abs(pc.eta()) > 1.4 and (abs(pc.eta()) < 1.6))
        continue;
      if (abs(pc.eta()) > 2.5)
        continue;

      // 0.0001 < DeltaR(photon,muon) < 0.5 requirement
      double dRPhoMu = deltaR(muon->eta(), muon->phi(), pc.eta(), pc.phi());
      if (dRPhoMu < 0.0001)
        continue;
      if (dRPhoMu > 0.5)
        continue;

      bool skipPhoton = false;
      bool closest = true;

      for (auto othermuon = muons->begin(); othermuon != muons->end(); ++othermuon) {
        if (othermuon->pt() < ptCut or abs(othermuon->eta()) > etaCut or muon == othermuon)
          continue;
        double dRPhoMuOther = deltaR(othermuon->eta(), othermuon->phi(), pc.eta(), pc.phi());
        if (dRPhoMuOther < dRPhoMu) {
          closest = false;
          break;
        }
      }
      if (!closest)
        continue;

      // Check that is not in footprint of an electron
      pat::PackedCandidateRef pfcandRef = pat::PackedCandidateRef(pfcands, iter_pf - pfcands->begin());

      for (auto electrons_iter = electrons->begin(); electrons_iter != electrons->end(); ++electrons_iter) {
        for (auto const& cand : electrons_iter->associatedPackedPFCandidates()) {
          if (!cand.isAvailable())
            continue;
          if (cand.id() != pfcandRef.id())
            throw cms::Exception("Configuration")
                << "The electron associatedPackedPFCandidates item does not have "
                << "the same ID of packed candidate collection used for cleaning the electron footprint: " << cand.id()
                << " (" << pfcandRef.id() << ")\n";
          if (cand.key() == pfcandRef.key()) {
            skipPhoton = true;
            break;
          }
        }
        if (skipPhoton)
          break;
      }

      if (skipPhoton)
        continue;

      // use only isolated photons (very loose prelection can be tightened on analysis level)
      float photon_relIso03 = computeRelativeIsolation(pc, *pfcands, 0.3, 0.0001);
      if (photon_relIso03 > isoCut)
        continue;
      double metric = deltaR(muon->eta(), muon->phi(), pc.eta(), pc.phi()) / (pc.pt() * pc.pt());
      if (metric > drEtCut)
        continue;
      fsrPhotons->push_back(pat::GenericParticle(pc));
      fsrPhotons->back().addUserFloat("relIso03", photon_relIso03);  // isolation, no CHS
      fsrPhotons->back().addUserCand("associatedMuon", reco::CandidatePtr(muons, muon - muons->begin()));
      fsrPhotons->back().addUserFloat("dROverEt2", metric);  // dR/et2 to the closest muon

      // FSR photon defined as the one with minimum value of DeltaR/Et^2
      if (photonPosition == -1 or metric < distance_metric_min) {
        distance_metric_min = metric;
        photonPosition = fsrPhotons->size() - 1;
      }
    }
  }

  edm::OrphanHandle<std::vector<pat::GenericParticle>> oh = iEvent.put(std::move(fsrPhotons));
}

double MuonFSRProducer::computeRelativeIsolation(const pat::PackedCandidate& photon,
                                                 const pat::PackedCandidateCollection& pfcands,
                                                 const double& isoConeMax,
                                                 const double& isoConeMin) const {
  double ptsum = 0;

  for (auto pfcand : pfcands) {
    // Isolation cone requirement
    double dRIsoCone = deltaR(photon.eta(), photon.phi(), pfcand.eta(), pfcand.phi());
    if (dRIsoCone > isoConeMax)
      continue;
    if (dRIsoCone < isoConeMin)
      continue;

    if (pfcand.charge() != 0 && abs(pfcand.pdgId()) == 211 && pfcand.pt() > 0.2) {
      if (dRIsoCone > 0.0001)
        ptsum += pfcand.pt();
    } else if (pfcand.charge() == 0 && (abs(pfcand.pdgId()) == 22 || abs(pfcand.pdgId()) == 130) && pfcand.pt() > 0.5) {
      if (dRIsoCone > 0.01)
        ptsum += pfcand.pt();
    }
  }

  return ptsum / photon.pt();
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonFSRProducer);
