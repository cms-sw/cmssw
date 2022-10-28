/** \class LeptonFSRProducer
 * Search for FSR photons for muons and electrons.
 *
 * Photon candidates are searched among the "packedPFCandidates" collection with the specified cuts, and are required to be isolated 
 * (relIso, with a cone of 0.3) and not to be in the footprint of all electrons in the "electrons" collection.
 * Each photon is matched by DeltaR to the closest among all muons and electrons and stored if passing dR/ET^2<deltaROverEt2Max.
 * In addition ValueMaps are stored, with links to one photon per muon/electron. For this purpose, if more than a photon
 * is matched to a lepton, the lowest-DR/ET^2 is chosen.
 *
 */

#include <memory>
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
#include "DataFormats/Common/interface/ValueMap.h"

class LeptonFSRProducer : public edm::global::EDProducer<> {
public:
  explicit LeptonFSRProducer(const edm::ParameterSet& iConfig)
      : pfcands_{consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("packedPFCandidates"))},
        electronsForVeto_{consumes<pat::ElectronCollection>(iConfig.getParameter<edm::InputTag>("slimmedElectrons"))},
        muons_{consumes<edm::View<reco::Muon>>(iConfig.getParameter<edm::InputTag>("muons"))},
        electrons_{consumes<edm::View<reco::GsfElectron>>(iConfig.getParameter<edm::InputTag>("electrons"))},
        ptCutMu(iConfig.getParameter<double>("muonPtMin")),
        etaCutMu(iConfig.getParameter<double>("muonEtaMax")),
        ptCutE(iConfig.getParameter<double>("elePtMin")),
        etaCutE(iConfig.getParameter<double>("eleEtaMax")),
        photonPtCut(iConfig.getParameter<double>("photonPtMin")),
        drEtCut(iConfig.getParameter<double>("deltaROverEt2Max")),
        isoCut(iConfig.getParameter<double>("isolation")),
        drSafe(0.0001) {
    produces<std::vector<pat::GenericParticle>>();
    produces<edm::ValueMap<int>>("muFsrIndex");
    produces<edm::ValueMap<int>>("eleFsrIndex");
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
        ->setComment("collection of muons to match with FSR ");
    desc.add<edm::InputTag>("electrons", edm::InputTag("slimmedElectrons"))
        ->setComment("collection of electrons to match with FSR ");
    desc.add<double>("muonPtMin", 3.)->setComment("minimum pt of the muon to look for a near photon");
    desc.add<double>("muonEtaMax", 2.4)->setComment("max eta of the muon to look for a near photon");
    desc.add<double>("elePtMin", 5.)->setComment("minimum pt of the electron to look for a near photon");
    desc.add<double>("eleEtaMax", 2.5)->setComment("max eta of the electron to look for a near photon");
    desc.add<double>("photonPtMin", 2.0)->setComment("minimum photon Pt");
    desc.add<double>("deltaROverEt2Max", 0.05)->setComment("max ratio of deltaR(lep,photon) over et2 of the photon");
    desc.add<double>("isolation", 2.0)->setComment("photon relative isolation cut");

    descriptions.addWithDefaultLabel(desc);
  }
  ~LeptonFSRProducer() override = default;

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  double computeRelativeIsolation(const pat::PackedCandidate& photon,
                                  const pat::PackedCandidateCollection& pfcands,
                                  const double& isoConeMax2,
                                  const double& isoConeMin2) const;

  bool electronFootprintVeto(pat::PackedCandidateRef& pfcandRef,
                             edm::Handle<pat::ElectronCollection> electronsForVeto) const;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<pat::PackedCandidateCollection> pfcands_;
  const edm::EDGetTokenT<pat::ElectronCollection> electronsForVeto_;
  const edm::EDGetTokenT<edm::View<reco::Muon>> muons_;
  const edm::EDGetTokenT<edm::View<reco::GsfElectron>> electrons_;
  const double ptCutMu;
  const double etaCutMu;
  const double ptCutE;
  const double etaCutE;
  const double photonPtCut;
  const double drEtCut;
  const double isoCut;
  const double drSafe;
};

void LeptonFSRProducer::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace std;

  edm::Handle<pat::PackedCandidateCollection> pfcands;
  iEvent.getByToken(pfcands_, pfcands);
  edm::Handle<edm::View<reco::Muon>> muons;
  iEvent.getByToken(muons_, muons);
  edm::Handle<edm::View<reco::GsfElectron>> electrons;
  iEvent.getByToken(electrons_, electrons);
  edm::Handle<pat::ElectronCollection> electronsForVeto;
  iEvent.getByToken(electronsForVeto_, electronsForVeto);

  // The output collection of FSR photons
  auto fsrPhotons = std::make_unique<std::vector<pat::GenericParticle>>();

  std::vector<int> muPhotonIdxs(muons->size(), -1);
  std::vector<double> muPhotonDRET2(muons->size(), 1e9);

  std::vector<int> elePhotonIdxs(electrons->size(), -1);
  std::vector<double> elePhotonDRET2(electrons->size(), 1e9);

  //----------------------
  // Loop on photon candidates
  //----------------------

  for (auto pc = pfcands->begin(); pc != pfcands->end(); pc++) {
    // consider only photons, with pT and eta cuts
    if (abs(pc->pdgId()) != 22 || pc->pt() < photonPtCut || abs(pc->eta()) > 2.5)
      continue;

    //------------------------------------------------------
    // Get the closest lepton
    //------------------------------------------------------
    double dRMin(0.5);
    int closestMu = -1;
    int closestEle = -1;
    double photon_relIso03 = 1e9;  // computed only if necessary
    bool skipPhoton = false;

    for (auto muon = muons->begin(); muon != muons->end(); ++muon) {
      if (muon->pt() < ptCutMu || std::abs(muon->eta()) > etaCutMu)
        continue;

      int muonIdx = muon - muons->begin();
      double dR = deltaR(muon->eta(), muon->phi(), pc->eta(), pc->phi());
      if (dR < dRMin && dR > drSafe && dR < drEtCut * pc->pt() * pc->pt()) {
        // Check if photon is isolated
        photon_relIso03 = computeRelativeIsolation(*pc, *pfcands, 0.3 * 0.3, drSafe * drSafe);
        if (photon_relIso03 > isoCut) {
          skipPhoton = true;
          break;  // break loop on muons -> photon will be skipped
        }
        // Check that photon is not in footprint of an electron
        pat::PackedCandidateRef pfcandRef = pat::PackedCandidateRef(pfcands, pc - pfcands->begin());
        skipPhoton = electronFootprintVeto(pfcandRef, electronsForVeto);
        if (skipPhoton)
          break;  // break loop on muons -> photon will be skipped

        // Candidate matching
        dRMin = dR;
        closestMu = muonIdx;
      }
    }  // end of loop on muons

    if (skipPhoton)
      continue;  // photon does not pass iso or ele footprint veto; do not look for electrons

    for (auto ele = electrons->begin(); ele != electrons->end(); ++ele) {
      if (ele->pt() < ptCutE || std::abs(ele->eta()) > etaCutE)
        continue;

      int eleIdx = ele - electrons->begin();
      double dR = deltaR(ele->eta(), ele->phi(), pc->eta(), pc->phi());
      if (dR < dRMin && dR > drSafe && dR < drEtCut * pc->pt() * pc->pt()) {
        // Check if photon is isolated (no need to recompute iso if already done for muons above)
        if (photon_relIso03 > 1e8) {
          photon_relIso03 = computeRelativeIsolation(*pc, *pfcands, 0.3 * 0.3, drSafe * drSafe);
        }
        if (photon_relIso03 > isoCut) {
          break;  // break loop on electrons -> photon will be skipped
        }
        // Check that photon is not in footprint of an electron
        pat::PackedCandidateRef pfcandRef = pat::PackedCandidateRef(pfcands, pc - pfcands->begin());
        if (electronFootprintVeto(pfcandRef, electronsForVeto)) {
          break;  // break loop on electrons -> photon will be skipped
        }

        // Candidate matching
        dRMin = dR;
        closestEle = eleIdx;
        closestMu = -1;  // reset match to muons
      }
    }  // end loop on electrons

    if (closestMu >= 0 || closestEle >= 0) {
      // Add FSR photon to the output collection
      double dRET2 = dRMin / pc->pt() / pc->pt();
      int iPhoton = fsrPhotons->size();
      fsrPhotons->push_back(pat::GenericParticle(*pc));
      fsrPhotons->back().addUserFloat("relIso03", photon_relIso03);
      fsrPhotons->back().addUserFloat("dROverEt2", dRET2);

      if (closestMu >= 0) {
        fsrPhotons->back().addUserCand("associatedMuon", reco::CandidatePtr(muons, closestMu));
        // Store the backlink to the photon: choose the lowest-dRET2 photon for each mu...
        if (dRET2 < muPhotonDRET2[closestMu]) {
          muPhotonDRET2[closestMu] = dRET2;
          muPhotonIdxs[closestMu] = iPhoton;
        }
      } else if (closestEle >= 0) {
        // ...and same for eles
        fsrPhotons->back().addUserCand("associatedElectron", reco::CandidatePtr(electrons, closestEle));
        if (dRET2 < elePhotonDRET2[closestEle]) {
          elePhotonDRET2[closestEle] = dRET2;
          elePhotonIdxs[closestEle] = iPhoton;
        }
      }
    }
  }  // end of loop over pfCands

  edm::OrphanHandle<std::vector<pat::GenericParticle>> oh = iEvent.put(std::move(fsrPhotons));

  {
    std::unique_ptr<edm::ValueMap<int>> bareIdx(new edm::ValueMap<int>());
    edm::ValueMap<int>::Filler fillerBareIdx(*bareIdx);
    fillerBareIdx.insert(muons, muPhotonIdxs.begin(), muPhotonIdxs.end());
    fillerBareIdx.fill();
    iEvent.put(std::move(bareIdx), "muFsrIndex");
  }

  {
    std::unique_ptr<edm::ValueMap<int>> bareIdx(new edm::ValueMap<int>());
    edm::ValueMap<int>::Filler fillerBareIdx(*bareIdx);
    fillerBareIdx.insert(electrons, elePhotonIdxs.begin(), elePhotonIdxs.end());
    fillerBareIdx.fill();
    iEvent.put(std::move(bareIdx), "eleFsrIndex");
  }
}

double LeptonFSRProducer::computeRelativeIsolation(const pat::PackedCandidate& photon,
                                                   const pat::PackedCandidateCollection& pfcands,
                                                   const double& isoConeMax2,
                                                   const double& isoConeMin2) const {
  double ptsum = 0;

  for (const auto& pfcand : pfcands) {
    // Isolation cone
    double dRIsoCone2 = deltaR2(photon.eta(), photon.phi(), pfcand.eta(), pfcand.phi());
    if (dRIsoCone2 > isoConeMax2 || dRIsoCone2 < isoConeMin2)
      continue;

    // Charged hadrons
    if (pfcand.charge() != 0 && abs(pfcand.pdgId()) == 211 && pfcand.pt() > 0.2 && dRIsoCone2 > drSafe * drSafe) {
      ptsum += pfcand.pt();
      // Neutral hadrons + photons
    } else if (pfcand.charge() == 0 && (abs(pfcand.pdgId()) == 22 || abs(pfcand.pdgId()) == 130) && pfcand.pt() > 0.5 &&
               dRIsoCone2 > 0.01 * 0.01) {
      ptsum += pfcand.pt();
    }
  }

  return ptsum / photon.pt();
}

bool LeptonFSRProducer::electronFootprintVeto(pat::PackedCandidateRef& pfcandRef,
                                              edm::Handle<pat::ElectronCollection> electronsForVeto) const {
  bool skipPhoton = false;
  for (auto electrons_iter = electronsForVeto->begin(); electrons_iter != electronsForVeto->end(); ++electrons_iter) {
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
  }
  return skipPhoton;
}

//define this as a plug-in
DEFINE_FWK_MODULE(LeptonFSRProducer);
