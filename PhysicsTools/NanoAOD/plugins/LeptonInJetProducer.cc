// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "fastjet/PseudoJet.hh"
#include <fastjet/JetDefinition.hh>
#include <TLorentzVector.h>
#include <TMath.h>

#include "PhysicsTools/NanoAOD/interface/MatchingUtils.h"

template <typename T>
class LeptonInJetProducer : public edm::global::EDProducer<> {
public:
  explicit LeptonInJetProducer(const edm::ParameterSet &iConfig)
      : srcJet_(consumes<edm::View<pat::Jet>>(iConfig.getParameter<edm::InputTag>("src"))),
        srcEle_(consumes<edm::View<pat::Electron>>(iConfig.getParameter<edm::InputTag>("srcEle"))),
        srcMu_(consumes<edm::View<pat::Muon>>(iConfig.getParameter<edm::InputTag>("srcMu"))) {
    produces<edm::ValueMap<float>>("lsf3");
    produces<edm::ValueMap<int>>("muIdx3SJ");
    produces<edm::ValueMap<int>>("eleIdx3SJ");
  }
  ~LeptonInJetProducer() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, edm::EventSetup const &) const override;

  static bool orderPseudoJet(fastjet::PseudoJet j1, fastjet::PseudoJet j2);
  std::tuple<float, float> calculateLSF(std::vector<fastjet::PseudoJet> iCParticles,
                                        std::vector<fastjet::PseudoJet> &ljets,
                                        float ilPt,
                                        float ilEta,
                                        float ilPhi,
                                        int ilId,
                                        double dr,
                                        int nsj) const;

  edm::EDGetTokenT<edm::View<pat::Jet>> srcJet_;
  edm::EDGetTokenT<edm::View<pat::Electron>> srcEle_;
  edm::EDGetTokenT<edm::View<pat::Muon>> srcMu_;
};

// ------------ method called to produce the data  ------------
template <typename T>
void LeptonInJetProducer<T>::produce(edm::StreamID streamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  // needs jet collection (srcJet), leptons collection
  auto srcJet = iEvent.getHandle(srcJet_);
  const auto &eleProd = iEvent.get(srcEle_);
  const auto &muProd = iEvent.get(srcMu_);

  unsigned int nJet = srcJet->size();
  unsigned int nEle = eleProd.size();
  unsigned int nMu = muProd.size();

  std::vector<float> vlsf3;
  std::vector<int> vmuIdx3SJ;
  std::vector<int> veleIdx3SJ;

  // Find leptons in jets
  for (unsigned int ij = 0; ij < nJet; ij++) {
    const pat::Jet &itJet = (*srcJet)[ij];
    if (itJet.pt() <= 10)
      continue;
    std::vector<fastjet::PseudoJet> lClusterParticles;
    float lepPt(-1), lepEta(-1), lepPhi(-1);
    int lepId(-1);
    for (auto const &d : itJet.daughterPtrVector()) {
      fastjet::PseudoJet p(d->px(), d->py(), d->pz(), d->energy());
      lClusterParticles.emplace_back(p);
    }
    int ele_pfmatch_index = -1;
    int mu_pfmatch_index = -1;

    // match to leading and closest electron or muon
    double dRmin(0.8), dRele(999), dRmu(999), dRtmp(999);
    for (unsigned int il(0); il < nEle; il++) {
      const auto &lep = eleProd.at(il);
      if (matchByCommonSourceCandidatePtr(lep, itJet)) {
        dRtmp = reco::deltaR(itJet.eta(), itJet.phi(), lep.eta(), lep.phi());
        if (dRtmp < dRmin && dRtmp < dRele && lep.pt() > lepPt) {
          lepPt = lep.pt();
          lepEta = lep.eta();
          lepPhi = lep.phi();
          lepId = 11;
          ele_pfmatch_index = il;
          dRele = dRtmp;
          break;
        }
      }
    }
    for (unsigned int il(0); il < nMu; il++) {
      const auto &lep = muProd.at(il);
      if (matchByCommonSourceCandidatePtr(lep, itJet)) {
        dRtmp = reco::deltaR(itJet.eta(), itJet.phi(), lep.eta(), lep.phi());
        if (dRtmp < dRmin && dRtmp < dRele && dRtmp < dRmu && lep.pt() > lepPt) {
          lepPt = lep.pt();
          lepEta = lep.eta();
          lepPhi = lep.phi();
          lepId = 13;
          ele_pfmatch_index = -1;
          mu_pfmatch_index = il;
          dRmu = dRtmp;
          break;
        }
      }
    }

    std::vector<fastjet::PseudoJet> psub_3;
    std::sort(lClusterParticles.begin(), lClusterParticles.end(), orderPseudoJet);
    auto lsf_3 = calculateLSF(lClusterParticles, psub_3, lepPt, lepEta, lepPhi, lepId, 2.0, 3);
    vlsf3.push_back(std::get<0>(lsf_3));
    veleIdx3SJ.push_back(ele_pfmatch_index);
    vmuIdx3SJ.push_back(mu_pfmatch_index);
  }

  // Filling table
  auto lsf3V = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler fillerlsf3(*lsf3V);
  fillerlsf3.insert(srcJet, vlsf3.begin(), vlsf3.end());
  fillerlsf3.fill();
  iEvent.put(std::move(lsf3V), "lsf3");

  auto muIdx3SJV = std::make_unique<edm::ValueMap<int>>();
  edm::ValueMap<int>::Filler fillermuIdx3SJ(*muIdx3SJV);
  fillermuIdx3SJ.insert(srcJet, vmuIdx3SJ.begin(), vmuIdx3SJ.end());
  fillermuIdx3SJ.fill();
  iEvent.put(std::move(muIdx3SJV), "muIdx3SJ");

  auto eleIdx3SJV = std::make_unique<edm::ValueMap<int>>();
  edm::ValueMap<int>::Filler fillereleIdx3SJ(*eleIdx3SJV);
  fillereleIdx3SJ.insert(srcJet, veleIdx3SJ.begin(), veleIdx3SJ.end());
  fillereleIdx3SJ.fill();
  iEvent.put(std::move(eleIdx3SJV), "eleIdx3SJ");
}

template <typename T>
bool LeptonInJetProducer<T>::orderPseudoJet(fastjet::PseudoJet j1, fastjet::PseudoJet j2) {
  return j1.perp2() > j2.perp2();
}

template <typename T>
std::tuple<float, float> LeptonInJetProducer<T>::calculateLSF(std::vector<fastjet::PseudoJet> iCParticles,
                                                              std::vector<fastjet::PseudoJet> &lsubjets,
                                                              float ilPt,
                                                              float ilEta,
                                                              float ilPhi,
                                                              int ilId,
                                                              double dr,
                                                              int nsj) const {
  float lsf(-1), lmd(-1);
  if (ilPt > 0 && (ilId == 11 || ilId == 13)) {
    TLorentzVector ilep;
    if (ilId == 11)
      ilep.SetPtEtaPhiM(ilPt, ilEta, ilPhi, 0.000511);
    if (ilId == 13)
      ilep.SetPtEtaPhiM(ilPt, ilEta, ilPhi, 0.105658);
    fastjet::JetDefinition lCJet_def(fastjet::kt_algorithm, dr);
    fastjet::ClusterSequence lCClust_seq(iCParticles, lCJet_def);
    if (dr > 0.5) {
      lsubjets = sorted_by_pt(lCClust_seq.exclusive_jets_up_to(nsj));
    } else {
      lsubjets = sorted_by_pt(lCClust_seq.inclusive_jets());
    }
    int lId(-1);
    double dRmin = 999.;
    for (unsigned int i0 = 0; i0 < lsubjets.size(); i0++) {
      double dR = reco::deltaR(lsubjets[i0].eta(), lsubjets[i0].phi(), ilep.Eta(), ilep.Phi());
      if (dR < dRmin) {
        dRmin = dR;
        lId = i0;
      }
    }
    if (lId != -1) {
      TLorentzVector pVec;
      pVec.SetPtEtaPhiM(lsubjets[lId].pt(), lsubjets[lId].eta(), lsubjets[lId].phi(), lsubjets[lId].m());
      lsf = ilep.Pt() / pVec.Pt();
      lmd = (ilep - pVec).M() / pVec.M();
    }
  }
  return std::tuple<float, float>(lsf, lmd);
}

template <typename T>
void LeptonInJetProducer<T>::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src")->setComment("jet input collection");
  desc.add<edm::InputTag>("srcEle")->setComment("electron input collection");
  desc.add<edm::InputTag>("srcMu")->setComment("muon input collection");
  descriptions.addWithDefaultLabel(desc);
}

typedef LeptonInJetProducer<pat::Jet> LepInJetProducer;

//define this as a plug-in
DEFINE_FWK_MODULE(LepInJetProducer);
