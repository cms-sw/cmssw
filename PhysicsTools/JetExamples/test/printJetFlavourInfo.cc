// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetMatching/interface/JetFlavourInfo.h"
#include "DataFormats/JetMatching/interface/JetFlavourInfoMatching.h"

#include "DataFormats/Math/interface/deltaR.h"

// system include files
#include <memory>

class printJetFlavourInfo : public edm::one::EDAnalyzer<> {
public:
  explicit printJetFlavourInfo(const edm::ParameterSet&);
  ~printJetFlavourInfo(){};
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

private:
  edm::EDGetTokenT<reco::JetFlavourInfoMatchingCollection> jetFlavourInfosToken_;
  edm::EDGetTokenT<reco::JetFlavourInfoMatchingCollection> subjetFlavourInfosToken_;
  edm::EDGetTokenT<edm::View<reco::Jet> > groomedJetsToken_;
  bool useSubjets_;
};

printJetFlavourInfo::printJetFlavourInfo(const edm::ParameterSet& iConfig) {
  jetFlavourInfosToken_ =
      consumes<reco::JetFlavourInfoMatchingCollection>(iConfig.getParameter<edm::InputTag>("jetFlavourInfos"));
  subjetFlavourInfosToken_ = mayConsume<reco::JetFlavourInfoMatchingCollection>(
      iConfig.exists("subjetFlavourInfos") ? iConfig.getParameter<edm::InputTag>("subjetFlavourInfos")
                                           : edm::InputTag());
  groomedJetsToken_ = mayConsume<edm::View<reco::Jet> >(
      iConfig.exists("groomedJets") ? iConfig.getParameter<edm::InputTag>("groomedJets") : edm::InputTag());
  useSubjets_ = (iConfig.exists("subjetFlavourInfos") && iConfig.exists("groomedJets"));
}

void printJetFlavourInfo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<reco::JetFlavourInfoMatchingCollection> theJetFlavourInfos;
  iEvent.getByToken(jetFlavourInfosToken_, theJetFlavourInfos);

  edm::Handle<reco::JetFlavourInfoMatchingCollection> theSubjetFlavourInfos;
  edm::Handle<edm::View<reco::Jet> > groomedJets;

  std::vector<int> matchedIndices;
  if (useSubjets_) {
    iEvent.getByToken(subjetFlavourInfosToken_, theSubjetFlavourInfos);
    iEvent.getByToken(groomedJetsToken_, groomedJets);

    // match groomed and original jet
    std::vector<bool> jetLocks(theJetFlavourInfos->size(), false);
    std::vector<int> jetIndices;

    for (size_t gj = 0; gj < groomedJets->size(); ++gj) {
      double matchedDR2 = 1e9;
      int matchedIdx = -1;

      if (groomedJets->at(gj).pt() > 0.)  // skips pathological cases of groomed jets with Pt=0
      {
        for (reco::JetFlavourInfoMatchingCollection::const_iterator j = theJetFlavourInfos->begin();
             j != theJetFlavourInfos->end();
             ++j) {
          if (jetLocks.at(j - theJetFlavourInfos->begin()))
            continue;  // skip jets that have already been matched

          double tempDR2 = reco::deltaR2(
              j->first->rapidity(), j->first->phi(), groomedJets->at(gj).rapidity(), groomedJets->at(gj).phi());
          if (tempDR2 < matchedDR2) {
            matchedDR2 = tempDR2;
            matchedIdx = (j - theJetFlavourInfos->begin());
          }
        }
      }

      if (matchedIdx >= 0)
        jetLocks.at(matchedIdx) = true;
      jetIndices.push_back(matchedIdx);
    }

    for (size_t j = 0; j < theJetFlavourInfos->size(); ++j) {
      std::vector<int>::iterator matchedIndex = std::find(jetIndices.begin(), jetIndices.end(), j);

      matchedIndices.push_back(matchedIndex != jetIndices.end() ? std::distance(jetIndices.begin(), matchedIndex) : -1);
    }
  }

  for (reco::JetFlavourInfoMatchingCollection::const_iterator j = theJetFlavourInfos->begin();
       j != theJetFlavourInfos->end();
       ++j) {
    std::cout << "-------------------- Jet Flavour Info --------------------" << std::endl;

    const reco::Jet* aJet = (*j).first.get();
    reco::JetFlavourInfo aInfo = (*j).second;
    std::cout << std::setprecision(2) << std::setw(6) << std::fixed << "[printJetFlavourInfo] Jet "
              << (j - theJetFlavourInfos->begin()) << " pt, eta, rapidity, phi = " << aJet->pt() << ", " << aJet->eta()
              << ", " << aJet->rapidity() << ", " << aJet->phi() << std::endl;
    // ----------------------- Hadrons -------------------------------
    std::cout << "                      Hadron-based flavour: " << aInfo.getHadronFlavour() << std::endl;

    const reco::GenParticleRefVector& bHadrons = aInfo.getbHadrons();
    std::cout << "                      # of clustered b hadrons: " << bHadrons.size() << std::endl;
    for (reco::GenParticleRefVector::const_iterator it = bHadrons.begin(); it != bHadrons.end(); ++it) {
      float dist = reco::deltaR(aJet->eta(), aJet->phi(), (*it)->eta(), (*it)->phi());
      float dist2 = reco::deltaR(aJet->rapidity(), aJet->phi(), (*it)->rapidity(), (*it)->phi());
      std::cout << "                        b hadron " << (it - bHadrons.begin())
                << " PdgID, status, (pt,eta,rapidity,phi), dR(eta-phi), dR(rap-phi) = " << (*it)->pdgId() << ", "
                << (*it)->status() << ", (" << (*it)->pt() << "," << (*it)->eta() << "," << (*it)->rapidity() << ","
                << (*it)->phi() << "), " << dist << ", " << dist2 << std::endl;
    }

    const reco::GenParticleRefVector& cHadrons = aInfo.getcHadrons();
    std::cout << "                      # of clustered c hadrons: " << cHadrons.size() << std::endl;
    for (reco::GenParticleRefVector::const_iterator it = cHadrons.begin(); it != cHadrons.end(); ++it) {
      float dist = reco::deltaR(aJet->eta(), aJet->phi(), (*it)->eta(), (*it)->phi());
      float dist2 = reco::deltaR(aJet->rapidity(), aJet->phi(), (*it)->rapidity(), (*it)->phi());
      std::cout << "                        c hadron " << (it - cHadrons.begin())
                << " PdgID, status, (pt,eta,rapidity,phi), dR(eta-phi), dR(rap-phi) = " << (*it)->pdgId() << ", "
                << (*it)->status() << ", (" << (*it)->pt() << "," << (*it)->eta() << "," << (*it)->rapidity() << ","
                << (*it)->phi() << "), " << dist << ", " << dist2 << std::endl;
    }
    // ----------------------- Partons -------------------------------
    std::cout << "                      Parton-based flavour: " << aInfo.getPartonFlavour() << std::endl;

    const reco::GenParticleRefVector& partons = aInfo.getPartons();
    std::cout << "                      # of clustered partons: " << partons.size() << std::endl;
    for (reco::GenParticleRefVector::const_iterator it = partons.begin(); it != partons.end(); ++it) {
      float dist = reco::deltaR(aJet->eta(), aJet->phi(), (*it)->eta(), (*it)->phi());
      float dist2 = reco::deltaR(aJet->rapidity(), aJet->phi(), (*it)->rapidity(), (*it)->phi());
      std::cout << "                        Parton " << (it - partons.begin())
                << " PdgID, status, (pt,eta,rapidity,phi), dR(eta-phi), dR(rap-phi) = " << (*it)->pdgId() << ", "
                << (*it)->status() << ", (" << (*it)->pt() << "," << (*it)->eta() << "," << (*it)->rapidity() << ","
                << (*it)->phi() << "), " << dist << ", " << dist2 << std::endl;
    }
    // ----------------------- Leptons -------------------------------
    const reco::GenParticleRefVector& leptons = aInfo.getLeptons();
    std::cout << "                      # of clustered leptons: " << leptons.size() << std::endl;
    for (reco::GenParticleRefVector::const_iterator it = leptons.begin(); it != leptons.end(); ++it) {
      float dist = reco::deltaR(aJet->eta(), aJet->phi(), (*it)->eta(), (*it)->phi());
      float dist2 = reco::deltaR(aJet->rapidity(), aJet->phi(), (*it)->rapidity(), (*it)->phi());
      std::cout << "                        Lepton " << (it - leptons.begin())
                << " PdgID, status, (pt,eta,rapidity,phi), dR(eta-phi), dR(rap-phi) = " << (*it)->pdgId() << ", "
                << (*it)->status() << ", (" << (*it)->pt() << "," << (*it)->eta() << "," << (*it)->rapidity() << ","
                << (*it)->phi() << "), " << dist << ", " << dist2 << std::endl;
    }

    if (useSubjets_) {
      if (matchedIndices.at(j - theJetFlavourInfos->begin()) < 0) {
        std::cout << "  ----------------------- Subjet Flavour Info -----------------------" << std::endl;
        std::cout << "  No subjets assigned to this jet" << std::endl;
        continue;
      }

      // loop over subjets
      std::cout << "  ----------------------- Subjet Flavour Info -----------------------" << std::endl;
      for (size_t s = 0; s < groomedJets->at(matchedIndices.at(j - theJetFlavourInfos->begin())).numberOfDaughters();
           ++s) {
        const edm::Ptr<reco::Candidate>& subjet =
            groomedJets->at(matchedIndices.at(j - theJetFlavourInfos->begin())).daughterPtr(s);

        for (reco::JetFlavourInfoMatchingCollection::const_iterator sj = theSubjetFlavourInfos->begin();
             sj != theSubjetFlavourInfos->end();
             ++sj) {
          if (subjet != edm::Ptr<reco::Candidate>((*sj).first.id(), (*sj).first.get(), (*sj).first.key()))
            continue;

          const reco::Jet* aSubjet = (*sj).first.get();
          aInfo = (*sj).second;
          std::cout << std::setprecision(2) << std::setw(6) << std::fixed << "  [printSubjetFlavourInfo] Subjet " << s
                    << " pt, eta, rapidity, phi, dR(eta-phi), dR(rap-phi) = " << aSubjet->pt() << ", " << aSubjet->eta()
                    << ", " << aSubjet->rapidity() << ", " << aSubjet->phi() << ", "
                    << reco::deltaR(aSubjet->eta(), aSubjet->phi(), aJet->eta(), aJet->phi()) << ", "
                    << reco::deltaR(aSubjet->rapidity(), aSubjet->phi(), aJet->rapidity(), aJet->phi()) << std::endl;
          // ----------------------- Hadrons -------------------------------
          std::cout << "                           Hadron-based flavour: " << aInfo.getHadronFlavour() << std::endl;

          const reco::GenParticleRefVector& bHadrons = aInfo.getbHadrons();
          std::cout << "                           # of assigned b hadrons: " << bHadrons.size() << std::endl;
          for (reco::GenParticleRefVector::const_iterator it = bHadrons.begin(); it != bHadrons.end(); ++it) {
            float dist = reco::deltaR(aSubjet->eta(), aSubjet->phi(), (*it)->eta(), (*it)->phi());
            float dist2 = reco::deltaR(aSubjet->rapidity(), aSubjet->phi(), (*it)->rapidity(), (*it)->phi());
            std::cout << "                             b hadron " << (it - bHadrons.begin())
                      << " PdgID, status, (pt,eta,rapidity,phi), dR(eta-phi), dR(rap-phi) = " << (*it)->pdgId() << ", "
                      << (*it)->status() << ", (" << (*it)->pt() << "," << (*it)->eta() << "," << (*it)->rapidity()
                      << "," << (*it)->phi() << "), " << dist << ", " << dist2 << std::endl;
          }

          const reco::GenParticleRefVector& cHadrons = aInfo.getcHadrons();
          std::cout << "                           # of assigned c hadrons: " << cHadrons.size() << std::endl;
          for (reco::GenParticleRefVector::const_iterator it = cHadrons.begin(); it != cHadrons.end(); ++it) {
            float dist = reco::deltaR(aSubjet->eta(), aSubjet->phi(), (*it)->eta(), (*it)->phi());
            float dist2 = reco::deltaR(aSubjet->rapidity(), aSubjet->phi(), (*it)->rapidity(), (*it)->phi());
            std::cout << "                             c hadron " << (it - cHadrons.begin())
                      << " PdgID, status, (pt,eta,rapidity,phi), dR(eta-phi), dR(rap-phi) = " << (*it)->pdgId() << ", "
                      << (*it)->status() << ", (" << (*it)->pt() << "," << (*it)->eta() << "," << (*it)->rapidity()
                      << "," << (*it)->phi() << "), " << dist << ", " << dist2 << std::endl;
          }
          // ----------------------- Partons -------------------------------
          std::cout << "                           Parton-based flavour: " << aInfo.getPartonFlavour() << std::endl;

          const reco::GenParticleRefVector& partons = aInfo.getPartons();
          std::cout << "                           # of assigned partons: " << partons.size() << std::endl;
          for (reco::GenParticleRefVector::const_iterator it = partons.begin(); it != partons.end(); ++it) {
            float dist = reco::deltaR(aSubjet->eta(), aSubjet->phi(), (*it)->eta(), (*it)->phi());
            float dist2 = reco::deltaR(aSubjet->rapidity(), aSubjet->phi(), (*it)->rapidity(), (*it)->phi());
            std::cout << "                             Parton " << (it - partons.begin())
                      << " PdgID, status, (pt,eta,rapidity,phi), dR(eta-phi), dR(rap-phi) = " << (*it)->pdgId() << ", "
                      << (*it)->status() << ", (" << (*it)->pt() << "," << (*it)->eta() << "," << (*it)->rapidity()
                      << "," << (*it)->phi() << "), " << dist << ", " << dist2 << std::endl;
          }
          // ----------------------- Leptons -------------------------------
          const reco::GenParticleRefVector& leptons = aInfo.getLeptons();
          std::cout << "                           # of assigned leptons: " << leptons.size() << std::endl;
          for (reco::GenParticleRefVector::const_iterator it = leptons.begin(); it != leptons.end(); ++it) {
            float dist = reco::deltaR(aSubjet->eta(), aSubjet->phi(), (*it)->eta(), (*it)->phi());
            float dist2 = reco::deltaR(aSubjet->rapidity(), aSubjet->phi(), (*it)->rapidity(), (*it)->phi());
            std::cout << "                             Lepton " << (it - leptons.begin())
                      << " PdgID, status, (pt,eta,rapidity,phi), dR(eta-phi), dR(rap-phi) = " << (*it)->pdgId() << ", "
                      << (*it)->status() << ", (" << (*it)->pt() << "," << (*it)->eta() << "," << (*it)->rapidity()
                      << "," << (*it)->phi() << "), " << dist << ", " << dist2 << std::endl;
          }
        }
      }
    }
  }
}

DEFINE_FWK_MODULE(printJetFlavourInfo);
