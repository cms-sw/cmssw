// Package:    SinglePhotonJetPlusHOFilter
// Class:      SinglePhotonJetPlusHOFilter
//
/*
 Description: [one line class summary]
Skimming of SinglePhoton data set for the study of HO absolute weight calculation
* Skimming Efficiency : ~ 2 %
 Implementation:
     [Notes on implementation]
     For Secondary Datasets (SD)
*/
//
// Original Author:  Gobinda Majumder & Suman Chatterjee
//         Created:  Fri July 29 14:52:17 IST 2016
// $Id$
//
//

// system include files
#include <memory>

// class declaration
//

#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

class SinglePhotonJetPlusHOFilter : public edm::global::EDFilter<> {
public:
  explicit SinglePhotonJetPlusHOFilter(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  const double jtptthr_;
  const double jtetath_;
  const double hothres_;
  const double pho_Ptcut_;

  const edm::EDGetTokenT<reco::PFJetCollection> tok_PFJets_;
  const edm::EDGetTokenT<reco::PFClusterCollection> tok_hoht_;
  const edm::EDGetTokenT<edm::View<reco::Photon> > tok_photons_;
};

SinglePhotonJetPlusHOFilter::SinglePhotonJetPlusHOFilter(const edm::ParameterSet& iConfig)
    : jtptthr_{iConfig.getParameter<double>("Ptcut")},
      jtetath_{iConfig.getParameter<double>("Etacut")},
      hothres_{iConfig.getParameter<double>("HOcut")},
      pho_Ptcut_{iConfig.getParameter<double>("Pho_Ptcut")},
      tok_PFJets_{consumes<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("PFJets"))},
      tok_hoht_{consumes<reco::PFClusterCollection>(iConfig.getParameter<edm::InputTag>("particleFlowClusterHO"))},
      tok_photons_{consumes<edm::View<reco::Photon> >(iConfig.getParameter<edm::InputTag>("Photons"))} {}

// ------------ method called on each new Event  ------------
bool SinglePhotonJetPlusHOFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace std;
  using namespace edm;
  using namespace reco;

  bool passed = false;
  vector<pair<double, double> > jetdirection;
  vector<double> jetspt;
  if (auto PFJets = iEvent.getHandle(tok_PFJets_)) {
    for (const auto& jet : *PFJets) {
      if ((jet.pt() < jtptthr_) || (abs(jet.eta()) > jtetath_))
        continue;

      jetdirection.emplace_back(jet.eta(), jet.phi());
      jetspt.push_back(jet.pt());
      passed = true;
    }
  }
  if (!passed)
    return passed;

  bool pho_passed = false;
  vector<pair<double, double> > phodirection;
  vector<double> phopT;
  if (auto photons = iEvent.getHandle(tok_photons_)) {
    for (const auto& gamma1 : *photons) {
      if (gamma1.pt() < pho_Ptcut_)
        continue;
      phodirection.emplace_back(gamma1.eta(), gamma1.phi());
      phopT.push_back(gamma1.pt());

      pho_passed = true;
    }
  }

  if (!pho_passed)
    return false;

  bool isJetDir = false;
  if (auto hhoht = iEvent.getHandle(tok_hoht_)) {
    const auto& hoht = *hhoht;
    if (!hoht.empty()) {
      for (const auto& jet : jetdirection) {
        bool matched = false;
        for (const auto& ph : phodirection) {
          if (abs(deltaPhi(ph.second, jet.second)) > 2.0) {
            matched = true;
            break;
          }
        }
        if (matched) {
          for (const auto& ij : hoht) {
            double hoenr = ij.energy();
            if (hoenr < hothres_)
              continue;

            const math::XYZPoint& cluster_pos = ij.position();

            double hoeta = cluster_pos.eta();
            double hophi = cluster_pos.phi();

            double delta = deltaR2(jet.first, jet.second, hoeta, hophi);
            if (delta < 0.5) {
              isJetDir = true;
              break;
            }
          }
        }
        if (isJetDir) {
          break;
        }
      }
    }
  }

  return isJetDir;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SinglePhotonJetPlusHOFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<double>("Ptcut", 90.0);
  desc.add<double>("Etacut", 1.5);
  desc.add<double>("HOcut", 5);
  desc.add<double>("Pho_Ptcut", 120);
  desc.add<edm::InputTag>("PFJets");
  desc.add<edm::InputTag>("particleFlowClusterHO");
  desc.add<edm::InputTag>("Photons");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SinglePhotonJetPlusHOFilter);
