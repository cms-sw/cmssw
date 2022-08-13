// -*- C++ -*-
//
// Package:    JetPlusTracks
// Class:      JetPlusTrackAddonSeedProducer
//
/**\class JetPlusTrackAddonSeedProducer JetPlusTrackAddonSeedProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Olga Kodolova,40 R-A12,+41227671273,
//         Created:  Fri Feb 19 10:14:02 CET 2010
//
//

// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/TrackJetCollection.h"
#include "DataFormats/JetReco/interface/TrackJet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <string>

using namespace std;
class JetPlusTrackAddonSeedProducer : public edm::stream::EDProducer<> {
public:
  explicit JetPlusTrackAddonSeedProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  const double dRcone_;
  const bool usePAT_;
  const edm::EDGetTokenT<edm::View<reco::CaloJet> > input_jets_token_;
  const edm::EDGetTokenT<edm::View<reco::TrackJet> > input_trackjets_token_;
  const edm::EDGetTokenT<reco::VertexCollection> input_vertex_token_;
  const edm::EDGetTokenT<std::vector<pat::PackedCandidate> > tokenPFCandidates_;
  const edm::EDGetTokenT<CaloTowerCollection> input_ctw_token_;
};

JetPlusTrackAddonSeedProducer::JetPlusTrackAddonSeedProducer(const edm::ParameterSet& iConfig)
    : dRcone_(iConfig.getParameter<double>("dRcone")),
      usePAT_(iConfig.getParameter<bool>("UsePAT")),
      input_jets_token_(consumes<edm::View<reco::CaloJet> >(iConfig.getParameter<edm::InputTag>("srcCaloJets"))),
      input_trackjets_token_(consumes<edm::View<reco::TrackJet> >(iConfig.getParameter<edm::InputTag>("srcTrackJets"))),
      input_vertex_token_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("srcPVs"))),
      tokenPFCandidates_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("PFCandidates"))),
      input_ctw_token_(consumes<CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("towerMaker"))) {
  //register your products
  produces<reco::CaloJetCollection>();
}

void JetPlusTrackAddonSeedProducer::fillDescriptions(edm::ConfigurationDescriptions& iDescriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("dRcone", 0.4);
  desc.add<bool>("UsePAT", false);
  desc.add<edm::InputTag>("srcCaloJets", edm::InputTag("ak4CaloJets"));
  desc.add<edm::InputTag>("srcTrackJets", edm::InputTag("ak4TrackJets"));
  desc.add<edm::InputTag>("srcPVs", edm::InputTag("primaryVertex"));
  desc.add<edm::InputTag>("PFCandidates", edm::InputTag("PFCandidates"));
  desc.add<edm::InputTag>("towerMaker", edm::InputTag("towerMaker"));
  iDescriptions.addWithDefaultLabel(desc);
}

// ------------ method called to produce the data  ------------
void JetPlusTrackAddonSeedProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  // get stuff from Event
  auto const& jets = iEvent.get(input_jets_token_);
  auto const& jetsTrackJets = iEvent.get(input_trackjets_token_);
  auto pCaloOut = std::make_unique<reco::CaloJetCollection>();

  for (auto const& jet : jetsTrackJets) {
    int iflag = 0;
    for (auto const& oldjet : jets) {
      double dr2 = deltaR2(jet, oldjet);
      if (dr2 < dRcone_ * dRcone_) {
        iflag = 1;
      }
    }  // Calojets

    if (iflag == 1)
      continue;
    double caloen = 0.;
    double hadinho = 0.;
    double hadinhb = 0.;
    double hadinhe = 0.;
    double hadinhf = 0.;
    double emineb = 0.;
    double eminee = 0.;
    double eminhf = 0.;
    double eefraction = 0.;
    double hhfraction = 0.;
    int ncand = 0;

    if (usePAT_) {
      auto const& pfCandidates = iEvent.get(tokenPFCandidates_);
      for (auto const& pf : pfCandidates) {
        double dr2 = deltaR2(jet, pf);
        if (dr2 > dRcone_ * dRcone_)
          continue;
        // jetconstit
        caloen = caloen + pf.energy() * pf.caloFraction();
        hadinho += 0.;
        if (std::abs(pf.eta()) <= 1.4) {
          hadinhb += pf.energy() * pf.caloFraction() * pf.hcalFraction();
          emineb += pf.energy() * pf.caloFraction() * (1. - pf.hcalFraction());
        } else if (std::abs(pf.eta()) < 3.) {
          hadinhe += pf.energy() * pf.caloFraction() * pf.hcalFraction();
          eminee += pf.energy() * pf.caloFraction() * (1. - pf.hcalFraction());
        } else {
          hadinhf += pf.energy() * pf.caloFraction() * pf.hcalFraction();
          eminhf += pf.energy() * pf.caloFraction() * (1. - pf.hcalFraction());
        }
        ncand++;
      }  // pfcandidates
    } else {
      auto const& cts = iEvent.get(input_ctw_token_);
      for (auto const& ct : cts) {
        double dr2 = deltaR2(jet, ct);
        if (dr2 > dRcone_ * dRcone_)
          continue;
        caloen = caloen + ct.energy();
        hadinho += ct.energyInHO();
        hadinhb += ct.energyInHB();
        hadinhe += ct.energyInHE();
        hadinhf += 0.5 * ct.energyInHF();
        emineb += ct.energy() - ct.energyInHB() - ct.energyInHO();
        eminee += ct.energy() - ct.energyInHE();
        eminhf += 0.5 * ct.energyInHF();
        ncand++;
      }
    }
    eefraction = (emineb + eminee) / caloen;
    hhfraction = (hadinhb + hadinhe + hadinhf + hadinho) / caloen;

    double trackp = jet.p();
    if (caloen <= 0.)
      caloen = 0.001;
    math::XYZTLorentzVector pcalo4(caloen * jet.p4() / trackp);
    reco::CaloJet::Specific calospe;
    calospe.mTowersArea = ncand;
    calospe.mHadEnergyInHO = hadinho;
    calospe.mHadEnergyInHB = hadinhb;
    calospe.mHadEnergyInHE = hadinhe;
    calospe.mHadEnergyInHF = hadinhf;
    calospe.mEmEnergyInEB = emineb;
    calospe.mEmEnergyInEE = eminee;
    calospe.mEmEnergyInHF = eminhf;
    calospe.mEnergyFractionEm = eefraction / caloen;
    calospe.mEnergyFractionHadronic = hhfraction / caloen;

    reco::CaloJet mycalo(pcalo4, jet.primaryVertex()->position(), calospe);
    mycalo.setJetArea(M_PI * dRcone_ * dRcone_);
    pCaloOut->push_back(mycalo);

  }  // trackjets
  iEvent.put(std::move(pCaloOut));
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(JetPlusTrackAddonSeedProducer);
