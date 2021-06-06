// -*- C++ -*-
//
// Package:    ProdTutorial/L1ECALPrefiringWeightProducer
// Class:      L1ECALPrefiringWeightProducer
//
/**\class L1ECALPrefiringWeightProducer L1ECALPrefiringWeightProducer.cc ProdTutorial/L1ECALPrefiringWeightProducer/plugins/L1ECALPrefiringWeightProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  localusers user
//         Created:  Thu, 08 Nov 2018 16:16:00 GMT
//
//

// system include files
#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "TH2.h"

#include <iostream>
enum fluctuations { central = 0, up, down };

class L1ECALPrefiringWeightProducer : public edm::global::EDProducer<> {
public:
  explicit L1ECALPrefiringWeightProducer(const edm::ParameterSet&);
  ~L1ECALPrefiringWeightProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  double getPrefiringRate(double eta, double pt, TH2F* h_prefmap, fluctuations fluctuation) const;

  edm::EDGetTokenT<std::vector<pat::Photon> > photons_token_;
  edm::EDGetTokenT<std::vector<pat::Jet> > jets_token_;

  TH2F* h_prefmap_photon;
  TH2F* h_prefmap_jet;
  std::string dataera_;
  bool useEMpt_;
  double prefiringRateSystUnc_;
  double jetMaxMuonFraction_;
  bool skipwarnings_;
};

L1ECALPrefiringWeightProducer::L1ECALPrefiringWeightProducer(const edm::ParameterSet& iConfig) {
  photons_token_ = consumes<std::vector<pat::Photon> >(iConfig.getParameter<edm::InputTag>("ThePhotons"));
  jets_token_ = consumes<std::vector<pat::Jet> >(iConfig.getParameter<edm::InputTag>("TheJets"));

  dataera_ = iConfig.getParameter<std::string>("DataEra");
  useEMpt_ = iConfig.getParameter<bool>("UseJetEMPt");
  prefiringRateSystUnc_ = iConfig.getParameter<double>("PrefiringRateSystematicUncty");
  jetMaxMuonFraction_ = iConfig.getParameter<double>("JetMaxMuonFraction");
  skipwarnings_ = iConfig.getParameter<bool>("SkipWarnings");

  TFile* file_prefiringmaps_;
  std::string fname = iConfig.getParameter<std::string>("L1Maps");
  edm::FileInPath mapsfilepath("PhysicsTools/PatUtils/data/" + fname);
  file_prefiringmaps_ = new TFile(mapsfilepath.fullPath().c_str(), "read");
  if (file_prefiringmaps_ == nullptr && !skipwarnings_)
    std::cout << "File with maps not found. All prefiring weights set to 0. " << std::endl;

  TString mapphotonfullname = "L1prefiring_photonptvseta_" + dataera_;
  if (!file_prefiringmaps_->Get(mapphotonfullname) && !skipwarnings_)
    std::cout << "Photon map not found. All photons prefiring weights set to 0. " << std::endl;
  h_prefmap_photon = (TH2F*)file_prefiringmaps_->Get(mapphotonfullname);

  TString mapjetfullname = (useEMpt_) ? "L1prefiring_jetemptvseta_" + dataera_ : "L1prefiring_jetptvseta_" + dataera_;
  if (!file_prefiringmaps_->Get(mapjetfullname) && !skipwarnings_)
    std::cout << "Jet map not found. All jets prefiring weights set to 0. " << std::endl;
  h_prefmap_jet = (TH2F*)file_prefiringmaps_->Get(mapjetfullname);
  file_prefiringmaps_->Close();
  delete file_prefiringmaps_;
  produces<double>("nonPrefiringProb").setBranchAlias("nonPrefiringProb");
  produces<double>("nonPrefiringProbUp").setBranchAlias("nonPrefiringProbUp");
  produces<double>("nonPrefiringProbDown").setBranchAlias("nonPrefiringProbDown");
}

L1ECALPrefiringWeightProducer::~L1ECALPrefiringWeightProducer() {
  delete h_prefmap_photon;
  delete h_prefmap_jet;
}

void L1ECALPrefiringWeightProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  //Photons
  edm::Handle<std::vector<pat::Photon> > thePhotons;
  iEvent.getByToken(photons_token_, thePhotons);

  //Jets
  edm::Handle<std::vector<pat::Jet> > theJets;
  iEvent.getByToken(jets_token_, theJets);

  //Probability for the event NOT to prefire, computed with the prefiring maps per object.
  //Up and down values correspond to the resulting value when shifting up/down all prefiring rates in prefiring maps.
  double nonPrefiringProba[3] = {1., 1., 1.};  //0: central, 1: up, 2: down

  for (const auto fluct : {fluctuations::central, fluctuations::up, fluctuations::down}) {
    for (const auto& photon : *thePhotons) {
      double pt_gam = photon.pt();
      double eta_gam = photon.eta();
      if (pt_gam < 20.)
        continue;
      if (fabs(eta_gam) < 2.)
        continue;
      if (fabs(eta_gam) > 3.)
        continue;
      double prefiringprob_gam = getPrefiringRate(eta_gam, pt_gam, h_prefmap_photon, fluct);
      nonPrefiringProba[fluct] *= (1. - prefiringprob_gam);
    }

    //Now applying the prefiring maps to jets in the affected regions.
    for (const auto& jet : *theJets) {
      double pt_jet = jet.pt();
      double eta_jet = jet.eta();
      double phi_jet = jet.phi();
      if (pt_jet < 20.)
        continue;
      if (fabs(eta_jet) < 2.)
        continue;
      if (fabs(eta_jet) > 3.)
        continue;
      if (jetMaxMuonFraction_ > 0 && jet.muonEnergyFraction() > jetMaxMuonFraction_)
        continue;
      //Loop over photons to remove overlap
      double nonprefiringprobfromoverlappingphotons = 1.;
      for (const auto& photon : *thePhotons) {
        double pt_gam = photon.pt();
        double eta_gam = photon.eta();
        double phi_gam = photon.phi();
        if (pt_gam < 20.)
          continue;
        if (fabs(eta_gam) < 2.)
          continue;
        if (fabs(eta_gam) > 3.)
          continue;
        double dR = reco::deltaR(eta_jet, phi_jet, eta_gam, phi_gam);
        if (dR > 0.4)
          continue;
        double prefiringprob_gam = getPrefiringRate(eta_gam, pt_gam, h_prefmap_photon, fluct);
        nonprefiringprobfromoverlappingphotons *= (1. - prefiringprob_gam);
      }
      //useEMpt =true if one wants to use maps parametrized vs Jet EM pt instead of pt.
      if (useEMpt_)
        pt_jet *= (jet.neutralEmEnergyFraction() + jet.chargedEmEnergyFraction());
      double nonprefiringprobfromoverlappingjet = 1. - getPrefiringRate(eta_jet, pt_jet, h_prefmap_jet, fluct);

      if (nonprefiringprobfromoverlappingphotons == 1.) {
        nonPrefiringProba[fluct] *= nonprefiringprobfromoverlappingjet;
      }
      //If overlapping photons have a non prefiring rate larger than the jet, then replace these weights by the jet one
      else if (nonprefiringprobfromoverlappingphotons > nonprefiringprobfromoverlappingjet) {
        if (nonprefiringprobfromoverlappingphotons != 0.) {
          nonPrefiringProba[fluct] *= nonprefiringprobfromoverlappingjet / nonprefiringprobfromoverlappingphotons;
        } else {
          nonPrefiringProba[fluct] = 0.;
        }
      }
      //Last case: if overlapping photons have a non prefiring rate smaller than the jet, don't consider the jet in the event weight, and do nothing.
    }
  }

  auto nonPrefiringProb = std::make_unique<double>(nonPrefiringProba[0]);
  auto nonPrefiringProbUp = std::make_unique<double>(nonPrefiringProba[1]);
  auto nonPrefiringProbDown = std::make_unique<double>(nonPrefiringProba[2]);
  iEvent.put(std::move(nonPrefiringProb), "nonPrefiringProb");
  iEvent.put(std::move(nonPrefiringProbUp), "nonPrefiringProbUp");
  iEvent.put(std::move(nonPrefiringProbDown), "nonPrefiringProbDown");
}

double L1ECALPrefiringWeightProducer::getPrefiringRate(double eta,
                                                       double pt,
                                                       TH2F* h_prefmap,
                                                       fluctuations fluctuation) const {
  if (h_prefmap == nullptr && !skipwarnings_)
    std::cout << "Prefiring map not found, setting prefiring rate to 0 " << std::endl;
  if (h_prefmap == nullptr)
    return 0.;
  //Check pt is not above map overflow
  int nbinsy = h_prefmap->GetNbinsY();
  double maxy = h_prefmap->GetYaxis()->GetBinLowEdge(nbinsy + 1);
  if (pt >= maxy)
    pt = maxy - 0.01;
  int thebin = h_prefmap->FindBin(eta, pt);

  double prefrate = h_prefmap->GetBinContent(thebin);

  double statuncty = h_prefmap->GetBinError(thebin);
  double systuncty = prefiringRateSystUnc_ * prefrate;

  if (fluctuation == up)
    prefrate = std::min(1., prefrate + sqrt(pow(statuncty, 2) + pow(systuncty, 2)));
  if (fluctuation == down)
    prefrate = std::max(0., prefrate - sqrt(pow(statuncty, 2) + pow(systuncty, 2)));
  return prefrate;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1ECALPrefiringWeightProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("ThePhotons", edm::InputTag("slimmedPhotons"));
  desc.add<edm::InputTag>("TheJets", edm::InputTag("slimmedJets"));
  desc.add<std::string>("L1Maps", "L1PrefiringMaps.root");
  desc.add<std::string>("DataEra", "2017BtoF");
  desc.add<bool>("UseJetEMPt", false);
  desc.add<double>("PrefiringRateSystematicUncty", 0.2);
  desc.add<double>("JetMaxMuonFraction", 0.5);
  desc.add<bool>("SkipWarnings", true);
  descriptions.add("l1ECALPrefiringWeightProducer", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1ECALPrefiringWeightProducer);
