// -*- C++ -*-
//
// Package:    ProdTutorial/L1PrefiringWeightProducer
// Class:      L1PrefiringWeightProducer
//
/**\class L1PrefiringWeightProducer L1PrefiringWeightProducer.cc ProdTutorial/L1ECALPrefiringWeightProducer/plugins/L1ECALPrefiringWeightProducer.cc

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
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "TFile.h"
#include "TF1.h"
#include "TH2.h"

#include <iostream>
enum fluctuations { central = 0, up, down };

class L1PrefiringWeightProducer : public edm::global::EDProducer<> {
public:
  explicit L1PrefiringWeightProducer(const edm::ParameterSet&);
  ~L1PrefiringWeightProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  double getPrefiringRate(double eta, double pt, TH2F* h_prefmap, fluctuations fluctuation) const;
  double getPrefiringRateMuon(double eta, double phi, double pt, fluctuations fluctuation) const;

  edm::EDGetTokenT<std::vector<pat::Photon> > photons_token_;
  edm::EDGetTokenT<std::vector<pat::Jet> > jets_token_;
  edm::EDGetTokenT<std::vector<pat::Muon> > muon_token_;

  TF1* parametrization0p0To0p2_;
  TF1* parametrization0p2To0p3_;
  TF1* parametrization0p3To0p55_;
  TF1* parametrization0p55To0p83_;
  TF1* parametrization0p83To1p24_;
  TF1* parametrization1p24To1p4_;
  TF1* parametrization1p4To1p6_;
  TF1* parametrization1p6To1p8_;
  TF1* parametrization1p8To2p1_;
  TF1* parametrization2p1To2p25_;
  TF1* parametrization2p25To2p4_;
  TF1* parametrizationHotSpot_;

  TH2F* h_prefmap_photon;
  TH2F* h_prefmap_jet;
  std::string dataera_;
  std::string dataeraMuon_;
  bool useEMpt_;
  double prefiringRateSystUnc_;
  double jetMaxMuonFraction_;
  bool skipwarnings_;
};

L1PrefiringWeightProducer::L1PrefiringWeightProducer(const edm::ParameterSet& iConfig) {
  photons_token_ = consumes<std::vector<pat::Photon> >(iConfig.getParameter<edm::InputTag>("ThePhotons"));
  jets_token_ = consumes<std::vector<pat::Jet> >(iConfig.getParameter<edm::InputTag>("TheJets"));
  muon_token_ = consumes<std::vector<pat::Muon> >(iConfig.getParameter<edm::InputTag>("TheMuons"));

  dataera_ = iConfig.getParameter<std::string>("DataEraECAL");
  dataeraMuon_ = iConfig.getParameter<std::string>("DataEraMuon");
  useEMpt_ = iConfig.getParameter<bool>("UseJetEMPt");
  prefiringRateSystUnc_ = iConfig.getParameter<double>("PrefiringRateSystematicUncty");
  jetMaxMuonFraction_ = iConfig.getParameter<double>("JetMaxMuonFraction");
  skipwarnings_ = iConfig.getParameter<bool>("SkipWarnings");

  TFile* file_prefiringmaps_;
  std::string fname = iConfig.getParameter<std::string>("L1Maps");
  edm::FileInPath mapsfilepath("PhysicsTools/PatUtils/data/" + fname);
  file_prefiringmaps_ = new TFile(mapsfilepath.fullPath().c_str(), "read");
  if (file_prefiringmaps_ == nullptr && !skipwarnings_)
    edm::LogWarning("L1PrefireWeightProducer") << "File with maps not found. All prefiring weights set to 0. " << std::endl;

  TString mapphotonfullname = "L1prefiring_photonptvseta_" + dataera_;
  if (!file_prefiringmaps_->Get(mapphotonfullname) && !skipwarnings_)
    edm::LogWarning("L1PrefireWeightProducer") << "Photon map not found. All photons prefiring weights set to 0. " << std::endl;
  h_prefmap_photon = (TH2F*)file_prefiringmaps_->Get(mapphotonfullname);

  TString mapjetfullname = (useEMpt_) ? "L1prefiring_jetemptvseta_" + dataera_ : "L1prefiring_jetptvseta_" + dataera_;
  if (!file_prefiringmaps_->Get(mapjetfullname) && !skipwarnings_)
    edm::LogWarning("L1PrefireWeightProducer") << "Jet map not found. All jets prefiring weights set to 0. " << std::endl;
  h_prefmap_jet = (TH2F*)file_prefiringmaps_->Get(mapjetfullname);
  file_prefiringmaps_->Close();
  delete file_prefiringmaps_;

  TFile* file_prefiringparams_;
  std::string fnameMuon = iConfig.getParameter<std::string>("L1MuonParametrizations");
  edm::FileInPath paramsfilepath("PhysicsTools/PatUtils/data/" + fnameMuon);
  file_prefiringparams_ = new TFile(paramsfilepath.fullPath().c_str(), "read");
  if (file_prefiringparams_ == nullptr && !skipwarnings_)
    edm::LogWarning("L1PrefireWeightProducer") << "File with muon parametrizations not found. All prefiring weights set to 0." << std::endl;

  TString paramName = "L1prefiring_muonparam_0.0To0.2_" + dataeraMuon_;
  parametrization0p0To0p2_ = (TF1*)file_prefiringparams_->Get(paramName);
  paramName = "L1prefiring_muonparam_0.2To0.3_" + dataeraMuon_;
  parametrization0p2To0p3_ = (TF1*)file_prefiringparams_->Get(paramName);
  paramName = "L1prefiring_muonparam_0.3To0.55_" + dataeraMuon_;
  parametrization0p3To0p55_ = (TF1*)file_prefiringparams_->Get(paramName);
  paramName = "L1prefiring_muonparam_0.55To0.83_" + dataeraMuon_;
  parametrization0p55To0p83_ = (TF1*)file_prefiringparams_->Get(paramName);
  paramName = "L1prefiring_muonparam_0.83To1.24_" + dataeraMuon_;
  parametrization0p83To1p24_ = (TF1*)file_prefiringparams_->Get(paramName);
  paramName = "L1prefiring_muonparam_1.24To1.4_" + dataeraMuon_;
  parametrization1p24To1p4_ = (TF1*)file_prefiringparams_->Get(paramName);
  paramName = "L1prefiring_muonparam_1.4To1.6_" + dataeraMuon_;
  parametrization1p4To1p6_ = (TF1*)file_prefiringparams_->Get(paramName);
  paramName = "L1prefiring_muonparam_1.6To1.8_" + dataeraMuon_;
  parametrization1p6To1p8_ = (TF1*)file_prefiringparams_->Get(paramName);
  paramName = "L1prefiring_muonparam_1.8To2.1_" + dataeraMuon_;
  parametrization1p8To2p1_ = (TF1*)file_prefiringparams_->Get(paramName);
  paramName = "L1prefiring_muonparam_2.1To2.25_" + dataeraMuon_;
  parametrization2p1To2p25_ = (TF1*)file_prefiringparams_->Get(paramName);
  paramName = "L1prefiring_muonparam_2.25To2.4_" + dataeraMuon_;
  parametrization2p25To2p4_ = (TF1*)file_prefiringparams_->Get(paramName);
  paramName = "L1prefiring_muonparam_HotSpot_" + dataeraMuon_;
  parametrizationHotSpot_ = (TF1*)file_prefiringparams_->Get(paramName);

  produces<double>("nonPrefiringProb").setBranchAlias("nonPrefiringProb");
  produces<double>("nonPrefiringProbUp").setBranchAlias("nonPrefiringProbUp");
  produces<double>("nonPrefiringProbDown").setBranchAlias("nonPrefiringProbDown");

  produces<double>("nonPrefiringProbJet").setBranchAlias("nonPrefiringProbJet");
  produces<double>("nonPrefiringProbJetUp").setBranchAlias("nonPrefiringProbJetUp");
  produces<double>("nonPrefiringProbJetDown").setBranchAlias("nonPrefiringProbJetDown");

  produces<double>("nonPrefiringProbPhoton").setBranchAlias("nonPrefiringProbPhoton");
  produces<double>("nonPrefiringProbPhotonUp").setBranchAlias("nonPrefiringProbPhotonUp");
  produces<double>("nonPrefiringProbPhotonDown").setBranchAlias("nonPrefiringProbPhotonDown");

  produces<double>("nonPrefiringProbMuon").setBranchAlias("nonPrefiringProbMuon");
  produces<double>("nonPrefiringProbMuonUp").setBranchAlias("nonPrefiringProbMuonUp");
  produces<double>("nonPrefiringProbMuonDown").setBranchAlias("nonPrefiringProbMuonDown");
}

L1PrefiringWeightProducer::~L1PrefiringWeightProducer() {
  delete h_prefmap_photon;
  delete h_prefmap_jet;
  delete parametrization0p0To0p2_;
  delete parametrization0p2To0p3_;
  delete parametrization0p3To0p55_;
  delete parametrization0p55To0p83_;
  delete parametrization0p83To1p24_;
  delete parametrization1p24To1p4_;
  delete parametrization1p4To1p6_;
  delete parametrization1p6To1p8_;
  delete parametrization1p8To2p1_;
  delete parametrization2p1To2p25_;
  delete parametrization2p25To2p4_;
  delete parametrizationHotSpot_;
}

void L1PrefiringWeightProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  //Photons
  std::vector<pat::Photon>  thePhotons = iEvent.get(photons_token_);

  //Jets
  std::vector<pat::Jet> theJets = iEvent.get(jets_token_);

  //Muons
  std::vector<pat::Muon> theMuons = iEvent.get(muon_token_);

  //Probability for the event NOT to prefire, computed with the prefiring maps per object.
  //Up and down values correspond to the resulting value when shifting up/down all prefiring rates in prefiring maps.
  double nonPrefiringProba[3] = {1., 1., 1.};        //0: central, 1: up, 2: down
  double nonPrefiringProbaJet[3] = {1., 1., 1.};     //0: central, 1: up, 2: down
  double nonPrefiringProbaPhoton[3] = {1., 1., 1.};  //0: central, 1: up, 2: down
  double nonPrefiringProbaMuon[3] = {1., 1., 1.};    //0: central, 1: up, 2: down

  for (const auto fluct : {fluctuations::central, fluctuations::up, fluctuations::down}) {
    for (const auto& photon : thePhotons) {
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
      nonPrefiringProbaPhoton[fluct] *= (1. - prefiringprob_gam);
    }

    //Now applying the prefiring maps to jets in the affected regions.
    for (const auto& jet : theJets) {
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
      for (const auto& photon : thePhotons) {
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
        nonPrefiringProbaJet[fluct] *= nonprefiringprobfromoverlappingjet;
      }
      //If overlapping photons have a non prefiring rate larger than the jet, then replace these weights by the jet one
      else if (nonprefiringprobfromoverlappingphotons > nonprefiringprobfromoverlappingjet) {
        if (nonprefiringprobfromoverlappingphotons != 0.) {
          nonPrefiringProba[fluct] *= nonprefiringprobfromoverlappingjet / nonprefiringprobfromoverlappingphotons;
          nonPrefiringProbaJet[fluct] *= nonprefiringprobfromoverlappingjet / nonprefiringprobfromoverlappingphotons;
        } else {
          nonPrefiringProba[fluct] = 0.;
          nonPrefiringProbaJet[fluct] = 0.;
        }
      }
      //Last case: if overlapping photons have a non prefiring rate smaller than the jet, don't consider the jet in the event weight, and do nothing.
    }
    for (const auto& muon : theMuons) {
      double pt = muon.pt();
      double phi = muon.eta();
      double eta = muon.eta();
      // Remove crappy tracker muons which would not have prefired the L1 trigger
      if (pt < 5 && !muon.isStandAloneMuon())
        continue;
      double prefiringprob_mu = getPrefiringRateMuon(eta, phi, pt, fluct);
      nonPrefiringProba[fluct] *= (1. - prefiringprob_mu);
      nonPrefiringProbaMuon[fluct] *= (1. - prefiringprob_mu);
    }
  }
  auto nonPrefiringProb = std::make_unique<double>(nonPrefiringProba[0]);
  auto nonPrefiringProbUp = std::make_unique<double>(nonPrefiringProba[1]);
  auto nonPrefiringProbDown = std::make_unique<double>(nonPrefiringProba[2]);
  iEvent.put(std::move(nonPrefiringProb), "nonPrefiringProb");
  iEvent.put(std::move(nonPrefiringProbUp), "nonPrefiringProbUp");
  iEvent.put(std::move(nonPrefiringProbDown), "nonPrefiringProbDown");

  auto nonPrefiringProbJet = std::make_unique<double>(nonPrefiringProbaJet[0]);
  auto nonPrefiringProbJetUp = std::make_unique<double>(nonPrefiringProbaJet[1]);
  auto nonPrefiringProbJetDown = std::make_unique<double>(nonPrefiringProbaJet[2]);
  iEvent.put(std::move(nonPrefiringProbJet), "nonPrefiringProbJet");
  iEvent.put(std::move(nonPrefiringProbJetUp), "nonPrefiringProbJetUp");
  iEvent.put(std::move(nonPrefiringProbJetDown), "nonPrefiringProbJetDown");

  auto nonPrefiringProbPhoton = std::make_unique<double>(nonPrefiringProbaPhoton[0]);
  auto nonPrefiringProbPhotonUp = std::make_unique<double>(nonPrefiringProbaPhoton[1]);
  auto nonPrefiringProbPhotonDown = std::make_unique<double>(nonPrefiringProbaPhoton[2]);
  iEvent.put(std::move(nonPrefiringProbPhoton), "nonPrefiringProbPhoton");
  iEvent.put(std::move(nonPrefiringProbPhotonUp), "nonPrefiringProbPhotonUp");
  iEvent.put(std::move(nonPrefiringProbPhotonDown), "nonPrefiringProbPhotonDown");

  auto nonPrefiringProbMuon = std::make_unique<double>(nonPrefiringProbaMuon[0]);
  auto nonPrefiringProbMuonUp = std::make_unique<double>(nonPrefiringProbaMuon[1]);
  auto nonPrefiringProbMuonDown = std::make_unique<double>(nonPrefiringProbaMuon[2]);
  iEvent.put(std::move(nonPrefiringProbMuon), "nonPrefiringProbMuon");
  iEvent.put(std::move(nonPrefiringProbMuonUp), "nonPrefiringProbMuonUp");
  iEvent.put(std::move(nonPrefiringProbMuonDown), "nonPrefiringProbMuonDown");
}

double L1PrefiringWeightProducer::getPrefiringRate(double eta,
                                                   double pt,
                                                   TH2F* h_prefmap,
                                                   fluctuations fluctuation) const {
  if (h_prefmap == nullptr && !skipwarnings_)
    edm::LogWarning("L1PrefireWeightProducer") << "Prefiring map not found, setting prefiring rate to 0 " << std::endl;
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
double L1PrefiringWeightProducer::getPrefiringRateMuon(double eta,
                                                       double phi,
                                                       double pt,
                                                       fluctuations fluctuation) const {
  double prefrate;
  double statuncty;
  if ((dataeraMuon_.find("2016") != std::string::npos) && (eta > 1.24 && eta < 1.6) &&
      (phi > 2.44346 && phi < 2.79253)) {
    if (parametrizationHotSpot_ == nullptr && !skipwarnings_)
      edm::LogWarning("L1PrefireWeightProducer") << "Prefiring parametrization not found, setting prefiring rate to 0 " << eta << " " << phi
                << std::endl;
    if (parametrizationHotSpot_ == nullptr)
      return 0.;
    prefrate = parametrizationHotSpot_->Eval(pt);
    statuncty = parametrizationHotSpot_->GetParError(2);
  } else if (std::abs(eta) < 0.2) {
    if (parametrization0p0To0p2_ == nullptr && !skipwarnings_)
      edm::LogWarning("L1PrefireWeightProducer") << "Prefiring parametrization not found, setting prefiring rate to 0 " << eta << " " << std::endl;
    if (parametrization0p0To0p2_ == nullptr)
      return 0.;
    prefrate = parametrization0p0To0p2_->Eval(pt);
    statuncty = parametrization0p0To0p2_->GetParError(2);
  } else if (std::abs(eta) < 0.3) {
    if (parametrization0p2To0p3_ == nullptr && !skipwarnings_)
      edm::LogWarning("L1PrefireWeightProducer") << "Prefiring parametrization not found, setting prefiring rate to 0 " << eta << " " << std::endl;
    if (parametrization0p2To0p3_ == nullptr)
      return 0.;

    prefrate = parametrization0p2To0p3_->Eval(pt);
    statuncty = parametrization0p2To0p3_->GetParError(2);
  } else if (std::abs(eta) < 0.55) {
    if (parametrization0p3To0p55_ == nullptr && !skipwarnings_)
      edm::LogWarning("L1PrefireWeightProducer") << "Prefiring parametrization not found, setting prefiring rate to 0 " << eta << " " << std::endl;
    if (parametrization0p3To0p55_ == nullptr)
      return 0.;

    prefrate = parametrization0p3To0p55_->Eval(pt);
    statuncty = parametrization0p3To0p55_->GetParError(2);
  } else if (std::abs(eta) < 0.83) {
    if (parametrization0p55To0p83_ == nullptr && !skipwarnings_)
      edm::LogWarning("L1PrefireWeightProducer") << "Prefiring parametrization not found, setting prefiring rate to 0 " << eta << " " << std::endl;
    if (parametrization0p55To0p83_ == nullptr)
      return 0.;

    prefrate = parametrization0p55To0p83_->Eval(pt);
    statuncty = parametrization0p55To0p83_->GetParError(2);
  } else if (std::abs(eta) < 1.24) {
    if (parametrization0p83To1p24_ == nullptr && !skipwarnings_)
      edm::LogWarning("L1PrefireWeightProducer") << "Prefiring parametrization not found, setting prefiring rate to 0 " << eta << " " << std::endl;
    if (parametrization0p83To1p24_ == nullptr)
      return 0.;

    prefrate = parametrization0p83To1p24_->Eval(pt);
    statuncty = parametrization0p83To1p24_->GetParError(2);
  } else if (std::abs(eta) < 1.4) {
    if (parametrization1p24To1p4_ == nullptr && !skipwarnings_)
      edm::LogWarning("L1PrefireWeightProducer") << "Prefiring parametrization not found, setting prefiring rate to 0 " << eta << " " << std::endl;
    if (parametrization1p24To1p4_ == nullptr)
      return 0.;

    prefrate = parametrization1p24To1p4_->Eval(pt);
    statuncty = parametrization1p24To1p4_->GetParError(2);
  } else if (std::abs(eta) < 1.6) {
    if (parametrization1p4To1p6_ == nullptr && !skipwarnings_)
      edm::LogWarning("L1PrefireWeightProducer") << "Prefiring parametrization not found, setting prefiring rate to 0 " << eta << " " << std::endl;
    if (parametrization1p4To1p6_ == nullptr)
      return 0.;

    prefrate = parametrization1p4To1p6_->Eval(pt);
    statuncty = parametrization1p4To1p6_->GetParError(2);
  } else if (std::abs(eta) < 1.8) {
    if (parametrization1p6To1p8_ == nullptr && !skipwarnings_)
      edm::LogWarning("L1PrefireWeightProducer") << "Prefiring parametrization not found, setting prefiring rate to 0 " << eta << " " << std::endl;
    if (parametrization1p6To1p8_ == nullptr)
      return 0.;

    prefrate = parametrization1p6To1p8_->Eval(pt);
    statuncty = parametrization1p6To1p8_->GetParError(2);
  } else if (std::abs(eta) < 2.1) {
    if (parametrization1p8To2p1_ == nullptr && !skipwarnings_)
      edm::LogWarning("L1PrefireWeightProducer") << "Prefiring parametrization not found, setting prefiring rate to 0 " << eta << " " << std::endl;
    if (parametrization1p8To2p1_ == nullptr)
      return 0.;

    prefrate = parametrization1p8To2p1_->Eval(pt);
    statuncty = parametrization1p8To2p1_->GetParError(2);
  } else if (std::abs(eta) < 2.25) {
    if (parametrization2p1To2p25_ == nullptr && !skipwarnings_)
      edm::LogWarning("L1PrefireWeightProducer") << "Prefiring parametrization not found, setting prefiring rate to 0 " << eta << " " << std::endl;
    if (parametrization2p1To2p25_ == nullptr)
      return 0.;

    prefrate = parametrization2p1To2p25_->Eval(pt);
    statuncty = parametrization2p1To2p25_->GetParError(2);
  } else if (std::abs(eta) < 2.4) {
    if (parametrization2p25To2p4_ == nullptr && !skipwarnings_)
      edm::LogWarning("L1PrefireWeightProducer") << "Prefiring parametrization not found, setting prefiring rate to 0 " << eta << " " << std::endl;
    if (parametrization2p25To2p4_ == nullptr)
      return 0.;

    prefrate = parametrization2p25To2p4_->Eval(pt);
    statuncty = parametrization2p25To2p4_->GetParError(2);
  } else
    return 0.;
  double systuncty = prefiringRateSystUnc_ * prefrate;

  if (fluctuation == up)
    prefrate = std::min(1., prefrate + sqrt(pow(statuncty, 2) + pow(systuncty, 2)));
  if (fluctuation == down)
    prefrate = std::max(0., prefrate - sqrt(pow(statuncty, 2) + pow(systuncty, 2)));
  return prefrate;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1PrefiringWeightProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("TheMuons", edm::InputTag("slimmedMuons"));
  desc.add<edm::InputTag>("ThePhotons", edm::InputTag("slimmedPhotons"));
  desc.add<edm::InputTag>("TheJets", edm::InputTag("slimmedJets"));
  desc.add<std::string>("L1Maps", "L1PrefiringMaps.root");
  desc.add<std::string>("L1MuonParametrizations", "L1MuonPrefiringParametriations.root");
  desc.add<std::string>("DataEraECAL", "2017BtoF");
  desc.add<std::string>("DataEraMuon", "2016");
  desc.add<bool>("UseJetEMPt", false);
  desc.add<double>("PrefiringRateSystematicUncty", 0.2);
  desc.add<double>("JetMaxMuonFraction", 0.5);
  desc.add<bool>("SkipWarnings", true);
  descriptions.add("l1PrefiringWeightProducer", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1PrefiringWeightProducer);
