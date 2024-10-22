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
#include "FWCore/Framework/interface/stream/EDProducer.h"

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
enum fluctuations { central = 0, up, down, upStat, downStat, upSyst, downSyst };

class L1PrefiringWeightProducer : public edm::stream::EDProducer<> {
public:
  explicit L1PrefiringWeightProducer(const edm::ParameterSet&);
  ~L1PrefiringWeightProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  double getPrefiringRateEcal(double eta, double pt, TH2F* h_prefmap, fluctuations fluctuation) const;
  double getPrefiringRateMuon(double eta, double phi, double pt, fluctuations fluctuation) const;

  const edm::EDGetTokenT<std::vector<pat::Photon> > photons_token_;
  const edm::EDGetTokenT<std::vector<pat::Jet> > jets_token_;
  const edm::EDGetTokenT<std::vector<pat::Muon> > muon_token_;

  const edm::EDPutTokenT<float> nonPrefiringProbToken_;
  const edm::EDPutTokenT<float> nonPrefiringProbUpToken_;
  const edm::EDPutTokenT<float> nonPrefiringProbDownToken_;

  const edm::EDPutTokenT<float> nonPrefiringProbECALToken_;
  const edm::EDPutTokenT<float> nonPrefiringProbECALUpToken_;
  const edm::EDPutTokenT<float> nonPrefiringProbECALDownToken_;

  const edm::EDPutTokenT<float> nonPrefiringProbMuonToken_;
  const edm::EDPutTokenT<float> nonPrefiringProbMuonUpToken_;
  const edm::EDPutTokenT<float> nonPrefiringProbMuonDownToken_;
  const edm::EDPutTokenT<float> nonPrefiringProbMuonUpSystToken_;
  const edm::EDPutTokenT<float> nonPrefiringProbMuonDownSystToken_;
  const edm::EDPutTokenT<float> nonPrefiringProbMuonUpStatToken_;
  const edm::EDPutTokenT<float> nonPrefiringProbMuonDownStatToken_;

  std::unique_ptr<TFile> file_prefiringmaps_;
  std::unique_ptr<TFile> file_prefiringparams_;

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

  TH2F* h_prefmap_photon_;
  TH2F* h_prefmap_jet_;
  const std::string dataeraEcal_;
  const std::string dataeraMuon_;
  const bool useEMpt_;
  const double prefiringRateSystUncEcal_;
  const double prefiringRateSystUncMuon_;
  const double jetMaxMuonFraction_;
  bool missingInputEcal_;
  bool missingInputMuon_;
};

L1PrefiringWeightProducer::L1PrefiringWeightProducer(const edm::ParameterSet& iConfig)
    : photons_token_(consumes<std::vector<pat::Photon> >(iConfig.getParameter<edm::InputTag>("ThePhotons"))),
      jets_token_(consumes<std::vector<pat::Jet> >(iConfig.getParameter<edm::InputTag>("TheJets"))),
      muon_token_(consumes<std::vector<pat::Muon> >(iConfig.getParameter<edm::InputTag>("TheMuons"))),
      nonPrefiringProbToken_(produces<float>("nonPrefiringProb")),
      nonPrefiringProbUpToken_(produces<float>("nonPrefiringProbUp")),
      nonPrefiringProbDownToken_(produces<float>("nonPrefiringProbDown")),
      nonPrefiringProbECALToken_(produces<float>("nonPrefiringProbECAL")),
      nonPrefiringProbECALUpToken_(produces<float>("nonPrefiringProbECALUp")),
      nonPrefiringProbECALDownToken_(produces<float>("nonPrefiringProbECALDown")),
      nonPrefiringProbMuonToken_(produces<float>("nonPrefiringProbMuon")),
      nonPrefiringProbMuonUpToken_(produces<float>("nonPrefiringProbMuonUp")),
      nonPrefiringProbMuonDownToken_(produces<float>("nonPrefiringProbMuonDown")),
      nonPrefiringProbMuonUpSystToken_(produces<float>("nonPrefiringProbMuonSystUp")),
      nonPrefiringProbMuonDownSystToken_(produces<float>("nonPrefiringProbMuonSystDown")),
      nonPrefiringProbMuonUpStatToken_(produces<float>("nonPrefiringProbMuonStatUp")),
      nonPrefiringProbMuonDownStatToken_(produces<float>("nonPrefiringProbMuonStatDown")),
      dataeraEcal_(iConfig.getParameter<std::string>("DataEraECAL")),
      dataeraMuon_(iConfig.getParameter<std::string>("DataEraMuon")),
      useEMpt_(iConfig.getParameter<bool>("UseJetEMPt")),
      prefiringRateSystUncEcal_(iConfig.getParameter<double>("PrefiringRateSystematicUnctyECAL")),
      prefiringRateSystUncMuon_(iConfig.getParameter<double>("PrefiringRateSystematicUnctyMuon")),
      jetMaxMuonFraction_(iConfig.getParameter<double>("JetMaxMuonFraction")) {
  missingInputEcal_ = false;
  missingInputMuon_ = false;

  std::string fname = iConfig.getParameter<std::string>("L1Maps");
  edm::FileInPath mapsfilepath("PhysicsTools/PatUtils/data/" + fname);
  file_prefiringmaps_ = std::make_unique<TFile>(mapsfilepath.fullPath().c_str(), "read");
  if (file_prefiringmaps_ == nullptr) {
    missingInputEcal_ = true;
    edm::LogError("L1PrefireWeightProducer")
        << "File with maps not found. All prefiring weights set to 0. " << std::endl;
  }
  TString mapphotonfullname = "L1prefiring_photonptvseta_" + dataeraEcal_;
  if (!file_prefiringmaps_->Get(mapphotonfullname)) {
    missingInputEcal_ = true;
    edm::LogError("L1PrefireWeightProducer")
        << "Photon map not found. All photons prefiring weights set to 0. " << std::endl;
  }
  h_prefmap_photon_ = file_prefiringmaps_->Get<TH2F>(mapphotonfullname);
  TString mapjetfullname =
      (useEMpt_) ? "L1prefiring_jetemptvseta_" + dataeraEcal_ : "L1prefiring_jetptvseta_" + dataeraEcal_;
  if (!file_prefiringmaps_->Get(mapjetfullname)) {
    missingInputEcal_ = true;
    edm::LogError("L1PrefireWeightProducer") << "Jet map not found. All jets prefiring weights set to 0. " << std::endl;
  }
  h_prefmap_jet_ = file_prefiringmaps_->Get<TH2F>(mapjetfullname);
  file_prefiringmaps_->Close();

  std::string fnameMuon = iConfig.getParameter<std::string>("L1MuonParametrizations");
  edm::FileInPath paramsfilepath("PhysicsTools/PatUtils/data/" + fnameMuon);
  file_prefiringparams_ = std::make_unique<TFile>(paramsfilepath.fullPath().c_str(), "read");
  if (file_prefiringparams_ == nullptr) {
    missingInputMuon_ = true;
    edm::LogError("L1PrefireWeightProducer")
        << "File with muon parametrizations not found. All prefiring weights set to 0." << std::endl;
  }
  TString paramName = "L1prefiring_muonparam_0.0To0.2_" + dataeraMuon_;
  parametrization0p0To0p2_ = file_prefiringparams_->Get<TF1>(paramName);
  paramName = "L1prefiring_muonparam_0.2To0.3_" + dataeraMuon_;
  parametrization0p2To0p3_ = file_prefiringparams_->Get<TF1>(paramName);
  paramName = "L1prefiring_muonparam_0.3To0.55_" + dataeraMuon_;
  parametrization0p3To0p55_ = file_prefiringparams_->Get<TF1>(paramName);
  paramName = "L1prefiring_muonparam_0.55To0.83_" + dataeraMuon_;
  parametrization0p55To0p83_ = file_prefiringparams_->Get<TF1>(paramName);
  paramName = "L1prefiring_muonparam_0.83To1.24_" + dataeraMuon_;
  parametrization0p83To1p24_ = file_prefiringparams_->Get<TF1>(paramName);
  paramName = "L1prefiring_muonparam_1.24To1.4_" + dataeraMuon_;
  parametrization1p24To1p4_ = file_prefiringparams_->Get<TF1>(paramName);
  paramName = "L1prefiring_muonparam_1.4To1.6_" + dataeraMuon_;
  parametrization1p4To1p6_ = file_prefiringparams_->Get<TF1>(paramName);
  paramName = "L1prefiring_muonparam_1.6To1.8_" + dataeraMuon_;
  parametrization1p6To1p8_ = file_prefiringparams_->Get<TF1>(paramName);
  paramName = "L1prefiring_muonparam_1.8To2.1_" + dataeraMuon_;
  parametrization1p8To2p1_ = file_prefiringparams_->Get<TF1>(paramName);
  paramName = "L1prefiring_muonparam_2.1To2.25_" + dataeraMuon_;
  parametrization2p1To2p25_ = file_prefiringparams_->Get<TF1>(paramName);
  paramName = "L1prefiring_muonparam_2.25To2.4_" + dataeraMuon_;
  parametrization2p25To2p4_ = file_prefiringparams_->Get<TF1>(paramName);

  if (parametrization0p0To0p2_ == nullptr || parametrization0p2To0p3_ == nullptr ||
      parametrization0p3To0p55_ == nullptr || parametrization0p55To0p83_ == nullptr ||
      parametrization0p83To1p24_ == nullptr || parametrization1p24To1p4_ == nullptr ||
      parametrization1p4To1p6_ == nullptr || parametrization1p6To1p8_ == nullptr ||
      parametrization1p8To2p1_ == nullptr || parametrization2p1To2p25_ == nullptr ||
      parametrization2p25To2p4_ == nullptr) {
    missingInputMuon_ = true;
    edm::LogError("L1PrefireWeightProducer")
        << "Muon parametrization not found for at least one bin. All prefiring weights set to 0." << std::endl;
  }

  paramName = "L1prefiring_muonparam_HotSpot_" + dataeraMuon_;
  parametrizationHotSpot_ = file_prefiringparams_->Get<TF1>(paramName);
  file_prefiringparams_->Close();
  if ((dataeraMuon_.find("2016") != std::string::npos) && parametrizationHotSpot_ == nullptr) {
    missingInputMuon_ = true;
    edm::LogError("L1PrefireWeightProducer")
        << "Year is 2016 and no Muon parametrization is found for hot spot. All prefiring weights set to 0."
        << std::endl;
  }
}

L1PrefiringWeightProducer::~L1PrefiringWeightProducer() {}

void L1PrefiringWeightProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  //Photons
  const std::vector<pat::Photon>& thePhotons = iEvent.get(photons_token_);

  //Jets
  const std::vector<pat::Jet>& theJets = iEvent.get(jets_token_);

  //Muons
  const std::vector<pat::Muon>& theMuons = iEvent.get(muon_token_);

  //Probability for the event NOT to prefire, computed with the prefiring maps per object.
  //Up and down values correspond to the resulting value when shifting up/down all prefiring rates in prefiring maps.
  double nonPrefiringProba[3] = {1., 1., 1.};      //0: central, 1: up, 2: down
  double nonPrefiringProbaECAL[3] = {1., 1., 1.};  //0: central, 1: up, 2: down
  double nonPrefiringProbaMuon[7] = {
      1., 1., 1., 1., 1., 1., 1.};  //0: central, 1: up, 2: down, 3: up stat, 4: down stat, 5: up syst, 6: down syst

  for (const auto fluct : {fluctuations::central, fluctuations::up, fluctuations::down}) {
    if (!missingInputEcal_) {
      for (const auto& photon : thePhotons) {
        double pt_gam = photon.pt();
        double eta_gam = photon.eta();
        if (pt_gam < 20.)
          continue;
        if (fabs(eta_gam) < 2.)
          continue;
        if (fabs(eta_gam) > 3.)
          continue;
        double prefiringprob_gam = getPrefiringRateEcal(eta_gam, pt_gam, h_prefmap_photon_, fluct);
        nonPrefiringProbaECAL[fluct] *= (1. - prefiringprob_gam);
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
        bool foundOverlappingPhotons = false;
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
          double dR2 = reco::deltaR2(eta_jet, phi_jet, eta_gam, phi_gam);
          if (dR2 > 0.16)
            continue;
          double prefiringprob_gam = getPrefiringRateEcal(eta_gam, pt_gam, h_prefmap_photon_, fluct);
          nonprefiringprobfromoverlappingphotons *= (1. - prefiringprob_gam);
          foundOverlappingPhotons = true;
        }
        //useEMpt =true if one wants to use maps parametrized vs Jet EM pt instead of pt.
        if (useEMpt_)
          pt_jet *= (jet.neutralEmEnergyFraction() + jet.chargedEmEnergyFraction());
        double nonprefiringprobfromoverlappingjet = 1. - getPrefiringRateEcal(eta_jet, pt_jet, h_prefmap_jet_, fluct);

        if (!foundOverlappingPhotons) {
          nonPrefiringProbaECAL[fluct] *= nonprefiringprobfromoverlappingjet;
        }
        //If overlapping photons have a non prefiring rate larger than the jet, then replace these weights by the jet one
        else if (nonprefiringprobfromoverlappingphotons > nonprefiringprobfromoverlappingjet) {
          if (nonprefiringprobfromoverlappingphotons > 0.) {
            nonPrefiringProbaECAL[fluct] *= nonprefiringprobfromoverlappingjet / nonprefiringprobfromoverlappingphotons;
          } else {
            nonPrefiringProbaECAL[fluct] = 0.;
          }
        }
        //Last case: if overlapping photons have a non prefiring rate smaller than the jet, don't consider the jet in the event weight, and do nothing.
      }
    }
    //Now calculate prefiring weights for muons
    if (!missingInputMuon_) {
      for (const auto& muon : theMuons) {
        double pt = muon.pt();
        double phi = muon.phi();
        double eta = muon.eta();
        // Remove crappy tracker muons which would not have prefired the L1 trigger
        if (pt < 5 || !muon.isLooseMuon())
          continue;
        double prefiringprob_mu = getPrefiringRateMuon(eta, phi, pt, fluct);
        nonPrefiringProbaMuon[fluct] *= (1. - prefiringprob_mu);
      }
    }
  }
  // Calculate combined weight as product of the weight for individual objects
  for (const auto fluct : {fluctuations::central, fluctuations::up, fluctuations::down}) {
    nonPrefiringProba[fluct] = nonPrefiringProbaECAL[fluct] * nonPrefiringProbaMuon[fluct];
  }
  // Calculate statistical and systematic uncertainty separately in the muon case
  for (const auto fluct :
       {fluctuations::upSyst, fluctuations::downSyst, fluctuations::upStat, fluctuations::downStat}) {
    if (!missingInputMuon_) {
      for (const auto& muon : theMuons) {
        double pt = muon.pt();
        double phi = muon.phi();
        double eta = muon.eta();
        // Remove crappy tracker muons which would not have prefired the L1 trigger
        if (pt < 5 || !muon.isLooseMuon())
          continue;
        double prefiringprob_mu = getPrefiringRateMuon(eta, phi, pt, fluct);
        nonPrefiringProbaMuon[fluct] *= (1. - prefiringprob_mu);
      }
    }
  }
  //Move global prefire weights, as well as those for muons, photons, and jets, to the event
  iEvent.emplace(nonPrefiringProbToken_, nonPrefiringProba[0]);
  iEvent.emplace(nonPrefiringProbUpToken_, nonPrefiringProba[1]);
  iEvent.emplace(nonPrefiringProbDownToken_, nonPrefiringProba[2]);

  iEvent.emplace(nonPrefiringProbECALToken_, nonPrefiringProbaECAL[0]);
  iEvent.emplace(nonPrefiringProbECALUpToken_, nonPrefiringProbaECAL[1]);
  iEvent.emplace(nonPrefiringProbECALDownToken_, nonPrefiringProbaECAL[2]);

  iEvent.emplace(nonPrefiringProbMuonToken_, nonPrefiringProbaMuon[0]);
  iEvent.emplace(nonPrefiringProbMuonUpToken_, nonPrefiringProbaMuon[1]);
  iEvent.emplace(nonPrefiringProbMuonDownToken_, nonPrefiringProbaMuon[2]);
  iEvent.emplace(nonPrefiringProbMuonUpStatToken_, nonPrefiringProbaMuon[3]);
  iEvent.emplace(nonPrefiringProbMuonDownStatToken_, nonPrefiringProbaMuon[4]);
  iEvent.emplace(nonPrefiringProbMuonUpSystToken_, nonPrefiringProbaMuon[5]);
  iEvent.emplace(nonPrefiringProbMuonDownSystToken_, nonPrefiringProbaMuon[6]);
}

double L1PrefiringWeightProducer::getPrefiringRateEcal(double eta,
                                                       double pt,
                                                       TH2F* h_prefmap,
                                                       fluctuations fluctuation) const {
  //Check pt is not above map overflow
  int nbinsy = h_prefmap->GetNbinsY();
  double maxy = h_prefmap->GetYaxis()->GetBinLowEdge(nbinsy + 1);
  if (pt >= maxy)
    pt = maxy - 0.01;
  int thebin = h_prefmap->FindBin(eta, pt);

  double prefrate = h_prefmap->GetBinContent(thebin);

  double statuncty = h_prefmap->GetBinError(thebin);
  double systuncty = prefiringRateSystUncEcal_ * prefrate;

  if (fluctuation == up)
    prefrate = std::min(1., prefrate + sqrt(pow(statuncty, 2) + pow(systuncty, 2)));
  else if (fluctuation == down)
    prefrate = std::max(0., prefrate - sqrt(pow(statuncty, 2) + pow(systuncty, 2)));
  if (prefrate > 1.) {
    edm::LogWarning("L1PrefireWeightProducer") << "Found a prefiring probability > 1. Setting to 1." << std::endl;
    return 1.;
  }
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
    prefrate = parametrizationHotSpot_->Eval(pt);
    statuncty = parametrizationHotSpot_->GetParError(2);
  } else if (std::abs(eta) < 0.2) {
    prefrate = parametrization0p0To0p2_->Eval(pt);
    statuncty = parametrization0p0To0p2_->GetParError(2);
  } else if (std::abs(eta) < 0.3) {
    prefrate = parametrization0p2To0p3_->Eval(pt);
    statuncty = parametrization0p2To0p3_->GetParError(2);
  } else if (std::abs(eta) < 0.55) {
    prefrate = parametrization0p3To0p55_->Eval(pt);
    statuncty = parametrization0p3To0p55_->GetParError(2);
  } else if (std::abs(eta) < 0.83) {
    prefrate = parametrization0p55To0p83_->Eval(pt);
    statuncty = parametrization0p55To0p83_->GetParError(2);
  } else if (std::abs(eta) < 1.24) {
    prefrate = parametrization0p83To1p24_->Eval(pt);
    statuncty = parametrization0p83To1p24_->GetParError(2);
  } else if (std::abs(eta) < 1.4) {
    prefrate = parametrization1p24To1p4_->Eval(pt);
    statuncty = parametrization1p24To1p4_->GetParError(2);
  } else if (std::abs(eta) < 1.6) {
    prefrate = parametrization1p4To1p6_->Eval(pt);
    statuncty = parametrization1p4To1p6_->GetParError(2);
  } else if (std::abs(eta) < 1.8) {
    prefrate = parametrization1p6To1p8_->Eval(pt);
    statuncty = parametrization1p6To1p8_->GetParError(2);
  } else if (std::abs(eta) < 2.1) {
    prefrate = parametrization1p8To2p1_->Eval(pt);
    statuncty = parametrization1p8To2p1_->GetParError(2);
  } else if (std::abs(eta) < 2.25) {
    prefrate = parametrization2p1To2p25_->Eval(pt);
    statuncty = parametrization2p1To2p25_->GetParError(2);
  } else if (std::abs(eta) < 2.4) {
    prefrate = parametrization2p25To2p4_->Eval(pt);
    statuncty = parametrization2p25To2p4_->GetParError(2);
  } else {
    LogDebug("L1PrefireWeightProducer") << "Muon outside of |eta| <= 2.4. Prefiring weight set to 0." << std::endl;
    return 0.;
  }
  double systuncty = prefiringRateSystUncMuon_ * prefrate;

  if (fluctuation == up)
    prefrate = std::min(1., prefrate + sqrt(pow(statuncty, 2) + pow(systuncty, 2)));
  else if (fluctuation == down)
    prefrate = std::max(0., prefrate - sqrt(pow(statuncty, 2) + pow(systuncty, 2)));
  else if (fluctuation == upSyst)
    prefrate = std::min(1., prefrate + systuncty);
  else if (fluctuation == downSyst)
    prefrate = std::max(0., prefrate - systuncty);
  else if (fluctuation == upStat)
    prefrate = std::min(1., prefrate + statuncty);
  else if (fluctuation == downStat)
    prefrate = std::max(0., prefrate - statuncty);

  if (prefrate > 1.) {
    edm::LogWarning("L1PrefireWeightProducer") << "Found a prefiring probability > 1. Setting to 1." << std::endl;
    return 1.;
  }
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
  desc.add<double>("PrefiringRateSystematicUnctyECAL", 0.2);
  desc.add<double>("PrefiringRateSystematicUnctyMuon", 0.2);
  desc.add<double>("JetMaxMuonFraction", 0.5);
  descriptions.add("l1PrefiringWeightProducer", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1PrefiringWeightProducer);
