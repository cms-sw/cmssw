// -*- C++ -*-
//
// Package:    L1Trigger/L1CaloTrigger
// Class:      Phase2L1CaloJetEmulator
//
/**\class Phase2L1CaloJetEmulator Phase2L1CaloJetEmulator.cc L1Trigger/L1CaloTrigger/plugins/Phase2L1CaloJetEmulator.cc

 Description: Producing GCT calo jets using GCT barrel, HGCal and HF towers, based on firmware logic.

 Implementation:
     Depends on producers for CaloTowerCollection, HGCalTowerBxCollection and HcalTrigPrimDigiCollection.
*/
//
// Original Author:  Pallabi Das
//         Created:  Tue, 11 Apr 2023 11:27:33 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/L1TCalorimeterPhase2/interface/CaloCrystalCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/CaloTower.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/CaloPFCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/Phase2L1CaloJet.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "L1Trigger/L1CaloTrigger/interface/Phase2L1CaloJetEmulator.h"
#include <ap_int.h>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "TF1.h"

//
// class declaration
//

class Phase2L1CaloJetEmulator : public edm::stream::EDProducer<> {
public:
  explicit Phase2L1CaloJetEmulator(const edm::ParameterSet&);
  ~Phase2L1CaloJetEmulator() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  float get_jet_pt_calibration(const float& jet_pt, const float& jet_eta) const;
  float get_tau_pt_calibration(const float& tau_pt, const float& tau_eta) const;

  // ----------member data ---------------------------
  edm::EDGetTokenT<l1tp2::CaloTowerCollection> caloTowerToken_;
  edm::EDGetTokenT<l1t::HGCalTowerBxCollection> hgcalTowerToken_;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> hfToken_;
  edm::ESGetToken<CaloTPGTranscoder, CaloTPGRecord> decoderTag_;
  std::vector<edm::ParameterSet> nHits_to_nvtx_params;
  std::vector<edm::ParameterSet> nvtx_to_PU_sub_params;
  std::map<std::string, TF1> nHits_to_nvtx_funcs;
  std::map<std::string, TF1> hgcalEM_nvtx_to_PU_sub_funcs;
  std::map<std::string, TF1> hgcalHad_nvtx_to_PU_sub_funcs;
  std::map<std::string, TF1> hf_nvtx_to_PU_sub_funcs;
  std::map<std::string, std::map<std::string, TF1>> all_nvtx_to_PU_sub_funcs;

  // For fetching jet pt calibrations
  std::vector<double> jetPtBins;
  std::vector<double> absEtaBinsBarrel;
  std::vector<double> jetCalibrationsBarrel;
  std::vector<double> absEtaBinsHGCal;
  std::vector<double> jetCalibrationsHGCal;
  std::vector<double> absEtaBinsHF;
  std::vector<double> jetCalibrationsHF;

  // For fetching tau pt calibrations
  std::vector<double> tauPtBins;
  std::vector<double> tauAbsEtaBinsBarrel;
  std::vector<double> tauCalibrationsBarrel;
  std::vector<double> tauAbsEtaBinsHGCal;
  std::vector<double> tauCalibrationsHGCal;

  // For storing jet calibrations
  std::vector<std::vector<double>> calibrationsBarrel;
  std::vector<std::vector<double>> calibrationsHGCal;
  std::vector<std::vector<double>> calibrationsHF;

  // For storing tau calibrations
  std::vector<std::vector<double>> tauPtCalibrationsBarrel;
  std::vector<std::vector<double>> tauPtCalibrationsHGCal;
};

//
// constructors and destructor
//
Phase2L1CaloJetEmulator::Phase2L1CaloJetEmulator(const edm::ParameterSet& iConfig)
    : caloTowerToken_(consumes<l1tp2::CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("gctFullTowers"))),
      hgcalTowerToken_(consumes<l1t::HGCalTowerBxCollection>(iConfig.getParameter<edm::InputTag>("hgcalTowers"))),
      hfToken_(consumes<HcalTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("hcalDigis"))),
      decoderTag_(esConsumes<CaloTPGTranscoder, CaloTPGRecord>(edm::ESInputTag("", ""))),
      nHits_to_nvtx_params(iConfig.getParameter<std::vector<edm::ParameterSet>>("nHits_to_nvtx_params")),
      nvtx_to_PU_sub_params(iConfig.getParameter<std::vector<edm::ParameterSet>>("nvtx_to_PU_sub_params")),
      jetPtBins(iConfig.getParameter<std::vector<double>>("jetPtBins")),
      absEtaBinsBarrel(iConfig.getParameter<std::vector<double>>("absEtaBinsBarrel")),
      jetCalibrationsBarrel(iConfig.getParameter<std::vector<double>>("jetCalibrationsBarrel")),
      absEtaBinsHGCal(iConfig.getParameter<std::vector<double>>("absEtaBinsHGCal")),
      jetCalibrationsHGCal(iConfig.getParameter<std::vector<double>>("jetCalibrationsHGCal")),
      absEtaBinsHF(iConfig.getParameter<std::vector<double>>("absEtaBinsHF")),
      jetCalibrationsHF(iConfig.getParameter<std::vector<double>>("jetCalibrationsHF")),
      tauPtBins(iConfig.getParameter<std::vector<double>>("tauPtBins")),
      tauAbsEtaBinsBarrel(iConfig.getParameter<std::vector<double>>("tauAbsEtaBinsBarrel")),
      tauCalibrationsBarrel(iConfig.getParameter<std::vector<double>>("tauCalibrationsBarrel")),
      tauAbsEtaBinsHGCal(iConfig.getParameter<std::vector<double>>("tauAbsEtaBinsHGCal")),
      tauCalibrationsHGCal(iConfig.getParameter<std::vector<double>>("tauCalibrationsHGCal")) {
  for (uint i = 0; i < nHits_to_nvtx_params.size(); i++) {
    edm::ParameterSet* pset = &nHits_to_nvtx_params.at(i);
    std::string calo = pset->getParameter<std::string>("fit");
    nHits_to_nvtx_funcs[calo.c_str()] = TF1(calo.c_str(), "[0] + [1] * x");
    nHits_to_nvtx_funcs[calo.c_str()].SetParameter(0, pset->getParameter<std::vector<double>>("nHits_params").at(0));
    nHits_to_nvtx_funcs[calo.c_str()].SetParameter(1, pset->getParameter<std::vector<double>>("nHits_params").at(1));
  }
  all_nvtx_to_PU_sub_funcs["hgcalEM"] = hgcalEM_nvtx_to_PU_sub_funcs;
  all_nvtx_to_PU_sub_funcs["hgcalHad"] = hgcalHad_nvtx_to_PU_sub_funcs;
  all_nvtx_to_PU_sub_funcs["hf"] = hf_nvtx_to_PU_sub_funcs;
  for (uint i = 0; i < nvtx_to_PU_sub_params.size(); i++) {
    edm::ParameterSet* pset = &nvtx_to_PU_sub_params.at(i);
    std::string calo = pset->getParameter<std::string>("calo");
    std::string iEta = pset->getParameter<std::string>("iEta");
    double p1 = pset->getParameter<std::vector<double>>("nvtx_params").at(0);
    double p2 = pset->getParameter<std::vector<double>>("nvtx_params").at(1);

    all_nvtx_to_PU_sub_funcs[calo.c_str()][iEta.c_str()] = TF1(calo.c_str(), "[0] + [1] * x");
    all_nvtx_to_PU_sub_funcs[calo.c_str()][iEta.c_str()].SetParameter(0, p1);
    all_nvtx_to_PU_sub_funcs[calo.c_str()][iEta.c_str()].SetParameter(1, p2);
  }

  // Fill the jet pt calibration 2D vector
  // Dimension 1 is AbsEta bin
  // Dimension 2 is jet pT bin which is filled with the actual callibration value
  // size()-1 b/c the inputs have lower and upper bounds
  // Do Barrel, then HGCal, then HF
  int index = 0;
  for (unsigned int abs_eta = 0; abs_eta < absEtaBinsBarrel.size() - 1; abs_eta++) {
    std::vector<double> pt_bin_calibs;
    for (unsigned int pt = 0; pt < jetPtBins.size() - 1; pt++) {
      pt_bin_calibs.push_back(jetCalibrationsBarrel.at(index));
      index++;
    }
    calibrationsBarrel.push_back(pt_bin_calibs);
  }

  index = 0;
  for (unsigned int abs_eta = 0; abs_eta < absEtaBinsHGCal.size() - 1; abs_eta++) {
    std::vector<double> pt_bin_calibs;
    for (unsigned int pt = 0; pt < jetPtBins.size() - 1; pt++) {
      pt_bin_calibs.push_back(jetCalibrationsHGCal.at(index));
      index++;
    }
    calibrationsHGCal.push_back(pt_bin_calibs);
  }

  index = 0;
  for (unsigned int abs_eta = 0; abs_eta < absEtaBinsHF.size() - 1; abs_eta++) {
    std::vector<double> pt_bin_calibs;
    for (unsigned int pt = 0; pt < jetPtBins.size() - 1; pt++) {
      pt_bin_calibs.push_back(jetCalibrationsHF.at(index));
      index++;
    }
    calibrationsHF.push_back(pt_bin_calibs);
  }

  // Fill the tau pt calibration 2D vector
  // Dimension 1 is AbsEta bin
  // Dimension 2 is tau pT bin which is filled with the actual calibration value
  // Do Barrel, then HGCal
  //
  // Note to future developers: be very concious of the order in which the calibrations are printed
  // out in tool which makse the cfg files.  You need to match that exactly when loading them and
  // using the calibrations below.
  index = 0;
  for (unsigned int abs_eta = 0; abs_eta < tauAbsEtaBinsBarrel.size() - 1; abs_eta++) {
    std::vector<double> pt_bin_calibs;
    for (unsigned int pt = 0; pt < tauPtBins.size() - 1; pt++) {
      pt_bin_calibs.push_back(tauCalibrationsBarrel.at(index));
      index++;
    }
    tauPtCalibrationsBarrel.push_back(pt_bin_calibs);
  }

  index = 0;
  for (unsigned int abs_eta = 0; abs_eta < tauAbsEtaBinsHGCal.size() - 1; abs_eta++) {
    std::vector<double> pt_bin_calibs;
    for (unsigned int pt = 0; pt < tauPtBins.size() - 1; pt++) {
      pt_bin_calibs.push_back(tauCalibrationsHGCal.at(index));
      index++;
    }
    tauPtCalibrationsHGCal.push_back(pt_bin_calibs);
  }

  produces<l1tp2::Phase2L1CaloJetCollection>("GCTJet");
}

Phase2L1CaloJetEmulator::~Phase2L1CaloJetEmulator() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void Phase2L1CaloJetEmulator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  std::unique_ptr<l1tp2::Phase2L1CaloJetCollection> jetCands(make_unique<l1tp2::Phase2L1CaloJetCollection>());

  // Assign ETs to each eta-half of the barrel region (17x72 --> 18x72 to be able to make 3x3 super towers)
  edm::Handle<std::vector<l1tp2::CaloTower>> caloTowerCollection;
  if (!iEvent.getByToken(caloTowerToken_, caloTowerCollection))
    edm::LogError("Phase2L1CaloJetEmulator") << "Failed to get towers from caloTowerCollection!";

  iEvent.getByToken(caloTowerToken_, caloTowerCollection);
  float GCTintTowers[nBarrelEta][nBarrelPhi];
  float realEta[nBarrelEta][nBarrelPhi];
  float realPhi[nBarrelEta][nBarrelPhi];
  for (const l1tp2::CaloTower& i : *caloTowerCollection) {
    int ieta = i.towerIEta();
    int iphi = i.towerIPhi();
    if (i.ecalTowerEt() > 1.)
      GCTintTowers[ieta][iphi] = i.ecalTowerEt();  // suppress <= 1 GeV towers
    else
      GCTintTowers[ieta][iphi] = 0;
    realEta[ieta][iphi] = i.towerEta();
    realPhi[ieta][iphi] = i.towerPhi();
  }

  float temporary[nBarrelEta / 2][nBarrelPhi];

  // HGCal and HF info used for nvtx estimation
  edm::Handle<l1t::HGCalTowerBxCollection> hgcalTowerCollection;
  if (!iEvent.getByToken(hgcalTowerToken_, hgcalTowerCollection))
    edm::LogError("Phase2L1CaloJetEmulator") << "Failed to get towers from hgcalTowerCollection!";
  l1t::HGCalTowerBxCollection hgcalTowerColl;
  iEvent.getByToken(hgcalTowerToken_, hgcalTowerCollection);
  hgcalTowerColl = (*hgcalTowerCollection.product());

  edm::Handle<HcalTrigPrimDigiCollection> hfHandle;
  if (!iEvent.getByToken(hfToken_, hfHandle))
    edm::LogError("Phase2L1CaloJetEmulator") << "Failed to get HcalTrigPrimDigi for HF!";
  iEvent.getByToken(hfToken_, hfHandle);

  int i_hgcalEM_hits_leq_threshold = 0;
  int i_hgcalHad_hits_leq_threshold = 0;
  int i_hf_hits_leq_threshold = 0;
  for (auto it = hgcalTowerColl.begin(0); it != hgcalTowerColl.end(0); it++) {
    if (it->etEm() <= 1.75 && it->etEm() >= 1.25) {
      i_hgcalEM_hits_leq_threshold++;
    }
    if (it->etHad() <= 1.25 && it->etHad() >= 0.75) {
      i_hgcalHad_hits_leq_threshold++;
    }
  }
  const auto& decoder = iSetup.getData(decoderTag_);
  for (const auto& hit : *hfHandle.product()) {
    double et = decoder.hcaletValue(hit.id(), hit.t0());
    if (abs(hit.id().ieta()) < l1t::CaloTools::kHFBegin)
      continue;
    if (abs(hit.id().ieta()) > l1t::CaloTools::kHFEnd)
      continue;
    if (et <= 15.0 && et >= 10.0)
      i_hf_hits_leq_threshold++;
  }

  double hgcalEM_nvtx = nHits_to_nvtx_funcs["hgcalEM"].Eval(i_hgcalEM_hits_leq_threshold);
  if (hgcalEM_nvtx < 0)
    hgcalEM_nvtx = 0;
  double hgcalHad_nvtx = nHits_to_nvtx_funcs["hgcalHad"].Eval(i_hgcalHad_hits_leq_threshold);
  if (hgcalHad_nvtx < 0)
    hgcalHad_nvtx = 0;
  double hf_nvtx = nHits_to_nvtx_funcs["hf"].Eval(i_hf_hits_leq_threshold);
  if (hf_nvtx < 0)
    hf_nvtx = 0;
  double EstimatedNvtx = (hgcalEM_nvtx + hgcalHad_nvtx + hf_nvtx) / 3.;

  // Assign ETs to each eta-half of the endcap region (18x72)
  float hgcalTowers[nHgcalEta][nHgcalPhi];
  float hgcalEta[nHgcalEta][nHgcalPhi];
  float hgcalPhi[nHgcalEta][nHgcalPhi];

  for (int iphi = 0; iphi < nHgcalPhi; iphi++) {
    for (int ieta = 0; ieta < nHgcalEta; ieta++) {
      hgcalTowers[ieta][iphi] = 0;
      if (ieta < nHgcalEta / 2)
        hgcalEta[ieta][iphi] = -3.045 + ieta * 0.087 + 0.0435;
      else
        hgcalEta[ieta][iphi] = 1.479 + (ieta - nHgcalEta / 2) * 0.087 + 0.0435;
      hgcalPhi[ieta][iphi] = -M_PI + (iphi * 2 * M_PI / nHgcalPhi) + (M_PI / nHgcalPhi);
    }
  }

  for (auto it = hgcalTowerColl.begin(0); it != hgcalTowerColl.end(0); it++) {
    float eta = it->eta();
    int ieta;
    if (eta < 0)
      ieta = 19 - it->id().iEta();
    else
      ieta = 20 + it->id().iEta();
    if (eta > 1.479)
      ieta = ieta - 4;
    int iphi = it->id().iPhi();

    float hgcal_etEm = it->etEm();
    float hgcal_etHad = it->etHad();
    std::string etaKey = "";
    if (abs(eta) <= 1.8)
      etaKey = "er1p4to1p8";
    else if (abs(eta) <= 2.1 && abs(eta) > 1.8)
      etaKey = "er1p8to2p1";
    else if (abs(eta) <= 2.4 && abs(eta) > 2.1)
      etaKey = "er2p1to2p4";
    else if (abs(eta) <= 2.7 && abs(eta) > 2.4)
      etaKey = "er2p4to2p7";
    else if (abs(eta) <= 3.1 && abs(eta) > 2.7)
      etaKey = "er2p7to3p1";
    if (!etaKey.empty()) {
      hgcal_etEm = it->etEm() - all_nvtx_to_PU_sub_funcs["hgcalEM"][etaKey].Eval(EstimatedNvtx);
      hgcal_etHad = it->etHad() - all_nvtx_to_PU_sub_funcs["hgcalHad"][etaKey].Eval(EstimatedNvtx);
    }

    if (hgcal_etEm < 0)
      hgcal_etEm = 0;
    if (hgcal_etHad < 0)
      hgcal_etHad = 0;
    if ((it->etEm() + it->etHad() > 1.) && abs(eta) > 1.479)
      hgcalTowers[ieta][iphi] = hgcal_etEm + hgcal_etHad;  // suppress <= 1 GeV towers
  }

  float temporary_hgcal[nHgcalEta / 2][nHgcalPhi];

  // Assign ETs to each eta-half of the forward region (12x72)
  float hfTowers[nHfEta][nHfPhi];
  float hfEta[nHfEta][nHfPhi];
  float hfPhi[nHfEta][nHfPhi];
  for (int iphi = 0; iphi < nHfPhi; iphi++) {
    for (int ieta = 0; ieta < nHfEta; ieta++) {
      hfTowers[ieta][iphi] = 0;
      int temp;
      if (ieta < nHfEta / 2)
        temp = ieta - l1t::CaloTools::kHFEnd;
      else
        temp = ieta - nHfEta / 2 + l1t::CaloTools::kHFBegin + 1;
      hfEta[ieta][iphi] = l1t::CaloTools::towerEta(temp);
      hfPhi[ieta][iphi] = -M_PI + (iphi * 2 * M_PI / nHfPhi) + (M_PI / nHfPhi);
    }
  }

  for (const auto& hit : *hfHandle.product()) {
    double et = decoder.hcaletValue(hit.id(), hit.t0());
    int ieta = 0;
    if (abs(hit.id().ieta()) < l1t::CaloTools::kHFBegin)
      continue;
    if (abs(hit.id().ieta()) > l1t::CaloTools::kHFEnd)
      continue;
    if (hit.id().ieta() <= -(l1t::CaloTools::kHFBegin + 1)) {
      ieta = hit.id().ieta() + l1t::CaloTools::kHFEnd;
    } else if (hit.id().ieta() >= (l1t::CaloTools::kHFBegin + 1)) {
      ieta = nHfEta / 2 + (hit.id().ieta() - (l1t::CaloTools::kHFBegin + 1));
    }
    int iphi = 0;
    if (hit.id().iphi() <= nHfPhi / 2)
      iphi = hit.id().iphi() + (nHfPhi / 2 - 1);
    else if (hit.id().iphi() > nHfPhi / 2)
      iphi = hit.id().iphi() - (nHfPhi / 2 + 1);
    if (abs(hit.id().ieta()) <= 33 && abs(hit.id().ieta()) >= 29)
      et = et - all_nvtx_to_PU_sub_funcs["hf"]["er29to33"].Eval(EstimatedNvtx);
    if (abs(hit.id().ieta()) <= 37 && abs(hit.id().ieta()) >= 34)
      et = et - all_nvtx_to_PU_sub_funcs["hf"]["er34to37"].Eval(EstimatedNvtx);
    if (abs(hit.id().ieta()) <= 41 && abs(hit.id().ieta()) >= 38)
      et = et - all_nvtx_to_PU_sub_funcs["hf"]["er38to41"].Eval(EstimatedNvtx);
    if (et < 0)
      et = 0;
    if (et > 1.)
      hfTowers[ieta][iphi] = et;  // suppress <= 1 GeV towers
  }

  float temporary_hf[nHfEta / 2][nHfPhi];

  // Begin creating jets
  // First create 3x3 super towers: 6x24 in barrel, endcap; 4x24 in forward
  // Then create up to 10 jets in each eta half of barrel, endcap, forward regions

  vector<l1tp2::Phase2L1CaloJet> halfBarrelJets, halfHgcalJets, halfHfJets;
  halfBarrelJets.clear();
  halfHgcalJets.clear();
  halfHfJets.clear();
  vector<l1tp2::Phase2L1CaloJet> allJets;
  allJets.clear();

  for (int k = 0; k < 2; k++) {
    halfBarrelJets.clear();
    halfHgcalJets.clear();
    halfHfJets.clear();
    gctobj::jetInfo jet[3 * nJets];

    // BARREL
    for (int iphi = 0; iphi < nBarrelPhi; iphi++) {
      for (int ieta = 0; ieta < nBarrelEta / 2; ieta++) {
        if (k == 0)
          temporary[ieta][iphi] = GCTintTowers[ieta][iphi];
        else
          temporary[ieta][iphi] = GCTintTowers[nBarrelEta / 2 + ieta][iphi];
      }
    }

    gctobj::GCTsupertower_t tempST[nSTEta][nSTPhi];
    gctobj::makeST(temporary, tempST);
    float TTseedThresholdBarrel = 5.;

    for (int i = 0; i < nJets; i++) {
      jet[i] = gctobj::getRegion(tempST, TTseedThresholdBarrel);
      l1tp2::Phase2L1CaloJet tempJet;
      int gctjeteta = jet[i].etaCenter;
      int gctjetphi = jet[i].phiCenter;
      tempJet.setJetIEta(gctjeteta + k * nBarrelEta / 2);
      tempJet.setJetIPhi(gctjetphi);
      float jeteta = realEta[gctjeteta + k * nBarrelEta / 2][gctjetphi];
      float jetphi = realPhi[gctjeteta + k * nBarrelEta / 2][gctjetphi];
      tempJet.setJetEta(jeteta);
      tempJet.setJetPhi(jetphi);
      tempJet.setJetEt(get_jet_pt_calibration(jet[i].energy, jeteta));
      tempJet.setTauEt(get_tau_pt_calibration(jet[i].tauEt, jeteta));
      tempJet.setTowerEt(jet[i].energyMax);
      int gcttowereta = jet[i].etaMax;
      int gcttowerphi = jet[i].phiMax;
      tempJet.setTowerIEta(gcttowereta + k * nBarrelEta / 2);
      tempJet.setTowerIPhi(gcttowerphi);
      float towereta = realEta[gcttowereta + k * nBarrelEta / 2][gcttowerphi];
      float towerphi = realPhi[gcttowereta + k * nBarrelEta / 2][gcttowerphi];
      tempJet.setTowerEta(towereta);
      tempJet.setTowerPhi(towerphi);
      reco::Candidate::PolarLorentzVector tempJetp4;
      tempJetp4.SetPt(tempJet.jetEt());
      tempJetp4.SetEta(tempJet.jetEta());
      tempJetp4.SetPhi(tempJet.jetPhi());
      tempJetp4.SetM(0.);
      tempJet.setP4(tempJetp4);

      if (jet[i].energy > 0.)
        halfBarrelJets.push_back(tempJet);
    }

    // ENDCAP
    for (int iphi = 0; iphi < nHgcalPhi; iphi++) {
      for (int ieta = 0; ieta < nHgcalEta / 2; ieta++) {
        if (k == 0)
          temporary_hgcal[ieta][iphi] = hgcalTowers[ieta][iphi];
        else
          temporary_hgcal[ieta][iphi] = hgcalTowers[nHgcalEta / 2 + ieta][iphi];
      }
    }

    gctobj::GCTsupertower_t tempST_hgcal[nSTEta][nSTPhi];
    gctobj::makeST_hgcal(temporary_hgcal, tempST_hgcal);
    float TTseedThresholdEndcap = 3.;
    for (int i = nJets; i < 2 * nJets; i++) {
      jet[i] = gctobj::getRegion(tempST_hgcal, TTseedThresholdEndcap);
      l1tp2::Phase2L1CaloJet tempJet;
      int hgcaljeteta = jet[i].etaCenter;
      int hgcaljetphi = jet[i].phiCenter;
      tempJet.setJetIEta(hgcaljeteta + k * nHgcalEta / 2);
      tempJet.setJetIPhi(hgcaljetphi);
      float jeteta = hgcalEta[hgcaljeteta + k * nHgcalEta / 2][hgcaljetphi];
      float jetphi = hgcalPhi[hgcaljeteta + k * nHgcalEta / 2][hgcaljetphi];
      tempJet.setJetEta(jeteta);
      tempJet.setJetPhi(jetphi);
      tempJet.setJetEt(get_jet_pt_calibration(jet[i].energy, jeteta));
      tempJet.setTauEt(get_tau_pt_calibration(jet[i].tauEt, jeteta));
      tempJet.setTowerEt(jet[i].energyMax);
      int hgcaltowereta = jet[i].etaMax;
      int hgcaltowerphi = jet[i].phiMax;
      tempJet.setTowerIEta(hgcaltowereta + k * nHgcalEta / 2);
      tempJet.setTowerIPhi(hgcaltowerphi);
      float towereta = hgcalEta[hgcaltowereta + k * nHgcalEta / 2][hgcaltowerphi];
      float towerphi = hgcalPhi[hgcaltowereta + k * nHgcalEta / 2][hgcaltowerphi];
      tempJet.setTowerEta(towereta);
      tempJet.setTowerPhi(towerphi);
      reco::Candidate::PolarLorentzVector tempJetp4;
      tempJetp4.SetPt(tempJet.jetEt());
      tempJetp4.SetEta(tempJet.jetEta());
      tempJetp4.SetPhi(tempJet.jetPhi());
      tempJetp4.SetM(0.);
      tempJet.setP4(tempJetp4);

      if (jet[i].energy > 0.)
        halfHgcalJets.push_back(tempJet);
    }

    // HF
    for (int iphi = 0; iphi < nHfPhi; iphi++) {
      for (int ieta = 0; ieta < nHfEta / 2; ieta++) {
        if (k == 0)
          temporary_hf[ieta][iphi] = hfTowers[ieta][iphi];
        else
          temporary_hf[ieta][iphi] = hfTowers[nHfEta / 2 + ieta][iphi];
      }
    }

    gctobj::GCTsupertower_t tempST_hf[nSTEta][nSTPhi];
    gctobj::makeST_hf(temporary_hf, tempST_hf);
    float TTseedThresholdHF = 5.;
    for (int i = 2 * nJets; i < 3 * nJets; i++) {
      jet[i] = gctobj::getRegion(tempST_hf, TTseedThresholdHF);
      l1tp2::Phase2L1CaloJet tempJet;
      int hfjeteta = jet[i].etaCenter;
      int hfjetphi = jet[i].phiCenter;
      tempJet.setJetIEta(hfjeteta + k * nHfEta / 2);
      tempJet.setJetIPhi(hfjetphi);
      float jeteta = hfEta[hfjeteta + k * nHfEta / 2][hfjetphi];
      float jetphi = hfPhi[hfjeteta + k * nHfEta / 2][hfjetphi];
      tempJet.setJetEta(jeteta);
      tempJet.setJetPhi(jetphi);
      tempJet.setJetEt(get_jet_pt_calibration(jet[i].energy, jeteta));
      tempJet.setTauEt(get_tau_pt_calibration(jet[i].tauEt, jeteta));
      tempJet.setTowerEt(jet[i].energyMax);
      int hftowereta = jet[i].etaMax;
      int hftowerphi = jet[i].phiMax;
      tempJet.setTowerIEta(hftowereta + k * nHfEta / 2);
      tempJet.setTowerIPhi(hftowerphi);
      float towereta = hfEta[hftowereta + k * nHfEta / 2][hftowerphi];
      float towerphi = hfPhi[hftowereta + k * nHfEta / 2][hftowerphi];
      tempJet.setTowerEta(towereta);
      tempJet.setTowerPhi(towerphi);
      reco::Candidate::PolarLorentzVector tempJetp4;
      tempJetp4.SetPt(tempJet.jetEt());
      tempJetp4.SetEta(tempJet.jetEta());
      tempJetp4.SetPhi(tempJet.jetPhi());
      tempJetp4.SetM(0.);
      tempJet.setP4(tempJetp4);

      if (jet[i].energy > 0.)
        halfHfJets.push_back(tempJet);
    }

    // Stitching:
    // if the jet eta is at the boundary: for HB it should be within 0-1 in -ve eta, 32-33 in +ve eta; for HE it should be within 0-1/16-17 in -ve eta, 34-35/18-19 in +ve eta; for HF it should be within 10-11 in -ve eta, 12-13 in +ve eta
    // then get the phi of that jet and check if there is a neighbouring jet with the same phi, then merge to the jet that has higher ET
    // in both eta/phi allow a maximum of one tower between jet centers for stitching

    for (size_t i = 0; i < halfHgcalJets.size(); i++) {
      if (halfHgcalJets.at(i).jetIEta() >= (nHgcalEta / 2 - 2) && halfHgcalJets.at(i).jetIEta() < (nHgcalEta / 2 + 2)) {
        float hgcal_ieta = k * nBarrelEta + halfHgcalJets.at(i).jetIEta();
        for (size_t j = 0; j < halfBarrelJets.size(); j++) {
          float barrel_ieta = nHgcalEta / 2 + halfBarrelJets.at(j).jetIEta();
          if (abs(barrel_ieta - hgcal_ieta) <= 2 &&
              abs(halfBarrelJets.at(j).jetIPhi() - halfHgcalJets.at(i).jetIPhi()) <= 2) {
            float totalet = halfBarrelJets.at(j).jetEt() + halfHgcalJets.at(i).jetEt();
            float totalTauEt = halfBarrelJets.at(j).tauEt() + halfHgcalJets.at(i).tauEt();
            if (halfBarrelJets.at(j).jetEt() > halfHgcalJets.at(i).jetEt()) {
              halfHgcalJets.at(i).setJetEt(0.);
              halfHgcalJets.at(i).setTauEt(0.);
              halfBarrelJets.at(j).setJetEt(totalet);
              halfBarrelJets.at(j).setTauEt(totalTauEt);
              reco::Candidate::PolarLorentzVector tempJetp4;
              tempJetp4.SetPt(totalet);
              tempJetp4.SetEta(halfBarrelJets.at(j).jetEta());
              tempJetp4.SetPhi(halfBarrelJets.at(j).jetPhi());
              tempJetp4.SetM(0.);
              halfBarrelJets.at(j).setP4(tempJetp4);
            } else {
              halfHgcalJets.at(i).setJetEt(totalet);
              halfHgcalJets.at(i).setTauEt(totalTauEt);
              halfBarrelJets.at(j).setJetEt(0.);
              halfBarrelJets.at(j).setTauEt(0.);
              reco::Candidate::PolarLorentzVector tempJetp4;
              tempJetp4.SetPt(totalet);
              tempJetp4.SetEta(halfHgcalJets.at(i).jetEta());
              tempJetp4.SetPhi(halfHgcalJets.at(i).jetPhi());
              tempJetp4.SetM(0.);
              halfHgcalJets.at(i).setP4(tempJetp4);
            }
          }
        }
      } else if (halfHgcalJets.at(i).jetIEta() < 2 || halfHgcalJets.at(i).jetIEta() >= (nHgcalEta - 2)) {
        float hgcal_ieta = k * nBarrelEta + nHfEta / 2 + halfHgcalJets.at(i).jetIEta();
        for (size_t j = 0; j < halfHfJets.size(); j++) {
          float hf_ieta = k * nBarrelEta + k * nHgcalEta + halfHfJets.at(j).jetIEta();
          if (abs(hgcal_ieta - hf_ieta) < 3 && abs(halfHfJets.at(j).jetIPhi() - halfHgcalJets.at(i).jetIPhi()) < 3) {
            float totalet = halfHfJets.at(j).jetEt() + halfHgcalJets.at(i).jetEt();
            float totalTauEt = halfHfJets.at(j).tauEt() + halfHgcalJets.at(i).tauEt();
            if (halfHfJets.at(j).jetEt() > halfHgcalJets.at(i).jetEt()) {
              halfHgcalJets.at(i).setJetEt(0.);
              halfHgcalJets.at(i).setTauEt(0.);
              halfHfJets.at(j).setJetEt(totalet);
              halfHfJets.at(j).setTauEt(totalTauEt);
              reco::Candidate::PolarLorentzVector tempJetp4;
              tempJetp4.SetPt(totalet);
              tempJetp4.SetEta(halfHfJets.at(j).jetEta());
              tempJetp4.SetPhi(halfHfJets.at(j).jetPhi());
              tempJetp4.SetM(0.);
              halfHfJets.at(j).setP4(tempJetp4);
            } else {
              halfHgcalJets.at(i).setJetEt(totalet);
              halfHgcalJets.at(i).setTauEt(totalTauEt);
              halfHfJets.at(j).setJetEt(0.);
              halfHfJets.at(j).setTauEt(0.);
              reco::Candidate::PolarLorentzVector tempJetp4;
              tempJetp4.SetPt(totalet);
              tempJetp4.SetEta(halfHgcalJets.at(i).jetEta());
              tempJetp4.SetPhi(halfHgcalJets.at(i).jetPhi());
              tempJetp4.SetM(0.);
              halfHgcalJets.at(i).setP4(tempJetp4);
            }
          }
        }
      }
    }

    // Write 6 leading jets from each eta half

    vector<l1tp2::Phase2L1CaloJet> halfAllJets;
    halfAllJets.clear();

    std::sort(halfBarrelJets.begin(), halfBarrelJets.end(), gctobj::compareByEt);
    for (size_t i = 0; i < halfBarrelJets.size(); i++) {
      if (halfBarrelJets.at(i).jetEt() > 0. && i < 6)
        halfAllJets.push_back(halfBarrelJets.at(i));
    }

    std::sort(halfHgcalJets.begin(), halfHgcalJets.end(), gctobj::compareByEt);
    for (size_t i = 0; i < halfHgcalJets.size(); i++) {
      if (halfHgcalJets.at(i).jetEt() > 0. && i < 6)
        halfAllJets.push_back(halfHgcalJets.at(i));
    }

    std::sort(halfHfJets.begin(), halfHfJets.end(), gctobj::compareByEt);
    for (size_t i = 0; i < halfHfJets.size(); i++) {
      if (halfHfJets.at(i).jetEt() > 0. && i < 6)
        halfAllJets.push_back(halfHfJets.at(i));
    }

    std::sort(halfAllJets.begin(), halfAllJets.end(), gctobj::compareByEt);
    for (size_t i = 0; i < halfAllJets.size(); i++) {
      if (halfAllJets.at(i).jetEt() > 0. && i < 6)
        allJets.push_back(halfAllJets.at(i));
    }
  }

  std::sort(allJets.begin(), allJets.end(), gctobj::compareByEt);
  for (size_t i = 0; i < allJets.size(); i++) {
    jetCands->push_back(allJets.at(i));
  }

  iEvent.put(std::move(jetCands), "GCTJet");
}

// Apply calibrations to HCAL energy based on Jet Eta, Jet pT
float Phase2L1CaloJetEmulator::get_jet_pt_calibration(const float& jet_pt, const float& jet_eta) const {
  float abs_eta = std::abs(jet_eta);
  float tmp_jet_pt = jet_pt;
  if (tmp_jet_pt > 499)
    tmp_jet_pt = 499;

  // Different indices sizes in different calo regions.
  // Barrel...
  size_t eta_index = 0;
  size_t pt_index = 0;
  float calib = 1.0;
  if (abs_eta <= 1.5) {
    // Start loop checking 2nd value
    for (unsigned int i = 1; i < absEtaBinsBarrel.size(); i++) {
      if (abs_eta <= absEtaBinsBarrel.at(i))
        break;
      eta_index++;
    }
    // Start loop checking 2nd value
    for (unsigned int i = 1; i < jetPtBins.size(); i++) {
      if (tmp_jet_pt <= jetPtBins.at(i))
        break;
      pt_index++;
    }
    calib = calibrationsBarrel[eta_index][pt_index];
  }                         // end Barrel
  else if (abs_eta <= 3.0)  // HGCal
  {
    // Start loop checking 2nd value
    for (unsigned int i = 1; i < absEtaBinsHGCal.size(); i++) {
      if (abs_eta <= absEtaBinsHGCal.at(i))
        break;
      eta_index++;
    }
    // Start loop checking 2nd value
    for (unsigned int i = 1; i < jetPtBins.size(); i++) {
      if (tmp_jet_pt <= jetPtBins.at(i))
        break;
      pt_index++;
    }
    calib = calibrationsHGCal[eta_index][pt_index];
  }     // end HGCal
  else  // HF
  {
    // Start loop checking 2nd value
    for (unsigned int i = 1; i < absEtaBinsHF.size(); i++) {
      if (abs_eta <= absEtaBinsHF.at(i))
        break;
      eta_index++;
    }
    // Start loop checking 2nd value
    for (unsigned int i = 1; i < jetPtBins.size(); i++) {
      if (tmp_jet_pt <= jetPtBins.at(i))
        break;
      pt_index++;
    }
    calib = calibrationsHF[eta_index][pt_index];
  }  // end HF

  return jet_pt * calib;
}

// Apply calibrations to tau pT based on L1EG info, EM Fraction, Tau Eta, Tau pT
float Phase2L1CaloJetEmulator::get_tau_pt_calibration(const float& tau_pt, const float& tau_eta) const {
  float abs_eta = std::abs(tau_eta);
  float tmp_tau_pt = tau_pt;
  if (tmp_tau_pt > 199)
    tmp_tau_pt = 199;

  // Different indices sizes in different calo regions.
  // Barrel...
  size_t eta_index = 0;
  size_t pt_index = 0;
  float calib = 1.0;
  if (abs_eta <= 1.5) {
    // Start loop checking 2nd value
    for (unsigned int i = 1; i < tauAbsEtaBinsBarrel.size(); i++) {
      if (abs_eta <= tauAbsEtaBinsBarrel.at(i))
        break;
      eta_index++;
    }
    // Start loop checking 2nd value
    for (unsigned int i = 1; i < tauPtBins.size(); i++) {
      if (tmp_tau_pt <= tauPtBins.at(i))
        break;
      pt_index++;
    }
    calib = tauPtCalibrationsBarrel[eta_index][pt_index];
  }                         // end Barrel
  else if (abs_eta <= 3.0)  // HGCal
  {
    // Start loop checking 2nd value
    for (unsigned int i = 1; i < tauAbsEtaBinsHGCal.size(); i++) {
      if (abs_eta <= tauAbsEtaBinsHGCal.at(i))
        break;
      eta_index++;
    }
    // Start loop checking 2nd value
    for (unsigned int i = 1; i < tauPtBins.size(); i++) {
      if (tmp_tau_pt <= tauPtBins.at(i))
        break;
      pt_index++;
    }
    calib = tauPtCalibrationsHGCal[eta_index][pt_index];
  }  // end HGCal

  return tau_pt * calib;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void Phase2L1CaloJetEmulator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gctFullTowers", edm::InputTag("l1tPhase2L1CaloEGammaEmulator", "GCTFullTowers"));
  desc.add<edm::InputTag>("hgcalTowers", edm::InputTag("l1tHGCalTowerProducer", "HGCalTowerProcessor"));
  desc.add<edm::InputTag>("hcalDigis", edm::InputTag("simHcalTriggerPrimitiveDigis"));

  edm::ParameterSetDescription nHits_params_validator;
  nHits_params_validator.add<string>("fit", "type");
  nHits_params_validator.add<vector<double>>("nHits_params", {1., 1.});
  std::vector<edm::ParameterSet> nHits_params_default;
  edm::ParameterSet nHits_params1;
  nHits_params1.addParameter<string>("fit", "hgcalEM");
  nHits_params1.addParameter<vector<double>>("nHits_params", {157.522, 0.090});
  nHits_params_default.push_back(nHits_params1);
  edm::ParameterSet nHits_params2;
  nHits_params2.addParameter<string>("fit", "hgcalHad");
  nHits_params2.addParameter<vector<double>>("nHits_params", {159.295, 0.178});
  nHits_params_default.push_back(nHits_params2);
  edm::ParameterSet nHits_params3;
  nHits_params3.addParameter<string>("fit", "hf");
  nHits_params3.addParameter<vector<double>>("nHits_params", {165.706, 0.153});
  nHits_params_default.push_back(nHits_params3);
  desc.addVPSet("nHits_to_nvtx_params", nHits_params_validator, nHits_params_default);

  edm::ParameterSetDescription nvtx_params_validator;
  nvtx_params_validator.add<string>("calo", "type");
  nvtx_params_validator.add<string>("iEta", "etaregion");
  nvtx_params_validator.add<vector<double>>("nvtx_params", {1., 1.});
  std::vector<edm::ParameterSet> nvtx_params_default;
  edm::ParameterSet nvtx_params1;
  nvtx_params1.addParameter<string>("calo", "hgcalEM");
  nvtx_params1.addParameter<string>("iEta", "er1p4to1p8");
  nvtx_params1.addParameter<vector<double>>("nvtx_params", {-0.011772, 0.004142});
  nvtx_params_default.push_back(nvtx_params1);
  edm::ParameterSet nvtx_params2;
  nvtx_params2.addParameter<string>("calo", "hgcalEM");
  nvtx_params2.addParameter<string>("iEta", "er1p8to2p1");
  nvtx_params2.addParameter<vector<double>>("nvtx_params", {-0.015488, 0.005410});
  nvtx_params_default.push_back(nvtx_params2);
  edm::ParameterSet nvtx_params3;
  nvtx_params3.addParameter<string>("calo", "hgcalEM");
  nvtx_params3.addParameter<string>("iEta", "er2p1to2p4");
  nvtx_params3.addParameter<vector<double>>("nvtx_params", {-0.021150, 0.006078});
  nvtx_params_default.push_back(nvtx_params3);
  edm::ParameterSet nvtx_params4;
  nvtx_params4.addParameter<string>("calo", "hgcalEM");
  nvtx_params4.addParameter<string>("iEta", "er2p4to2p7");
  nvtx_params4.addParameter<vector<double>>("nvtx_params", {-0.015705, 0.005339});
  nvtx_params_default.push_back(nvtx_params4);
  edm::ParameterSet nvtx_params5;
  nvtx_params5.addParameter<string>("calo", "hgcalEM");
  nvtx_params5.addParameter<string>("iEta", "er2p7to3p1");
  nvtx_params5.addParameter<vector<double>>("nvtx_params", {-0.018492, 0.005620});
  nvtx_params_default.push_back(nvtx_params5);
  edm::ParameterSet nvtx_params6;
  nvtx_params6.addParameter<string>("calo", "hgcalHad");
  nvtx_params6.addParameter<string>("iEta", "er1p4to1p8");
  nvtx_params6.addParameter<vector<double>>("nvtx_params", {0.005675, 0.000615});
  nvtx_params_default.push_back(nvtx_params6);
  edm::ParameterSet nvtx_params7;
  nvtx_params7.addParameter<string>("calo", "hgcalHad");
  nvtx_params7.addParameter<string>("iEta", "er1p8to2p1");
  nvtx_params7.addParameter<vector<double>>("nvtx_params", {0.004560, 0.001099});
  nvtx_params_default.push_back(nvtx_params7);
  edm::ParameterSet nvtx_params8;
  nvtx_params8.addParameter<string>("calo", "hgcalHad");
  nvtx_params8.addParameter<string>("iEta", "er2p1to2p4");
  nvtx_params8.addParameter<vector<double>>("nvtx_params", {0.000036, 0.001608});
  nvtx_params_default.push_back(nvtx_params8);
  edm::ParameterSet nvtx_params9;
  nvtx_params9.addParameter<string>("calo", "hgcalHad");
  nvtx_params9.addParameter<string>("iEta", "er2p4to2p7");
  nvtx_params9.addParameter<vector<double>>("nvtx_params", {0.000869, 0.001754});
  nvtx_params_default.push_back(nvtx_params9);
  edm::ParameterSet nvtx_params10;
  nvtx_params10.addParameter<string>("calo", "hgcalHad");
  nvtx_params10.addParameter<string>("iEta", "er2p7to3p1");
  nvtx_params10.addParameter<vector<double>>("nvtx_params", {-0.006574, 0.003134});
  nvtx_params_default.push_back(nvtx_params10);
  edm::ParameterSet nvtx_params11;
  nvtx_params11.addParameter<string>("calo", "hf");
  nvtx_params11.addParameter<string>("iEta", "er29to33");
  nvtx_params11.addParameter<vector<double>>("nvtx_params", {-0.203291, 0.044096});
  nvtx_params_default.push_back(nvtx_params11);
  edm::ParameterSet nvtx_params12;
  nvtx_params12.addParameter<string>("calo", "hf");
  nvtx_params12.addParameter<string>("iEta", "er34to37");
  nvtx_params12.addParameter<vector<double>>("nvtx_params", {-0.210922, 0.045628});
  nvtx_params_default.push_back(nvtx_params12);
  edm::ParameterSet nvtx_params13;
  nvtx_params13.addParameter<string>("calo", "hf");
  nvtx_params13.addParameter<string>("iEta", "er38to41");
  nvtx_params13.addParameter<vector<double>>("nvtx_params", {-0.229562, 0.050560});
  nvtx_params_default.push_back(nvtx_params13);
  desc.addVPSet("nvtx_to_PU_sub_params", nvtx_params_validator, nvtx_params_default);

  desc.add<vector<double>>("jetPtBins", {0.0,   5.0,   7.5,   10.0,  12.5,  15.0,  17.5,  20.0,  22.5,  25.0,  27.5,
                                         30.0,  35.0,  40.0,  45.0,  50.0,  55.0,  60.0,  65.0,  70.0,  75.0,  80.0,
                                         85.0,  90.0,  95.0,  100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0,
                                         180.0, 190.0, 200.0, 225.0, 250.0, 275.0, 300.0, 325.0, 400.0, 500.0});
  desc.add<vector<double>>("absEtaBinsBarrel", {0.00, 0.30, 0.60, 1.00, 1.50});
  desc.add<vector<double>>(
      "jetCalibrationsBarrel",
      {2.459, 2.320, 2.239, 2.166, 2.100, 2.040, 1.986, 1.937, 1.892, 1.852, 1.816, 1.768, 1.714, 1.670, 1.633, 1.603,
       1.578, 1.557, 1.540, 1.525, 1.513, 1.502, 1.493, 1.486, 1.479, 1.470, 1.460, 1.452, 1.445, 1.439, 1.433, 1.427,
       1.422, 1.417, 1.411, 1.403, 1.390, 1.377, 1.365, 1.352, 1.327, 1.284, 4.695, 3.320, 2.751, 2.361, 2.093, 1.908,
       1.781, 1.694, 1.633, 1.591, 1.562, 1.533, 1.511, 1.499, 1.492, 1.486, 1.482, 1.478, 1.474, 1.470, 1.467, 1.463,
       1.459, 1.456, 1.452, 1.447, 1.440, 1.433, 1.425, 1.418, 1.411, 1.404, 1.397, 1.390, 1.382, 1.370, 1.352, 1.334,
       1.316, 1.298, 1.262, 1.200, 5.100, 3.538, 2.892, 2.448, 2.143, 1.933, 1.789, 1.689, 1.620, 1.572, 1.539, 1.506,
       1.482, 1.469, 1.460, 1.455, 1.450, 1.446, 1.442, 1.438, 1.434, 1.431, 1.427, 1.423, 1.420, 1.414, 1.407, 1.400,
       1.392, 1.385, 1.378, 1.370, 1.363, 1.356, 1.348, 1.336, 1.317, 1.299, 1.281, 1.263, 1.226, 1.162, 3.850, 3.438,
       3.211, 3.017, 2.851, 2.708, 2.585, 2.479, 2.388, 2.310, 2.243, 2.159, 2.072, 2.006, 1.956, 1.917, 1.887, 1.863,
       1.844, 1.828, 1.814, 1.802, 1.791, 1.782, 1.773, 1.760, 1.744, 1.729, 1.714, 1.699, 1.685, 1.670, 1.656, 1.641,
       1.627, 1.602, 1.566, 1.530, 1.494, 1.458, 1.386, 1.260});
  desc.add<vector<double>>("absEtaBinsHGCal", {1.50, 1.90, 2.40, 3.00});
  desc.add<vector<double>>(
      "jetCalibrationsHGCal",
      {5.604,   4.578,  4.061,  3.647, 3.314, 3.047, 2.832, 2.660, 2.521, 2.410, 2.320, 2.216, 2.120, 2.056,
       2.013,   1.983,  1.961,  1.945, 1.932, 1.922, 1.913, 1.905, 1.898, 1.891, 1.884, 1.874, 1.861, 1.848,
       1.835,   1.822,  1.810,  1.797, 1.784, 1.771, 1.759, 1.736, 1.704, 1.673, 1.641, 1.609, 1.545, 1.434,
       4.385,   3.584,  3.177,  2.849, 2.584, 2.370, 2.197, 2.057, 1.944, 1.853, 1.780, 1.695, 1.616, 1.564,
       1.530,   1.507,  1.491,  1.480, 1.472, 1.466, 1.462, 1.459, 1.456, 1.453, 1.451, 1.447, 1.443, 1.439,
       1.435,   1.431,  1.427,  1.423, 1.419, 1.416, 1.412, 1.405, 1.395, 1.385, 1.376, 1.366, 1.346, 1.312,
       562.891, 68.647, 17.648, 5.241, 2.223, 1.490, 1.312, 1.270, 1.260, 1.259, 1.259, 1.260, 1.263, 1.265,
       1.267,   1.269,  1.271,  1.273, 1.275, 1.277, 1.279, 1.281, 1.283, 1.285, 1.287, 1.290, 1.295, 1.299,
       1.303,   1.307,  1.311,  1.315, 1.319, 1.323, 1.328, 1.335, 1.345, 1.355, 1.366, 1.376, 1.397, 1.433});
  desc.add<vector<double>>("absEtaBinsHF", {3.00, 3.60, 6.00});
  desc.add<vector<double>>(
      "jetCalibrationsHF",
      {8.169, 6.873, 6.155, 5.535, 5.001, 4.539, 4.141, 3.798, 3.501, 3.245, 3.024, 2.748, 2.463, 2.249,
       2.090, 1.971, 1.881, 1.814, 1.763, 1.725, 1.695, 1.673, 1.655, 1.642, 1.631, 1.618, 1.605, 1.596,
       1.588, 1.581, 1.575, 1.569, 1.563, 1.557, 1.551, 1.541, 1.527, 1.513, 1.498, 1.484, 1.456, 1.406,
       2.788, 2.534, 2.388, 2.258, 2.141, 2.037, 1.945, 1.862, 1.788, 1.722, 1.664, 1.587, 1.503, 1.436,
       1.382, 1.339, 1.305, 1.277, 1.255, 1.237, 1.223, 1.211, 1.201, 1.193, 1.186, 1.178, 1.170, 1.164,
       1.159, 1.154, 1.151, 1.147, 1.144, 1.141, 1.138, 1.133, 1.126, 1.118, 1.111, 1.104, 1.090, 1.064});
  desc.add<vector<double>>("tauPtBins", {0.0,  5.0,  7.5,  10.0, 12.5, 15.0, 20.0, 25.0,  30.0,  35.0,
                                         40.0, 45.0, 50.0, 55.0, 60.0, 70.0, 80.0, 100.0, 150.0, 200.0});
  desc.add<vector<double>>("tauAbsEtaBinsBarrel", {0.00, 0.30, 0.60, 1.00, 1.50});
  desc.add<vector<double>>("tauCalibrationsBarrel",
                           {1.067, 1.067, 1.067, 1.067, 1.067, 1.067, 1.067, 1.067, 1.067, 1.067, 1.067, 1.067, 1.067,
                            1.067, 1.067, 1.067, 1.067, 1.067, 1.067, 1.106, 1.106, 1.106, 1.106, 1.106, 1.106, 1.106,
                            1.106, 1.106, 1.106, 1.106, 1.106, 1.106, 1.106, 1.106, 1.106, 1.106, 1.106, 1.106, 1.102,
                            1.102, 1.102, 1.102, 1.102, 1.102, 1.102, 1.102, 1.102, 1.102, 1.102, 1.102, 1.102, 1.102,
                            1.102, 1.102, 1.102, 1.102, 1.102, 1.139, 1.139, 1.139, 1.139, 1.139, 1.139, 1.139, 1.139});
  desc.add<vector<double>>("tauAbsEtaBinsHGCal", {1.50, 1.90, 2.40, 3.00});
  desc.add<vector<double>>(
      "tauCalibrationsHGCal",
      {1.384, 1.384, 1.384, 1.384, 1.384, 1.384, 1.384, 1.384, 1.384, 1.384, 1.384, 1.384, 1.384, 1.384, 1.384,
       1.384, 1.384, 1.384, 1.384, 1.473, 1.473, 1.473, 1.473, 1.473, 1.473, 1.473, 1.473, 1.473, 1.473, 1.473,
       1.473, 1.473, 1.473, 1.473, 1.473, 1.473, 1.473, 1.473, 1.133, 1.133, 1.133, 1.133, 1.133, 1.133, 1.133,
       1.133, 1.133, 1.133, 1.133, 1.133, 1.133, 1.133, 1.133, 1.133, 1.133, 1.133, 1.133});

  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2L1CaloJetEmulator);
