// -*- C++ -*-
//
// Package: L1CaloTrigger
// Class: L1CaloJetProducer
//
/**\class L1CaloJetProducer L1CaloJetProducer.cc

Description: 
Beginning with HCAL TPs, create HCAL jet, then
take L1EG crystal clusters from L1EGammaCrystalsProducer.cc
and clusters them within fixed number of trigger towers

Implementation:
[Notes on implementation]
*/
//
// Original Author: Tyler Ruggles
// Created: Tue Aug 29 2018
// $Id$
//
//

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
//#include "DataFormats/L1TCalorimeterPhase2/interface/CaloCrystalCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/CaloJet.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/CaloTower.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

// For pT calibrations
#include "TF1.h"

// Run2/PhaseI output formats
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"

class L1CaloJetProducer : public edm::EDProducer {
public:
  explicit L1CaloJetProducer(const edm::ParameterSet &);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;
  //bool cluster_passes_base_cuts(float &cluster_pt, float &cluster_eta, float &iso, float &e2x5, float &e5x5) const;
  int tower_diPhi(int &iPhi_1, int &iPhi_2) const;
  int tower_diEta(int &iEta_1, int &iEta_2) const;
  float get_deltaR(reco::Candidate::PolarLorentzVector &p4_1, reco::Candidate::PolarLorentzVector &p4_2) const;
  float get_hcal_calibration(float &jet_pt, float &ecal_pt, float &ecal_L1EG_jet_pt, float &jet_eta) const;
  float get_tau_pt_calibration(float &tau_pt, float &ecal_pt, float &l1EG_pt, float &n_L1EGs, float &tau_eta) const;
  int loose_iso_tau_wp(float &tau_pt, float &tau_iso_et, float &tau_eta) const;

  double HcalTpEtMin;
  double EcalTpEtMin;
  double HGCalHadTpEtMin;
  double HGCalEmTpEtMin;
  double HFTpEtMin;
  double EtMinForSeedHit;
  double EtMinForCollection;
  double EtMinForTauCollection;

  // For fetching jet calibrations
  std::vector<double> jetPtBins;
  std::vector<double> emFractionBinsBarrel;
  std::vector<double> absEtaBinsBarrel;
  std::vector<double> jetCalibrationsBarrel;
  std::vector<double> emFractionBinsHGCal;
  std::vector<double> absEtaBinsHGCal;
  std::vector<double> jetCalibrationsHGCal;
  std::vector<double> emFractionBinsHF;
  std::vector<double> absEtaBinsHF;
  std::vector<double> jetCalibrationsHF;

  // For fetching tau calibrations
  std::vector<double> tauPtBins;
  std::vector<double> tauAbsEtaBinsBarrel;
  std::vector<double> tauCalibrationsBarrel;
  std::vector<edm::ParameterSet> tauL1egInfoBarrel;
  std::vector<double> tauAbsEtaBinsHGCal;
  std::vector<double> tauCalibrationsHGCal;
  std::vector<edm::ParameterSet> tauL1egInfoHGCal;

  // For storing jet calibrations
  std::vector<std::vector<std::vector<double>>> calibrationsBarrel;
  std::vector<std::vector<std::vector<double>>> calibrationsHGCal;
  std::vector<std::vector<std::vector<double>>> calibrationsHF;

  // For storing tau calibration info
  std::map<double, std::vector<double>> tauL1egInfoMapBarrel;
  std::map<double, std::vector<double>> tauL1egInfoMapHGCal;
  std::vector<double> tauL1egValuesBarrel;  // To preserve ordering
  std::vector<double> tauL1egValuesHGCal;   // To preserve ordering
  std::vector<std::vector<std::vector<std::vector<double>>>> tauPtCalibrationsBarrel;
  std::vector<std::vector<std::vector<std::vector<double>>>> tauPtCalibrationsHGCal;

  bool debug;
  edm::EDGetTokenT<l1tp2::CaloTowerCollection> l1TowerToken_;
  edm::Handle<l1tp2::CaloTowerCollection> l1CaloTowerHandle;

  // TF1s defining tau isolation thresholds
  TF1 isoTauBarrel = TF1("isoTauBarrelFunction", "([0] + [1]*TMath::Exp(-[2]*x))");
  TF1 isoTauHGCal = TF1("isoTauHGCalFunction", "([0] + [1]*TMath::Exp(-[2]*x))");

  class l1CaloJetObj {
  public:
    bool barrelSeeded = true;  // default to barrel seeded
    reco::Candidate::PolarLorentzVector jetCluster;
    reco::Candidate::PolarLorentzVector hcalJetCluster;
    reco::Candidate::PolarLorentzVector ecalJetCluster;
    reco::Candidate::PolarLorentzVector seedTower;
    //reco::Candidate::PolarLorentzVector leadingL1EG;
    reco::Candidate::PolarLorentzVector l1egJetCluster;
    float jetClusterET = 0.;
    float hcalJetClusterET = 0.;
    float ecalJetClusterET = 0.;
    float seedTowerET = 0.;
    float leadingL1EGET = 0.;
    float l1egJetClusterET = 0.;

    // For decay mode related checks with CaloTaus
    std::vector<std::vector<float>> associated_l1EGs_;

    int seed_iEta = -99;
    int seed_iPhi = -99;

    float hcal_seed = 0.;
    //float hcal_3x3 = 0.;
    float hcal_3x5 = 0.;
    //float hcal_5x5 = 0.;
    //float hcal_5x7 = 0.;
    float hcal_7x7 = 0.;
    //float hcal_2x3 = 0.;
    //float hcal_2x3_1 = 0.;
    //float hcal_2x3_2 = 0.;
    //float hcal_2x2 = 0.;
    //float hcal_2x2_1 = 0.;
    //float hcal_2x2_2 = 0.;
    //float hcal_2x2_3 = 0.;
    //float hcal_2x2_4 = 0.;
    float hcal_nHits = 0.;

    float ecal_seed = 0.;
    //float ecal_3x3 = 0.;
    float ecal_3x5 = 0.;
    //float ecal_5x5 = 0.;
    //float ecal_5x7 = 0.;
    float ecal_7x7 = 0.;
    //float ecal_2x3 = 0.;
    //float ecal_2x3_1 = 0.;
    //float ecal_2x3_2 = 0.;
    //float ecal_2x2 = 0.;
    //float ecal_2x2_1 = 0.;
    //float ecal_2x2_2 = 0.;
    //float ecal_2x2_3 = 0.;
    //float ecal_2x2_4 = 0.;
    float ecal_nHits = 0.;

    float l1eg_seed = 0.;
    //float l1eg_3x3 = 0.;
    float l1eg_3x5 = 0.;
    //float l1eg_5x5 = 0.;
    //float l1eg_5x7 = 0.;
    float l1eg_7x7 = 0.;
    //float l1eg_2x3 = 0.;
    //float l1eg_2x3_1 = 0.;
    //float l1eg_2x3_2 = 0.;
    //float l1eg_2x2 = 0.;
    //float l1eg_2x2_1 = 0.;
    //float l1eg_2x2_2 = 0.;
    //float l1eg_2x2_3 = 0.;
    //float l1eg_2x2_4 = 0.;
    float l1eg_nHits = 0.;
    float n_l1eg_HoverE_LessThreshold = 0.;

    float l1eg_nL1EGs = 0.;
    float l1eg_nL1EGs_standaloneSS = 0.;
    float l1eg_nL1EGs_standaloneIso = 0.;
    float l1eg_nL1EGs_trkMatchSS = 0.;
    float l1eg_nL1EGs_trkMatchIso = 0.;

    float total_seed = 0.;
    //float total_3x3 = 0.;
    float total_3x5 = 0.;
    //float total_5x5 = 0.;
    //float total_5x7 = 0.;
    float total_7x7 = 0.;
    //float total_2x3 = 0.;
    //float total_2x3_1 = 0.;
    //float total_2x3_2 = 0.;
    //float total_2x2 = 0.;
    //float total_2x2_1 = 0.;
    //float total_2x2_2 = 0.;
    //float total_2x2_3 = 0.;
    //float total_2x2_4 = 0.;
    float total_nHits = 0.;

    void Init() {
      SetJetClusterP4(0., 0., 0., 0.);
      SetHcalJetClusterP4(0., 0., 0., 0.);
      SetEcalJetClusterP4(0., 0., 0., 0.);
      SetSeedP4(0., 0., 0., 0.);
      //SetLeadingL1EGP4( 0., 0., 0., 0. );
      SetL1EGJetP4(0., 0., 0., 0.);
    }

    void SetJetClusterP4(double pt, double eta, double phi, double mass) {
      this->jetCluster.SetPt(pt);
      this->jetCluster.SetEta(eta);
      this->jetCluster.SetPhi(phi);
      this->jetCluster.SetM(mass);
    }
    void SetHcalJetClusterP4(double pt, double eta, double phi, double mass) {
      this->hcalJetCluster.SetPt(pt);
      this->hcalJetCluster.SetEta(eta);
      this->hcalJetCluster.SetPhi(phi);
      this->hcalJetCluster.SetM(mass);
    }
    void SetEcalJetClusterP4(double pt, double eta, double phi, double mass) {
      this->ecalJetCluster.SetPt(pt);
      this->ecalJetCluster.SetEta(eta);
      this->ecalJetCluster.SetPhi(phi);
      this->ecalJetCluster.SetM(mass);
    }
    void SetSeedP4(double pt, double eta, double phi, double mass) {
      this->seedTower.SetPt(pt);
      this->seedTower.SetEta(eta);
      this->seedTower.SetPhi(phi);
      this->seedTower.SetM(mass);
    }
    //void SetLeadingL1EGP4( double pt, double eta, double phi, double mass )
    //{
    //    this->leadingL1EG.SetPt( pt );
    //    this->leadingL1EG.SetEta( eta );
    //    this->leadingL1EG.SetPhi( phi );
    //    this->leadingL1EG.SetM( mass );
    //}
    void SetL1EGJetP4(double pt, double eta, double phi, double mass) {
      this->l1egJetCluster.SetPt(pt);
      this->l1egJetCluster.SetEta(eta);
      this->l1egJetCluster.SetPhi(phi);
      this->l1egJetCluster.SetM(mass);
    }
  };

  class simpleL1obj {
  public:
    bool stale = false;  // Hits become stale once used in clustering algorithm to prevent overlap in clusters
    bool associated_with_tower =
        false;                         // L1EGs become associated with a tower to find highest ET total for seeding jets
    bool passesStandaloneSS = false;   // Store whether any of the portions of a WP are passed
    bool passesStandaloneIso = false;  // Store whether any of the portions of a WP are passed
    bool passesTrkMatchSS = false;     // Store whether any of the portions of a WP are passed
    bool passesTrkMatchIso = false;    // Store whether any of the portions of a WP are passed
    reco::Candidate::PolarLorentzVector p4;

    void SetP4(double pt, double eta, double phi, double mass) {
      this->p4.SetPt(pt);
      this->p4.SetEta(eta);
      this->p4.SetPhi(phi);
      this->p4.SetM(mass);
    }
    inline float pt() const { return p4.pt(); };
    inline float eta() const { return p4.eta(); };
    inline float phi() const { return p4.phi(); };
    inline float M() const { return p4.M(); };
    inline reco::Candidate::PolarLorentzVector GetP4() const { return p4; };
  };

  class SimpleCaloHit {
  public:
    int towerIEta = -99;
    int towerIPhi = -99;
    float towerEta = -99;
    float towerPhi = -99;
    float ecalTowerEt = 0.;
    float hcalTowerEt = 0.;
    float l1egTowerEt = 0.;
    float total_tower_et = 0.;
    //float total_tower_plus_L1EGs_et=0.;
    bool stale = false;    // Hits become stale once used in clustering algorithm to prevent overlap in clusters
    bool isBarrel = true;  // Defaults to a barrel hit

    // L1EG info
    int nL1eg = 0;
    int l1egTrkSS = 0;
    int l1egTrkIso = 0;
    int l1egStandaloneSS = 0;
    int l1egStandaloneIso = 0;
  };
};

L1CaloJetProducer::L1CaloJetProducer(const edm::ParameterSet &iConfig)
    : HcalTpEtMin(iConfig.getParameter<double>("HcalTpEtMin")),                      // Should default to 0 MeV
      EcalTpEtMin(iConfig.getParameter<double>("EcalTpEtMin")),                      // Should default to 0 MeV
      HGCalHadTpEtMin(iConfig.getParameter<double>("HGCalHadTpEtMin")),              // Should default to 0 MeV
      HGCalEmTpEtMin(iConfig.getParameter<double>("HGCalEmTpEtMin")),                // Should default to 0 MeV
      HFTpEtMin(iConfig.getParameter<double>("HFTpEtMin")),                          // Should default to 0 MeV
      EtMinForSeedHit(iConfig.getParameter<double>("EtMinForSeedHit")),              // Should default to 2.5 GeV
      EtMinForCollection(iConfig.getParameter<double>("EtMinForCollection")),        // Testing 10 GeV
      EtMinForTauCollection(iConfig.getParameter<double>("EtMinForTauCollection")),  // Testing 10 GeV
      jetPtBins(iConfig.getParameter<std::vector<double>>("jetPtBins")),
      emFractionBinsBarrel(iConfig.getParameter<std::vector<double>>("emFractionBinsBarrel")),
      absEtaBinsBarrel(iConfig.getParameter<std::vector<double>>("absEtaBinsBarrel")),
      jetCalibrationsBarrel(iConfig.getParameter<std::vector<double>>("jetCalibrationsBarrel")),
      emFractionBinsHGCal(iConfig.getParameter<std::vector<double>>("emFractionBinsHGCal")),
      absEtaBinsHGCal(iConfig.getParameter<std::vector<double>>("absEtaBinsHGCal")),
      jetCalibrationsHGCal(iConfig.getParameter<std::vector<double>>("jetCalibrationsHGCal")),
      emFractionBinsHF(iConfig.getParameter<std::vector<double>>("emFractionBinsHF")),
      absEtaBinsHF(iConfig.getParameter<std::vector<double>>("absEtaBinsHF")),
      jetCalibrationsHF(iConfig.getParameter<std::vector<double>>("jetCalibrationsHF")),
      tauPtBins(iConfig.getParameter<std::vector<double>>("tauPtBins")),
      tauAbsEtaBinsBarrel(iConfig.getParameter<std::vector<double>>("tauAbsEtaBinsBarrel")),
      tauCalibrationsBarrel(iConfig.getParameter<std::vector<double>>("tauCalibrationsBarrel")),
      tauL1egInfoBarrel(iConfig.getParameter<std::vector<edm::ParameterSet>>("tauL1egInfoBarrel")),
      tauAbsEtaBinsHGCal(iConfig.getParameter<std::vector<double>>("tauAbsEtaBinsHGCal")),
      tauCalibrationsHGCal(iConfig.getParameter<std::vector<double>>("tauCalibrationsHGCal")),
      tauL1egInfoHGCal(iConfig.getParameter<std::vector<edm::ParameterSet>>("tauL1egInfoHGCal")),
      debug(iConfig.getParameter<bool>("debug")),
      l1TowerToken_(consumes<l1tp2::CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("l1CaloTowers")))

{
  produces<l1tp2::CaloJetsCollection>("L1CaloJetsNoCuts");
  //produces<l1tp2::CaloJetsCollection>("L1CaloJetsWithCuts");
  //produces<l1extra::L1JetParticleCollection>("L1CaloClusterCollectionWithCuts");
  produces<BXVector<l1t::Jet>>("L1CaloJetCollectionBXV");
  produces<BXVector<l1t::Tau>>("L1CaloTauCollectionBXV");

  if (debug) {
    LogDebug("L1CaloJetProducer") << " HcalTpEtMin = " << HcalTpEtMin << " EcalTpEtMin = " << EcalTpEtMin << "\n";
  }

  // Fill the calibration 3D vector
  // Dimension 1 is AbsEta bin
  // Dimension 2 is EM Fraction bin
  // Dimension 3 is jet pT bin which is filled with the actual callibration value
  // size()-1 b/c the inputs have lower and upper bounds
  // Do Barrel, then HGCal, then HF
  int index = 0;
  //calibrations[em_frac][abs_eta].push_back( jetCalibrationsBarrel.at(index) );
  for (unsigned int abs_eta = 0; abs_eta < absEtaBinsBarrel.size() - 1; abs_eta++) {
    std::vector<std::vector<double>> em_bins;
    for (unsigned int em_frac = 0; em_frac < emFractionBinsBarrel.size() - 1; em_frac++) {
      std::vector<double> pt_bin_calibs;
      for (unsigned int pt = 0; pt < jetPtBins.size() - 1; pt++) {
        pt_bin_calibs.push_back(jetCalibrationsBarrel.at(index));
        index++;
      }
      em_bins.push_back(pt_bin_calibs);
    }
    calibrationsBarrel.push_back(em_bins);
  }
  if (debug) {
    LogDebug("L1CaloJetProducer") << " Loading Barrel calibrations: Loaded " << index
                                  << " values vs. size() of input calibration file: "
                                  << int(jetCalibrationsBarrel.size()) << "\n";
  }

  index = 0;
  //calibrations[em_frac][abs_eta].push_back( jetCalibrationsHGCal.at(index) );
  for (unsigned int abs_eta = 0; abs_eta < absEtaBinsHGCal.size() - 1; abs_eta++) {
    std::vector<std::vector<double>> em_bins;
    for (unsigned int em_frac = 0; em_frac < emFractionBinsHGCal.size() - 1; em_frac++) {
      std::vector<double> pt_bin_calibs;
      for (unsigned int pt = 0; pt < jetPtBins.size() - 1; pt++) {
        pt_bin_calibs.push_back(jetCalibrationsHGCal.at(index));
        index++;
      }
      em_bins.push_back(pt_bin_calibs);
    }
    calibrationsHGCal.push_back(em_bins);
  }
  if (debug) {
    LogDebug("L1CaloJetProducer") << " Loading HGCal calibrations: Loaded " << index
                                  << " values vs. size() of input calibration file: "
                                  << int(jetCalibrationsHGCal.size()) << "\n";
  }

  index = 0;
  //calibrations[em_frac][abs_eta].push_back( jetCalibrationsHF.at(index) );
  for (unsigned int abs_eta = 0; abs_eta < absEtaBinsHF.size() - 1; abs_eta++) {
    std::vector<std::vector<double>> em_bins;
    for (unsigned int em_frac = 0; em_frac < emFractionBinsHF.size() - 1; em_frac++) {
      std::vector<double> pt_bin_calibs;
      for (unsigned int pt = 0; pt < jetPtBins.size() - 1; pt++) {
        pt_bin_calibs.push_back(jetCalibrationsHF.at(index));
        index++;
      }
      em_bins.push_back(pt_bin_calibs);
    }
    calibrationsHF.push_back(em_bins);
  }
  if (debug) {
    LogDebug("L1CaloJetProducer") << " Loading HF calibrations: Loaded " << index
                                  << " values vs. size() of input calibration file: " << int(jetCalibrationsHF.size())
                                  << "\n";
  }

  // Load Tau L1EG-base calibration info into maps
  for (auto &first : tauL1egInfoBarrel) {
    if (debug) {
      LogDebug("L1CaloJetProducer") << " barrel l1egCount = " << first.getParameter<double>("l1egCount") << "\n";
      for (auto &em_frac : first.getParameter<std::vector<double>>("l1egEmFractions")) {
        LogDebug("L1CaloJetProducer") << " - EM = " << em_frac << "\n";
      }
    }
    double l1egCount = first.getParameter<double>("l1egCount");
    std::vector<double> l1egEmFractions = first.getParameter<std::vector<double>>("l1egEmFractions");
    tauL1egInfoMapBarrel[l1egCount] = l1egEmFractions;
    tauL1egValuesBarrel.push_back(l1egCount);
  }
  std::sort(tauL1egValuesBarrel.begin(), tauL1egValuesBarrel.end());
  for (auto &first : tauL1egInfoHGCal) {
    if (debug) {
      LogDebug("L1CaloJetProducer") << " hgcal l1egCount = " << first.getParameter<double>("l1egCount") << "\n";
      for (auto &em_frac : first.getParameter<std::vector<double>>("l1egEmFractions")) {
        LogDebug("L1CaloJetProducer") << " - EM = " << em_frac << "\n";
      }
    }
    double l1egCount = first.getParameter<double>("l1egCount");
    std::vector<double> l1egEmFractions = first.getParameter<std::vector<double>>("l1egEmFractions");
    tauL1egInfoMapHGCal[l1egCount] = l1egEmFractions;
    tauL1egValuesHGCal.push_back(l1egCount);
  }
  std::sort(tauL1egValuesHGCal.begin(), tauL1egValuesHGCal.end());
  // Fill the calibration 4D vector
  // Dimension 1 is AbsEta bin
  // Dimension 2 is L1EG count
  // Dimension 3 is EM Fraction bin
  // Dimension 4 is tau pT bin which is filled with the actual callibration value
  // size()-1 b/c the inputs have lower and upper bounds (except L1EG b/c that is a cound)
  // Do Barrel, then HGCal
  //
  // Note to future developers: be very concious of the order in which the calibrations are printed
  // out in tool which makse the cfg files.  You need to match that exactly when loading them and
  // using the calibrations below.
  index = 0;
  for (unsigned int abs_eta = 0; abs_eta < tauAbsEtaBinsBarrel.size() - 1; abs_eta++) {
    std::vector<std::vector<std::vector<double>>> l1eg_bins;
    for (auto &l1eg_info : tauL1egInfoMapBarrel) {
      std::vector<std::vector<double>> em_bins;
      for (unsigned int em_frac = 0; em_frac < l1eg_info.second.size() - 1; em_frac++) {
        std::vector<double> pt_bin_calibs;
        for (unsigned int pt = 0; pt < tauPtBins.size() - 1; pt++) {
          pt_bin_calibs.push_back(tauCalibrationsBarrel.at(index));
          index++;
        }
        em_bins.push_back(pt_bin_calibs);
      }
      l1eg_bins.push_back(em_bins);
    }
    tauPtCalibrationsBarrel.push_back(l1eg_bins);
  }
  if (debug) {
    LogDebug("L1CaloJetProducer") << " Loading Barrel calibrations: Loaded " << index
                                  << " values vs. size() of input calibration file: "
                                  << int(tauCalibrationsBarrel.size()) << "\n";
  }

  index = 0;
  for (unsigned int abs_eta = 0; abs_eta < tauAbsEtaBinsHGCal.size() - 1; abs_eta++) {
    std::vector<std::vector<std::vector<double>>> l1eg_bins;
    for (const auto &l1eg_info : tauL1egInfoMapHGCal) {
      std::vector<std::vector<double>> em_bins;
      for (unsigned int em_frac = 0; em_frac < l1eg_info.second.size() - 1; em_frac++) {
        std::vector<double> pt_bin_calibs;
        for (unsigned int pt = 0; pt < tauPtBins.size() - 1; pt++) {
          pt_bin_calibs.push_back(tauCalibrationsHGCal.at(index));
          index++;
        }
        em_bins.push_back(pt_bin_calibs);
      }
      l1eg_bins.push_back(em_bins);
    }
    tauPtCalibrationsHGCal.push_back(l1eg_bins);
  }
  if (debug) {
    LogDebug("L1CaloJetProducer") << " Loading HGCal calibrations: Loaded " << index
                                  << " values vs. size() of input calibration file: "
                                  << int(tauCalibrationsHGCal.size()) << "\n";
  }

  isoTauBarrel.SetParameter(0, 0.30);
  isoTauBarrel.SetParameter(1, 0.31);
  isoTauBarrel.SetParameter(2, 0.040);
  isoTauHGCal.SetParameter(0, 0.34);
  isoTauHGCal.SetParameter(1, 0.35);
  isoTauHGCal.SetParameter(2, 0.051);
}

void L1CaloJetProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // Output collections
  std::unique_ptr<l1tp2::CaloJetsCollection> L1CaloJetsNoCuts(new l1tp2::CaloJetsCollection);
  //std::unique_ptr<l1tp2::CaloJetsCollection> L1CaloJetsWithCuts( new l1tp2::CaloJetsCollection );
  //std::unique_ptr<l1extra::L1JetParticleCollection> L1CaloClusterCollectionWithCuts( new l1extra::L1JetParticleCollection );
  std::unique_ptr<BXVector<l1t::Jet>> L1CaloJetCollectionBXV(new l1t::JetBxCollection);
  std::unique_ptr<BXVector<l1t::Tau>> L1CaloTauCollectionBXV(new l1t::TauBxCollection);

  // Load the ECAL+HCAL tower sums coming from L1EGammaCrystalsEmulatorProducer.cc
  std::vector<SimpleCaloHit> l1CaloTowers;

  iEvent.getByToken(l1TowerToken_, l1CaloTowerHandle);
  for (auto &hit : *l1CaloTowerHandle.product()) {
    SimpleCaloHit l1Hit;
    l1Hit.ecalTowerEt = hit.ecalTowerEt();
    l1Hit.hcalTowerEt = hit.hcalTowerEt();
    l1Hit.l1egTowerEt = hit.l1egTowerEt();
    // Add min ET thresholds for tower ET
    if (l1Hit.ecalTowerEt < EcalTpEtMin)
      l1Hit.ecalTowerEt = 0.0;
    if (l1Hit.hcalTowerEt < HcalTpEtMin)
      l1Hit.hcalTowerEt = 0.0;
    l1Hit.total_tower_et = l1Hit.ecalTowerEt + l1Hit.hcalTowerEt + l1Hit.l1egTowerEt;
    l1Hit.towerIEta = hit.towerIEta();
    l1Hit.towerIPhi = hit.towerIPhi();
    l1Hit.nL1eg = hit.nL1eg();
    l1Hit.l1egTrkSS = hit.l1egTrkSS();
    l1Hit.l1egTrkIso = hit.l1egTrkIso();
    l1Hit.l1egStandaloneSS = hit.l1egStandaloneSS();
    l1Hit.l1egStandaloneIso = hit.l1egStandaloneIso();
    l1Hit.isBarrel = hit.isBarrel();

    // FIXME There is an error in the L1EGammaCrystalsEmulatorProducer.cc which is
    // returning towers with minimal ECAL energy, and no HCAL energy with these
    // iEta/iPhi coordinates and eta = -88.653152 and phi = -99.000000.
    // Skip these for the time being until the upstream code has been debugged
    if ((int)l1Hit.towerIEta == -1016 && (int)l1Hit.towerIPhi == -962)
      continue;

    l1Hit.towerEta = hit.towerEta();
    l1Hit.towerPhi = hit.towerPhi();
    l1CaloTowers.push_back(l1Hit);
    if (debug) {
      LogDebug("L1CaloJetProducer") << " Tower iEta " << (int)l1Hit.towerIEta << " iPhi " << (int)l1Hit.towerIPhi
                                    << " eta " << l1Hit.towerEta << " phi " << l1Hit.towerPhi << " ecal_et "
                                    << l1Hit.ecalTowerEt << " hacl_et " << l1Hit.hcalTowerEt << " total_et "
                                    << l1Hit.total_tower_et << "\n";
    }
  }

  // Sort the ECAL+HCAL+L1EGs tower sums based on total ET
  std::sort(begin(l1CaloTowers), end(l1CaloTowers), [](const SimpleCaloHit &a, SimpleCaloHit &b) {
    return a.total_tower_et > b.total_tower_et;
  });

  /**************************************************************************
 * Begin with making CaloJets in 9x9 grid based on all energy not included in L1EG Objs.
 * For reference, Run-I used 12x12 grid and Stage-2/Phase-I used 9x9 grid.
 * We plan to further study this choice and possibly move towards a more circular shape
 * Create jetCluster within 9x9 of highest ET seed tower.
 * 9 trigger towers contains all of an ak-0.4 jets, but overshoots on the corners.
 ******************************************************************************/

  // Experimental parameters, don't want to bother with hardcoding them in data format
  std::map<std::string, float> params;

  std::vector<l1CaloJetObj> l1CaloJetObjs;

  // Count the number of unused HCAL TPs so we can stop while loop after done.
  // Clustering can also stop once there are no seed hits >= EtMinForSeedHit
  int n_towers = l1CaloTowers.size();
  int n_stale = 0;
  bool caloJetClusteringFinished = false;
  while (!caloJetClusteringFinished && n_towers != n_stale) {
    l1CaloJetObj caloJetObj;
    caloJetObj.Init();

    // First find highest ET ECAL+HCAL+L1EGs tower and use to seed the 9x9 Jet
    int cnt = 0;
    for (auto &l1CaloTower : l1CaloTowers) {
      cnt++;
      if (l1CaloTower.stale)
        continue;  // skip l1CaloTowers which are already used

      if (caloJetObj.jetClusterET == 0.0)  // this is the first l1CaloTower to seed the jet
      {
        // Check if the leading unused tower has ET < min for seeding a jet.
        // If so, stop jet clustering
        if (l1CaloTower.total_tower_et < EtMinForSeedHit) {
          caloJetClusteringFinished = true;
          continue;
        }
        l1CaloTower.stale = true;
        n_stale++;

        // Set seed location needed for delta iEta/iPhi, eta/phi comparisons later
        if (l1CaloTower.isBarrel)
          caloJetObj.barrelSeeded = true;
        else
          caloJetObj.barrelSeeded = false;

        // 3 4-vectors for ECAL, HCAL, ECAL+HCAL for adding together
        reco::Candidate::PolarLorentzVector hcalP4(
            l1CaloTower.hcalTowerEt, l1CaloTower.towerEta, l1CaloTower.towerPhi, 0.);
        reco::Candidate::PolarLorentzVector ecalP4(
            l1CaloTower.ecalTowerEt, l1CaloTower.towerEta, l1CaloTower.towerPhi, 0.);
        reco::Candidate::PolarLorentzVector l1egP4(
            l1CaloTower.l1egTowerEt, l1CaloTower.towerEta, l1CaloTower.towerPhi, 0.);
        reco::Candidate::PolarLorentzVector totalP4(
            l1CaloTower.total_tower_et, l1CaloTower.towerEta, l1CaloTower.towerPhi, 0.);

        if (hcalP4.pt() > 0) {
          caloJetObj.hcal_nHits++;
          caloJetObj.hcalJetCluster += hcalP4;
          caloJetObj.hcalJetClusterET += l1CaloTower.hcalTowerEt;
        }
        if (ecalP4.pt() > 0) {
          caloJetObj.ecal_nHits++;
          caloJetObj.ecalJetCluster += ecalP4;
          caloJetObj.ecalJetClusterET += l1CaloTower.ecalTowerEt;
        }
        if (l1egP4.pt() > 0) {
          caloJetObj.l1eg_nHits++;
          caloJetObj.l1egJetCluster += l1egP4;
          caloJetObj.l1egJetClusterET += l1CaloTower.l1egTowerEt;
          caloJetObj.l1eg_nL1EGs += l1CaloTower.nL1eg;

          caloJetObj.l1eg_nL1EGs_standaloneSS += l1CaloTower.l1egStandaloneSS;
          caloJetObj.l1eg_nL1EGs_standaloneIso += l1CaloTower.l1egStandaloneIso;
          caloJetObj.l1eg_nL1EGs_trkMatchSS += l1CaloTower.l1egTrkSS;
          caloJetObj.l1eg_nL1EGs_trkMatchIso += l1CaloTower.l1egTrkIso;

          if (l1CaloTower.isBarrel) {
            // For decay mode related checks with CaloTaus
            // only applicable in the barrel at the moment:
            // l1eg pt, HCAL ET, ECAL ET, dEta, dPhi, trkSS, trkIso, standaloneSS, standaloneIso
            std::vector<float> l1EG_info = {float(l1egP4.pt()),
                                            float(hcalP4.pt()),
                                            float(ecalP4.pt()),
                                            0.,
                                            0.,
                                            float(l1CaloTower.l1egTrkSS),
                                            float(l1CaloTower.l1egTrkIso),
                                            float(l1CaloTower.l1egStandaloneSS),
                                            float(l1CaloTower.l1egStandaloneIso)};
            if (l1EG_info[1] / (l1EG_info[0] + l1EG_info[2]) < 0.25) {
              caloJetObj.n_l1eg_HoverE_LessThreshold++;
            }
            caloJetObj.associated_l1EGs_.push_back(l1EG_info);
          }
        }
        if (totalP4.pt() > 0) {
          caloJetObj.total_nHits++;
          caloJetObj.jetCluster += totalP4;
          caloJetObj.jetClusterET += l1CaloTower.total_tower_et;
          caloJetObj.seedTower += totalP4;
          caloJetObj.seedTowerET += l1CaloTower.total_tower_et;
        }

        caloJetObj.seed_iEta = l1CaloTower.towerIEta;
        caloJetObj.seed_iPhi = l1CaloTower.towerIPhi;

        if (debug) {
          LogDebug("L1CaloJetProducer") << " -- hit " << cnt << " , seeding input     p4 pt "
                                        << l1CaloTower.total_tower_et << " eta " << l1CaloTower.towerEta << " phi "
                                        << l1CaloTower.towerPhi << "\n";
          LogDebug("L1CaloJetProducer") << " -- hit " << cnt << " , seeding input2    p4 pt " << totalP4.pt() << " eta "
                                        << totalP4.eta() << " phi " << totalP4.phi() << "\n";
          LogDebug("L1CaloJetProducer") << " -- hit " << cnt << " seeding reslt tot p4 pt " << caloJetObj.jetClusterET
                                        << " eta " << caloJetObj.jetCluster.eta() << " phi "
                                        << caloJetObj.jetCluster.phi() << "\n";
        }

        // Need to add the seed energy to the dR rings
        caloJetObj.hcal_seed += hcalP4.pt();
        //caloJetObj.hcal_3x3 += hcalP4.pt();
        caloJetObj.hcal_3x5 += hcalP4.pt();
        //caloJetObj.hcal_5x5 += hcalP4.pt();
        //caloJetObj.hcal_5x7 += hcalP4.pt();
        caloJetObj.hcal_7x7 += hcalP4.pt();
        caloJetObj.ecal_seed += ecalP4.pt();
        //caloJetObj.ecal_3x3 += ecalP4.pt();
        caloJetObj.ecal_3x5 += ecalP4.pt();
        //caloJetObj.ecal_5x5 += ecalP4.pt();
        //caloJetObj.ecal_5x7 += ecalP4.pt();
        caloJetObj.ecal_7x7 += ecalP4.pt();
        caloJetObj.l1eg_seed += l1egP4.pt();
        //caloJetObj.l1eg_3x3 += l1egP4.pt();
        caloJetObj.l1eg_3x5 += l1egP4.pt();
        //caloJetObj.l1eg_5x5 += l1egP4.pt();
        //caloJetObj.l1eg_5x7 += l1egP4.pt();
        caloJetObj.l1eg_7x7 += l1egP4.pt();
        caloJetObj.total_seed += totalP4.pt();
        //caloJetObj.total_3x3 += totalP4.pt();
        caloJetObj.total_3x5 += totalP4.pt();
        //caloJetObj.total_5x5 += totalP4.pt();
        //caloJetObj.total_5x7 += totalP4.pt();
        caloJetObj.total_7x7 += totalP4.pt();

        // Some discrimination vars, 2x2s and 2x3 including central seed
        //caloJetObj.hcal_2x3 += hcalP4.pt();
        //caloJetObj.hcal_2x3_1 += hcalP4.pt();
        //caloJetObj.hcal_2x3_2 += hcalP4.pt();
        //caloJetObj.ecal_2x3 += ecalP4.pt();
        //caloJetObj.ecal_2x3_1 += ecalP4.pt();
        //caloJetObj.ecal_2x3_2 += ecalP4.pt();
        //caloJetObj.l1eg_2x3 += l1egP4.pt();
        //caloJetObj.l1eg_2x3_1 += l1egP4.pt();
        //caloJetObj.l1eg_2x3_2 += l1egP4.pt();
        //caloJetObj.total_2x3 += totalP4.pt();
        //caloJetObj.total_2x3_1 += totalP4.pt();
        //caloJetObj.total_2x3_2 += totalP4.pt();

        //caloJetObj.hcal_2x2 += hcalP4.pt();
        //caloJetObj.hcal_2x2_1 += hcalP4.pt();
        //caloJetObj.hcal_2x2_2 += hcalP4.pt();
        //caloJetObj.hcal_2x2_3 += hcalP4.pt();
        //caloJetObj.hcal_2x2_4 += hcalP4.pt();
        //caloJetObj.ecal_2x2 += ecalP4.pt();
        //caloJetObj.ecal_2x2_1 += ecalP4.pt();
        //caloJetObj.ecal_2x2_2 += ecalP4.pt();
        //caloJetObj.ecal_2x2_3 += ecalP4.pt();
        //caloJetObj.ecal_2x2_4 += ecalP4.pt();
        //caloJetObj.l1eg_2x2 += l1egP4.pt();
        //caloJetObj.l1eg_2x2_1 += l1egP4.pt();
        //caloJetObj.l1eg_2x2_2 += l1egP4.pt();
        //caloJetObj.l1eg_2x2_3 += l1egP4.pt();
        //caloJetObj.l1eg_2x2_4 += l1egP4.pt();
        //caloJetObj.total_2x2 += totalP4.pt();
        //caloJetObj.total_2x2_1 += totalP4.pt();
        //caloJetObj.total_2x2_2 += totalP4.pt();
        //caloJetObj.total_2x2_3 += totalP4.pt();
        //caloJetObj.total_2x2_4 += totalP4.pt();
        continue;
      }

      // Unused l1CaloTowers which are not the initial seed
      // Depending on seed and tower locations calculate iEta/iPhi or eta/phi comparisons.
      // The defaults of 99 will automatically fail comparisons for the incorrect regions.
      int hit_iPhi = 99;
      int d_iEta = 99;
      int d_iPhi = 99;
      float d_eta = 99;
      float d_phi = 99;
      if (caloJetObj.barrelSeeded && l1CaloTower.isBarrel)  // use iEta/iPhi comparisons
      {
        hit_iPhi = l1CaloTower.towerIPhi;
        d_iEta = tower_diEta(caloJetObj.seed_iEta, l1CaloTower.towerIEta);
        d_iPhi = tower_diPhi(caloJetObj.seed_iPhi, hit_iPhi);
      } else  // either seed or tower are in HGCal or HF, use eta/phi
      {
        d_eta = caloJetObj.seedTower.eta() - l1CaloTower.towerEta;
        d_phi = reco::deltaPhi(caloJetObj.seedTower.phi(), l1CaloTower.towerPhi);
      }

      // 7x7 HCAL Trigger Towers
      // If seeded in barrel and hit is barrel then we can compare iEta/iPhi, else need to use eta/phi
      // in HGCal / transition region
      if ((abs(d_iEta) <= 3 && abs(d_iPhi) <= 3) || (fabs(d_eta) < 0.3 && fabs(d_phi) < 0.3)) {
        l1CaloTower.stale = true;
        n_stale++;

        // 3 4-vectors for ECAL, HCAL, ECAL+HCAL for adding together
        reco::Candidate::PolarLorentzVector hcalP4(
            l1CaloTower.hcalTowerEt, l1CaloTower.towerEta, l1CaloTower.towerPhi, 0.);
        reco::Candidate::PolarLorentzVector ecalP4(
            l1CaloTower.ecalTowerEt, l1CaloTower.towerEta, l1CaloTower.towerPhi, 0.);
        reco::Candidate::PolarLorentzVector l1egP4(
            l1CaloTower.l1egTowerEt, l1CaloTower.towerEta, l1CaloTower.towerPhi, 0.);
        reco::Candidate::PolarLorentzVector totalP4(
            l1CaloTower.total_tower_et, l1CaloTower.towerEta, l1CaloTower.towerPhi, 0.);

        if (hcalP4.pt() > 0) {
          caloJetObj.hcal_nHits++;
          caloJetObj.hcalJetCluster += hcalP4;
          caloJetObj.hcalJetClusterET += l1CaloTower.hcalTowerEt;
        }
        if (ecalP4.pt() > 0) {
          caloJetObj.ecal_nHits++;
          caloJetObj.ecalJetCluster += ecalP4;
          caloJetObj.ecalJetClusterET += l1CaloTower.ecalTowerEt;
        }
        if (l1egP4.pt() > 0) {
          caloJetObj.l1eg_nHits++;
          caloJetObj.l1egJetCluster += l1egP4;
          caloJetObj.l1egJetClusterET += l1CaloTower.l1egTowerEt;
          caloJetObj.l1eg_nL1EGs += l1CaloTower.nL1eg;
        }
        if (totalP4.pt() > 0) {
          caloJetObj.total_nHits++;
          caloJetObj.jetCluster += totalP4;
          caloJetObj.jetClusterET += l1CaloTower.total_tower_et;
        }

        if (debug) {
          LogDebug("L1CaloJetProducer") << " ---- hit " << cnt << " input     p4 pt " << totalP4.pt() << " eta "
                                        << totalP4.eta() << " phi " << totalP4.phi() << "\n";
          LogDebug("L1CaloJetProducer") << " ---- hit " << cnt << " resulting p4 pt " << caloJetObj.jetClusterET
                                        << " eta " << caloJetObj.jetCluster.eta() << " phi "
                                        << caloJetObj.jetCluster.phi() << "\n";
        }

        if ((abs(d_iEta) == 0 && abs(d_iPhi) == 0) || (fabs(d_eta) < 0.043 && fabs(d_phi) < 0.043)) {
          caloJetObj.hcal_seed += hcalP4.pt();
          caloJetObj.ecal_seed += ecalP4.pt();
          caloJetObj.l1eg_seed += l1egP4.pt();
          caloJetObj.total_seed += totalP4.pt();
        }
        //if ( (abs( d_iEta ) <= 1 && abs( d_iPhi ) <= 1) ||
        //    ( fabs( d_eta ) < 0.13 && fabs( d_phi ) < 0.13 ) )
        //{
        //    caloJetObj.hcal_3x3 += hcalP4.pt();
        //    caloJetObj.ecal_3x3 += ecalP4.pt();
        //    caloJetObj.l1eg_3x3 += l1egP4.pt();
        //    caloJetObj.total_3x3 += totalP4.pt();
        //}
        if ((abs(d_iEta) <= 1 && abs(d_iPhi) <= 2) || (fabs(d_eta) < 0.13 && fabs(d_phi) < 0.22)) {
          caloJetObj.hcal_3x5 += hcalP4.pt();
          caloJetObj.ecal_3x5 += ecalP4.pt();
          caloJetObj.l1eg_3x5 += l1egP4.pt();
          caloJetObj.total_3x5 += totalP4.pt();

          // Do this for 3x5 only
          if (l1egP4.pt() > 0) {
            caloJetObj.l1eg_nL1EGs_standaloneSS += l1CaloTower.l1egStandaloneSS;
            caloJetObj.l1eg_nL1EGs_standaloneIso += l1CaloTower.l1egStandaloneIso;
            caloJetObj.l1eg_nL1EGs_trkMatchSS += l1CaloTower.l1egTrkSS;
            caloJetObj.l1eg_nL1EGs_trkMatchIso += l1CaloTower.l1egTrkIso;

            // For decay mode related checks with CaloTaus
            // only applicable in the barrel at the moment:
            // l1eg pt, HCAL ET, ECAL ET, d_iEta, d_iPhi, trkSS, trkIso, standaloneSS, standaloneIso
            std::vector<float> l1EG_info = {float(l1egP4.pt()),
                                            float(hcalP4.pt()),
                                            float(ecalP4.pt()),
                                            float(d_iEta),
                                            float(d_iPhi),
                                            float(l1CaloTower.l1egTrkSS),
                                            float(l1CaloTower.l1egTrkIso),
                                            float(l1CaloTower.l1egStandaloneSS),
                                            float(l1CaloTower.l1egStandaloneIso)};
            if (l1EG_info[1] / (l1EG_info[0] + l1EG_info[2]) < 0.25) {
              caloJetObj.n_l1eg_HoverE_LessThreshold++;
            }
            caloJetObj.associated_l1EGs_.push_back(l1EG_info);
          }
        }
        //if ( ( abs( d_iEta ) <= 2 && abs( d_iPhi ) <= 2) ||
        //    ( fabs( d_eta ) < 0.22 && fabs( d_phi ) < 0.22 ) )
        //{
        //    caloJetObj.hcal_5x5 += hcalP4.pt();
        //    caloJetObj.ecal_5x5 += ecalP4.pt();
        //    caloJetObj.l1eg_5x5 += l1egP4.pt();
        //    caloJetObj.total_5x5 += totalP4.pt();
        //}
        //if ( ( abs( d_iEta ) <= 2 && abs( d_iPhi ) <= 3) ||
        //    ( fabs( d_eta ) < 0.22 && fabs( d_phi ) < 0.3 ) )
        //{
        //    caloJetObj.hcal_5x7 += hcalP4.pt();
        //    caloJetObj.ecal_5x7 += ecalP4.pt();
        //    caloJetObj.l1eg_5x7 += l1egP4.pt();
        //    caloJetObj.total_5x7 += totalP4.pt();
        //}
        if ((abs(d_iEta) <= 3 && abs(d_iPhi) <= 3) || (fabs(d_eta) < 0.3 && fabs(d_phi) < 0.3)) {
          caloJetObj.hcal_7x7 += hcalP4.pt();
          caloJetObj.ecal_7x7 += ecalP4.pt();
          caloJetObj.l1eg_7x7 += l1egP4.pt();
          caloJetObj.total_7x7 += totalP4.pt();
        }

        //// Some discrimination vars, 2x2s and 2x3s including central seed
        //// Barrel first, 2x3
        //if ( ( d_iEta == 0 || d_iEta == 1 ) && abs(d_iPhi) <= 1 )
        //{
        //    caloJetObj.hcal_2x3_1 += hcalP4.pt();
        //    caloJetObj.ecal_2x3_1 += ecalP4.pt();
        //    caloJetObj.l1eg_2x3_1 += l1egP4.pt();
        //    caloJetObj.total_2x3_1 += totalP4.pt();
        //}
        //if ( ( d_iEta == 0 || d_iEta == -1 ) && abs(d_iPhi) <= 1 )
        //{
        //    caloJetObj.hcal_2x3_2 += hcalP4.pt();
        //    caloJetObj.ecal_2x3_2 += ecalP4.pt();
        //    caloJetObj.l1eg_2x3_2 += l1egP4.pt();
        //    caloJetObj.total_2x3_2 += totalP4.pt();
        //}
        //// HGCal / HF
        //if ( fabs( d_eta ) < 0.087 && fabs( d_phi ) < 0.13 )
        //{
        //    caloJetObj.hcal_2x3 += hcalP4.pt();
        //    caloJetObj.ecal_2x3 += ecalP4.pt();
        //    caloJetObj.l1eg_2x3 += l1egP4.pt();
        //    caloJetObj.total_2x3 += totalP4.pt();
        //}

        //// Now 2x2 Barrel first
        //if ( ( d_iEta == 0 || d_iEta == 1 ) && ( d_iPhi == 0 || d_iPhi == 1 ) )
        //{
        //    caloJetObj.hcal_2x2_1 += hcalP4.pt();
        //    caloJetObj.ecal_2x2_1 += ecalP4.pt();
        //    caloJetObj.l1eg_2x2_1 += l1egP4.pt();
        //    caloJetObj.total_2x2_1 += totalP4.pt();
        //}
        //if ( ( d_iEta == 0 || d_iEta == 1 ) && ( d_iPhi == 0 || d_iPhi == -1 ) )
        //{
        //    caloJetObj.hcal_2x2_2 += hcalP4.pt();
        //    caloJetObj.ecal_2x2_2 += ecalP4.pt();
        //    caloJetObj.l1eg_2x2_2 += l1egP4.pt();
        //    caloJetObj.total_2x2_2 += totalP4.pt();
        //}
        //if ( ( d_iEta == 0 || d_iEta == -1 ) && ( d_iPhi == 0 || d_iPhi == 1 ) )
        //{
        //    caloJetObj.hcal_2x2_3 += hcalP4.pt();
        //    caloJetObj.ecal_2x2_3 += ecalP4.pt();
        //    caloJetObj.l1eg_2x2_3 += l1egP4.pt();
        //    caloJetObj.total_2x2_3 += totalP4.pt();
        //}
        //if ( ( d_iEta == 0 || d_iEta == -1 ) && ( d_iPhi == 0 || d_iPhi == -1 ) )
        //{
        //    caloJetObj.hcal_2x2_4 += hcalP4.pt();
        //    caloJetObj.ecal_2x2_4 += ecalP4.pt();
        //    caloJetObj.l1eg_2x2_4 += l1egP4.pt();
        //    caloJetObj.total_2x2_4 += totalP4.pt();
        //}
        //// HGCal / HF
        //if ( fabs( d_eta ) < 0.087 && fabs( d_phi ) < 0.087 )
        //{
        //    caloJetObj.hcal_2x2 += hcalP4.pt();
        //    caloJetObj.ecal_2x2 += ecalP4.pt();
        //    caloJetObj.l1eg_2x2 += l1egP4.pt();
        //    caloJetObj.total_2x2 += totalP4.pt();
        //}
      }
    }

    if (caloJetObj.jetClusterET > 0.0) {
      l1CaloJetObjs.push_back(caloJetObj);
    }

  }  // end while loop of HCAL TP clustering

  // Sort JetClusters so we can begin with the highest pt for next step of jet clustering
  std::sort(begin(l1CaloJetObjs), end(l1CaloJetObjs), [](const l1CaloJetObj &a, const l1CaloJetObj &b) {
    return a.jetClusterET > b.jetClusterET;
  });

  /**************************************************************************
 * Progress to adding L1EGs built from ECAL TPs  9x9 grid.
 * Recall, for 9x9 trigger towers gives diameter 0.78
 ******************************************************************************/

  // Cluster together the L1EGs around existing HCAL Jet
  // Cluster within dEta/dPhi 0.4 which is very close to 0.39 = 9x9/2
  //std::cout << " - Input L1EGs: " << crystalClustersVect.size() << std::endl;
  for (auto &caloJetObj : l1CaloJetObjs) {
    params["seed_pt"] = caloJetObj.seedTowerET;
    params["seed_eta"] = caloJetObj.seedTower.eta();
    params["seed_phi"] = caloJetObj.seedTower.phi();
    params["seed_iEta"] = caloJetObj.seed_iEta;
    params["seed_iPhi"] = caloJetObj.seed_iPhi;
    params["seed_energy"] = caloJetObj.seedTower.energy();

    params["hcal_pt"] = caloJetObj.hcalJetClusterET;
    params["hcal_seed"] = caloJetObj.hcal_seed;
    //params["hcal_3x3"] = caloJetObj.hcal_3x3;
    params["hcal_3x5"] = caloJetObj.hcal_3x5;
    //params["hcal_5x5"] = caloJetObj.hcal_5x5;
    //params["hcal_5x7"] = caloJetObj.hcal_5x7;
    params["hcal_7x7"] = caloJetObj.hcal_7x7;
    //params["hcal_2x3"] = std::max( caloJetObj.hcal_2x3, std::max( caloJetObj.hcal_2x3_1, caloJetObj.hcal_2x3_2 ));
    //params["hcal_2x2"] = std::max( caloJetObj.hcal_2x2, std::max( caloJetObj.hcal_2x2_1, std::max( caloJetObj.hcal_2x2_2, std::max( caloJetObj.hcal_2x2_3, caloJetObj.hcal_2x2_4 ))));
    params["hcal_nHits"] = caloJetObj.hcal_nHits;

    params["ecal_pt"] = caloJetObj.ecalJetClusterET;
    params["ecal_seed"] = caloJetObj.ecal_seed;
    //params["ecal_3x3"] = caloJetObj.ecal_3x3;
    params["ecal_3x5"] = caloJetObj.ecal_3x5;
    //params["ecal_5x5"] = caloJetObj.ecal_5x5;
    //params["ecal_5x7"] = caloJetObj.ecal_5x7;
    params["ecal_7x7"] = caloJetObj.ecal_7x7;
    //params["ecal_2x3"] = std::max( caloJetObj.ecal_2x3, std::max( caloJetObj.ecal_2x3_1, caloJetObj.ecal_2x3_2 ));
    //params["ecal_2x2"] = std::max( caloJetObj.ecal_2x2, std::max( caloJetObj.ecal_2x2_1, std::max( caloJetObj.ecal_2x2_2, std::max( caloJetObj.ecal_2x2_3, caloJetObj.ecal_2x2_4 ))));
    params["ecal_nHits"] = caloJetObj.ecal_nHits;

    params["l1eg_pt"] = caloJetObj.l1egJetClusterET;
    params["l1eg_seed"] = caloJetObj.l1eg_seed;
    //params["l1eg_3x3"] = caloJetObj.l1eg_3x3;
    params["l1eg_3x5"] = caloJetObj.l1eg_3x5;
    //params["l1eg_5x5"] = caloJetObj.l1eg_5x5;
    //params["l1eg_5x7"] = caloJetObj.l1eg_5x7;
    params["l1eg_7x7"] = caloJetObj.l1eg_7x7;
    //params["l1eg_2x3"] = std::max( caloJetObj.l1eg_2x3, std::max( caloJetObj.l1eg_2x3_1, caloJetObj.l1eg_2x3_2 ));
    //params["l1eg_2x2"] = std::max( caloJetObj.l1eg_2x2, std::max( caloJetObj.l1eg_2x2_1, std::max( caloJetObj.l1eg_2x2_2, std::max( caloJetObj.l1eg_2x2_3, caloJetObj.l1eg_2x2_4 ))));
    params["l1eg_nHits"] = caloJetObj.l1eg_nHits;
    params["l1eg_nL1EGs"] = caloJetObj.l1eg_nL1EGs;
    params["l1eg_nL1EGs_standaloneSS"] = caloJetObj.l1eg_nL1EGs_standaloneSS;
    params["l1eg_nL1EGs_standaloneIso"] = caloJetObj.l1eg_nL1EGs_standaloneIso;
    params["l1eg_nL1EGs_trkMatchSS"] = caloJetObj.l1eg_nL1EGs_trkMatchSS;
    params["l1eg_nL1EGs_trkMatchIso"] = caloJetObj.l1eg_nL1EGs_trkMatchIso;

    params["total_et"] = caloJetObj.jetClusterET;
    params["total_seed"] = caloJetObj.total_seed;
    //params["total_3x3"] = caloJetObj.total_3x3;
    params["total_3x5"] = caloJetObj.total_3x5;
    //params["total_5x5"] = caloJetObj.total_5x5;
    //params["total_5x7"] = caloJetObj.total_5x7;
    params["total_7x7"] = caloJetObj.total_7x7;
    //params["total_2x3"] = std::max( caloJetObj.total_2x3, std::max( caloJetObj.total_2x3_1, caloJetObj.total_2x3_2 ));
    //params["total_2x2"] = std::max( caloJetObj.total_2x2, std::max( caloJetObj.total_2x2_1, std::max( caloJetObj.total_2x2_2, std::max( caloJetObj.total_2x2_3, caloJetObj.total_2x2_4 ))));
    params["total_nHits"] = caloJetObj.total_nHits;
    //params["total_nTowers"] = total_nTowers;

    //// return -9 for energy and dR values for ecalJet as defaults
    float hovere = -9;
    if (caloJetObj.ecalJetClusterET > 0.0) {
      hovere = caloJetObj.hcalJetClusterET / (caloJetObj.ecalJetClusterET + caloJetObj.l1egJetClusterET);
    }

    params["jet_pt"] = caloJetObj.jetClusterET;
    params["jet_eta"] = caloJetObj.jetCluster.eta();
    params["jet_phi"] = caloJetObj.jetCluster.phi();
    params["jet_mass"] = caloJetObj.jetCluster.mass();
    params["jet_energy"] = caloJetObj.jetCluster.energy();

    // Calibrations
    params["hcal_calibration"] =
        get_hcal_calibration(params["jet_pt"], params["ecal_pt"], params["l1eg_pt"], params["jet_eta"]);
    params["hcal_pt_calibration"] = params["hcal_pt"] * params["hcal_calibration"];
    params["jet_pt_calibration"] = params["hcal_pt_calibration"] + params["ecal_pt"] + params["l1eg_pt"];

    // Tau Vars
    // The tau pT calibration is applied as a SF to the total raw pT
    // in contrast to the jet calibration above
    params["tau_pt_calibration_value"] = get_tau_pt_calibration(params["total_3x5"],
                                                                params["ecal_3x5"],
                                                                params["l1eg_3x5"],
                                                                caloJetObj.n_l1eg_HoverE_LessThreshold,
                                                                params["jet_eta"]);
    params["tau_pt"] = params["total_3x5"] * params["tau_pt_calibration_value"];
    params["n_l1eg_HoverE_LessThreshold"] = caloJetObj.n_l1eg_HoverE_LessThreshold;
    // Currently, applying the tau_pt calibration to the isolation region as well...
    // One could switch to using the calibrated jet_pt instead for the iso region...
    // This should be revisited - FIXME?
    params["tau_total_iso_et"] = params["jet_pt"] * params["tau_pt_calibration_value"];
    params["tau_iso_et"] = (params["jet_pt"] * params["tau_pt_calibration_value"]) - params["tau_pt"];
    params["loose_iso_tau_wp"] = float(loose_iso_tau_wp(params["tau_pt"], params["tau_iso_et"], params["jet_eta"]));

    float calibratedPt = -1;
    float ECalIsolation = -1;  // Need to loop over 7x7 crystals of unclustered energy
    float totalPtPUcorr = -1;
    l1tp2::CaloJet caloJet(caloJetObj.jetCluster, calibratedPt, hovere, ECalIsolation, totalPtPUcorr);
    caloJet.setExperimentalParams(params);
    caloJet.setAssociated_l1EGs(caloJetObj.associated_l1EGs_);

    // Only store jets passing ET threshold
    if (params["jet_pt_calibration"] >= EtMinForCollection) {
      L1CaloJetsNoCuts->push_back(caloJet);
      //L1CaloJetsWithCuts->push_back( caloJet );
      reco::Candidate::PolarLorentzVector jet_p4 = reco::Candidate::PolarLorentzVector(
          params["jet_pt_calibration"], caloJet.p4().eta(), caloJet.p4().phi(), caloJet.p4().M());
      L1CaloJetCollectionBXV->push_back(0, l1t::Jet(jet_p4));

      if (debug)
        LogDebug("L1CaloJetProducer") << " Made a Jet, eta " << caloJetObj.jetCluster.eta() << " phi "
                                      << caloJetObj.jetCluster.phi() << " pt " << caloJetObj.jetClusterET
                                      << " calibrated pt " << params["jet_pt_calibration"] << "\n";
    }

    // Only store taus passing ET threshold
    if (params["tau_pt"] >= EtMinForTauCollection) {
      short int tau_ieta = caloJetObj.seed_iEta;
      short int tau_iphi = caloJetObj.seed_iPhi;
      short int raw_et = params["total_3x5"];
      short int iso_et = params["tau_iso_et"];
      bool hasEM = false;
      if (params["l1eg_3x5"] > 0. || params["ecal_3x5"] > 0.) {
        hasEM = true;
      }
      int tau_qual = int(params["loose_iso_tau_wp"]);

      reco::Candidate::PolarLorentzVector tau_p4 = reco::Candidate::PolarLorentzVector(
          params["tau_pt"], caloJet.p4().eta(), caloJet.p4().phi(), caloJet.p4().M());
      l1t::Tau l1Tau = l1t::Tau(tau_p4, params["tau_pt"], caloJet.p4().eta(), caloJet.p4().phi(), tau_qual, iso_et);
      l1Tau.setTowerIEta(tau_ieta);
      l1Tau.setTowerIPhi(tau_iphi);
      l1Tau.setRawEt(raw_et);
      l1Tau.setIsoEt(iso_et);
      l1Tau.setHasEM(hasEM);
      l1Tau.setIsMerged(false);
      L1CaloTauCollectionBXV->push_back(0, l1Tau);

      if (debug) {
        LogDebug("L1CaloJetProducer") << " Made a Jet, eta " << l1Tau.eta() << " phi " << l1Tau.phi() << " pt "
                                      << l1Tau.rawEt() << " calibrated pt " << l1Tau.pt() << "\n";
      }
    }

  }  // end jetClusters loop

  iEvent.put(std::move(L1CaloJetsNoCuts), "L1CaloJetsNoCuts");
  //iEvent.put(std::move(L1CaloJetsWithCuts), "L1CaloJetsWithCuts" );
  //iEvent.put(std::move(L1CaloClusterCollectionWithCuts), "L1CaloClusterCollectionWithCuts" );
  iEvent.put(std::move(L1CaloJetCollectionBXV), "L1CaloJetCollectionBXV");
  iEvent.put(std::move(L1CaloTauCollectionBXV), "L1CaloTauCollectionBXV");
}

int L1CaloJetProducer::tower_diPhi(int &iPhi_1, int &iPhi_2) const {
  // 360 Crystals in full, 72 towers, half way is 36
  int PI = 36;
  int result = iPhi_1 - iPhi_2;
  while (result > PI)
    result -= 2 * PI;
  while (result <= -PI)
    result += 2 * PI;
  return result;
}

// Added b/c of the iEta jump from +1 to -1 across the barrel mid point
int L1CaloJetProducer::tower_diEta(int &iEta_1, int &iEta_2) const {
  // On same side of barrel
  if (iEta_1 * iEta_2 > 0)
    return iEta_1 - iEta_2;
  else
    return iEta_1 - iEta_2 - 1;
}

float L1CaloJetProducer::get_deltaR(reco::Candidate::PolarLorentzVector &p4_1,
                                    reco::Candidate::PolarLorentzVector &p4_2) const {
  // Check that pt is > 0 for both or else reco::deltaR returns bogus values
  if (p4_1.pt() > 0 && p4_2.pt() > 0) {
    return reco::deltaR(p4_1, p4_2);
  } else
    return -1;
}

// Apply calibrations to HCAL energy based on EM Fraction, Jet Eta, Jet pT
float L1CaloJetProducer::get_hcal_calibration(float &jet_pt,
                                              float &ecal_pt,
                                              float &ecal_L1EG_jet_pt,
                                              float &jet_eta) const {
  float em_frac = (ecal_L1EG_jet_pt + ecal_pt) / jet_pt;
  float abs_eta = fabs(jet_eta);
  float tmp_jet_pt = jet_pt;
  if (tmp_jet_pt > 499)
    tmp_jet_pt = 499;

  // Different indices sizes in different calo regions.
  // Barrel...
  size_t em_index = 0;
  size_t eta_index = 0;
  size_t pt_index = 0;
  float calib = 1.0;
  if (abs_eta <= 1.5) {
    // Start loop checking 2nd value
    for (unsigned int i = 1; i < emFractionBinsBarrel.size(); i++) {
      if (em_frac <= emFractionBinsBarrel.at(i))
        break;
      em_index++;
    }

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
    calib = calibrationsBarrel[eta_index][em_index][pt_index];
  }                         // end Barrel
  else if (abs_eta <= 3.0)  // HGCal
  {
    // Start loop checking 2nd value
    for (unsigned int i = 1; i < emFractionBinsHGCal.size(); i++) {
      if (em_frac <= emFractionBinsHGCal.at(i))
        break;
      em_index++;
    }

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
    calib = calibrationsHGCal[eta_index][em_index][pt_index];
  }     // end HGCal
  else  // HF
  {
    // Start loop checking 2nd value
    for (unsigned int i = 1; i < emFractionBinsHF.size(); i++) {
      if (em_frac <= emFractionBinsHF.at(i))
        break;
      em_index++;
    }

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
    calib = calibrationsHF[eta_index][em_index][pt_index];
  }  // end HF

  return calib;
}

// Apply calibrations to tau pT based on L1EG info, EM Fraction, Tau Eta, Tau pT
float L1CaloJetProducer::get_tau_pt_calibration(
    float &tau_pt, float &ecal_pt, float &l1EG_pt, float &n_L1EGs, float &tau_eta) const {
  float em_frac = (l1EG_pt + ecal_pt) / tau_pt;
  float abs_eta = fabs(tau_eta);
  float tmp_tau_pt = tau_pt;
  if (tmp_tau_pt > 199)
    tmp_tau_pt = 199;

  // Different indices sizes in different calo regions.
  // Barrel...
  size_t em_index = 0;
  size_t eta_index = 0;
  size_t n_L1EG_index = 0;
  size_t pt_index = 0;
  float calib = 1.0;
  // HERE
  if (abs_eta <= 1.5) {
    // Start loop checking 1st value
    for (unsigned int i = 0; i < tauL1egValuesBarrel.size(); i++) {
      if (n_L1EGs == tauL1egValuesBarrel.at(i))
        break;
      if (tauL1egValuesBarrel.at(i) == tauL1egValuesBarrel.back())
        break;  // to preven incrementing on last one
      n_L1EG_index++;
    }

    // Find key value pair matching n L1EGs
    for (auto &l1eg_info : tauL1egInfoMapBarrel) {
      if (l1eg_info.first != double(n_L1EG_index))
        continue;
      // Start loop checking 2nd value
      for (unsigned int i = 1; i < l1eg_info.second.size(); i++) {
        if (em_frac <= l1eg_info.second.at(i))
          break;
        em_index++;
      }
    }

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
    calib = tauPtCalibrationsBarrel[eta_index][n_L1EG_index][em_index][pt_index];
  }                         // end Barrel
  else if (abs_eta <= 3.0)  // HGCal
  {
    // Start loop checking 1st value
    for (unsigned int i = 0; i < tauL1egValuesHGCal.size(); i++) {
      if (n_L1EGs == tauL1egValuesHGCal.at(i))
        break;
      if (tauL1egValuesHGCal.at(i) == tauL1egValuesHGCal.back())
        break;  // to preven incrementing on last one
      n_L1EG_index++;
    }

    // Find key value pair matching n L1EGs
    for (auto &l1eg_info : tauL1egInfoMapHGCal) {
      if (l1eg_info.first != double(n_L1EG_index))
        continue;
      // Start loop checking 2nd value
      for (unsigned int i = 1; i < l1eg_info.second.size(); i++) {
        if (em_frac <= l1eg_info.second.at(i))
          break;
        em_index++;
      }
    }

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
    calib = tauPtCalibrationsHGCal[eta_index][n_L1EG_index][em_index][pt_index];
  }  // end HGCal
  else
    return calib;

  return calib;
}

// Loose IsoTau WP
int L1CaloJetProducer::loose_iso_tau_wp(float &tau_pt, float &tau_iso_et, float &tau_eta) const {
  // Fully relaxed above 100 GeV pT
  if (tau_pt > 100) {
    return 1;
  }
  // Split by barrel and HGCal
  // with Barrel first
  if (fabs(tau_eta) < 1.5) {
    if (isoTauBarrel.Eval(tau_pt) >= (tau_iso_et / tau_pt)) {
      return 1;
    } else {
      return 0;
    }
  }
  // HGCal
  if (fabs(tau_eta) < 3.0) {
    if (isoTauHGCal.Eval(tau_pt) >= (tau_iso_et / tau_pt)) {
      return 1;
    } else {
      return 0;
    }
  }
  // Beyond HGCal
  return 0;
}

DEFINE_FWK_MODULE(L1CaloJetProducer);
