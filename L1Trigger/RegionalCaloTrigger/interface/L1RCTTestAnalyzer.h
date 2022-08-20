// -*- C++ -*-
//
// Package:    L1RCTTestAnalyzer
// Class:      L1RCTTestAnalyzer
//
/**\class L1RCTTestAnalyzer L1RCTTestAnalyzer.cc
 src/L1RCTTestAnalyzer/src/L1RCTTestAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  pts/47
//         Created:  Thu Jul 13 21:38:08 CEST 2006
//
//

// system include files
#include <iostream>
#include <memory>
// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TTree.h"

// // outside of class
// bool compareEmCands(const L1CaloEmCand& cand1, const L1CaloEmCand& cand2)
// {
//   return (cand1.rank() < cand2.rank());
// }

//
// class declaration
//

class L1RCTTestAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit L1RCTTestAnalyzer(const edm::ParameterSet &);
  ~L1RCTTestAnalyzer() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  static bool compareEmCands(const L1CaloEmCand &cand1, const L1CaloEmCand &cand2);

  // ----------member data ---------------------------
  std::string m_HepMCProduct;
  bool showEmCands;
  bool showRegionSums;
  edm::InputTag ecalDigisLabel;
  edm::InputTag hcalDigisLabel;
  edm::InputTag rctDigisLabel;

  TTree *emTree;
  //   float emRank[8];
  //   float emIeta[8];
  //   float emIphi[8];
  //   float emIso[8];
  std::vector<int> emRank;
  std::vector<int> emIeta;
  std::vector<int> emIphi;
  std::vector<int> emIso;

  TH1F *h_emRank;
  TH1F *h_emRankOutOfTime;
  TH1F *h_emIeta;
  TH1F *h_emIphi;
  TH1F *h_emIso;
  TH2F *h_emRankInIetaIphi;
  // add isolated/non-iso?
  TH2F *h_emIsoInIetaIphi;
  TH2F *h_emNonIsoInIetaIphi;
  TH1F *h_emCandTimeSample;

  TH1F *h_regionSum;
  TH1F *h_regionIeta;
  TH1F *h_regionIphi;
  TH1F *h_regionMip;
  TH2F *h_regionSumInIetaIphi;
  // add bits in ieta/iphi?  tau, overflow, mip, quiet, finegrain?
  // (is fine grain same thing as mip??)
  TH2F *h_regionFGInIetaIphi;

  TH1F *h_towerMip;

  TH1F *h_ecalTimeSample;
  TH1F *h_hcalTimeSample;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//
