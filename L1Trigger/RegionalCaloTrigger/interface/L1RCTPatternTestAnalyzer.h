// -*- C++ -*-
//
// Package:    L1RCTPatternTestAnalyzer
// Class:      L1RCTPatternTestAnalyzer
//
/**\class L1RCTPatternTestAnalyzer L1RCTPatternTestAnalyzer.cc src/L1RCTPatternTestAnalyzer/src/L1RCTPatternTestAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  pts/47
//         Created:  Thu Jul 13 21:38:08 CEST 2006
// $Id: L1RCTPatternTestAnalyzer.h,v 1.10 2008/05/02 16:53:01 jleonard Exp $
//
//


// system include files
#include <memory>
#include <iostream>
#include <fstream>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "TH1F.h"
#include "TH2F.h"

//
// class declaration
//

class L1RCTPatternTestAnalyzer : public edm::EDAnalyzer {
public:
  explicit L1RCTPatternTestAnalyzer(const edm::ParameterSet&);
  ~L1RCTPatternTestAnalyzer();
  
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
private:
  // ----------member data ---------------------------
  std::string m_HepMCProduct;
  bool showEmCands;
  bool showRegionSums;
  bool limitTo64;  
  std::string testName;
  edm::InputTag ecalDigisLabel;
  edm::InputTag hcalDigisLabel;
  edm::InputTag rctDigisLabel;
  std::ofstream ofs;
  std::string fileName;
//   TH1F * h_emRank;
//   TH1F * h_emRankOutOfTime;
//   TH1F * h_emIeta;
//   TH1F * h_emIphi;
//   TH1F * h_emIso;
//   TH2F * h_emRankInIetaIphi;
//   // add isolated/non-iso?
//   TH2F * h_emIsoInIetaIphi;
//   TH2F * h_emNonIsoInIetaIphi;
//   TH1F * h_emCandTimeSample;

//   TH1F * h_regionSum;
//   TH1F * h_regionIeta;
//   TH1F * h_regionIphi;
//   TH1F * h_regionMip;
//   TH2F * h_regionSumInIetaIphi;
//   // add bits in ieta/iphi?  tau, overflow, mip, quiet, finegrain? 
//   // (is fine grain same thing as mip??)
//   TH2F * h_regionFGInIetaIphi;

//   TH1F * h_towerMip;

//   TH1F * h_ecalTimeSample;
//   TH1F * h_hcalTimeSample;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//
