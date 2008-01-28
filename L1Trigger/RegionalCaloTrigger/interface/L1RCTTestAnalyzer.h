// -*- C++ -*-
//
// Package:    L1RCTTestAnalyzer
// Class:      L1RCTTestAnalyzer
//
/**\class L1RCTTestAnalyzer L1RCTTestAnalyzer.cc src/L1RCTTestAnalyzer/src/L1RCTTestAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  pts/47
//         Created:  Thu Jul 13 21:38:08 CEST 2006
// $Id: L1RCTTestAnalyzer.h,v 1.5 2007/03/26 15:05:29 jleonard Exp $
//
//


// system include files
#include <memory>
#include <iostream>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "TH1F.h"

//
// class declaration
//

class L1RCTTestAnalyzer : public edm::EDAnalyzer {
public:
  explicit L1RCTTestAnalyzer(const edm::ParameterSet&);
  ~L1RCTTestAnalyzer();
  
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
private:
  // ----------member data ---------------------------
  std::string m_HepMCProduct;
  bool showEmCands;
  bool showRegionSums;
  TH1F * h_emRank;
  TH1F * h_regionSum;
  TH1F * h_emIeta;
  TH1F * h_regionMip;
  TH1F * h_towerMip;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//
