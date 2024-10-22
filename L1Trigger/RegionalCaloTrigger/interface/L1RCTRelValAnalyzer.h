// -*- C++ -*-
//
// Package:    L1RCTRelValAnalyzer
// Class:      L1RCTRelValAnalyzer
//
/**\class L1RCTRelValAnalyzer
 L1Trigger/RegionalCaloTrigger/interface/L1RCTRelValAnalyzer.h
 L1Trigger/RegionalCaloTrigger/plugins/L1RCTRelValAnalyzer.cc

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

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "TH1F.h"
#include "TH2F.h"

//
// class declaration
//

class L1RCTRelValAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit L1RCTRelValAnalyzer(const edm::ParameterSet &);
  ~L1RCTRelValAnalyzer() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  // ----------member data ---------------------------
  edm::EDGetTokenT<L1CaloEmCollection> m_rctEmCands;
  edm::EDGetTokenT<L1CaloRegionCollection> m_rctRegions;

  TH1F *h_emRank;
  TH1F *h_emIeta;
  TH1F *h_emIphi;
  TH2F *h_emIsoOccIetaIphi;
  TH2F *h_emNonIsoOccIetaIphi;

  TH1F *h_regionSum;
  TH2F *h_regionSumIetaIphi;
  TH2F *h_regionOccIetaIphi;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//
