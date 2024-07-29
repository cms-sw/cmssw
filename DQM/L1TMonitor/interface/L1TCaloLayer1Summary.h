// -*- C++ -*-
//
// Package:    L1TMonitor/L1TCaloLayer1Summary
// Class:      L1TCaloLayer1Summary
//
/**\class L1TCaloLayer1Summary L1TCaloLayer1Summary.cc Demo/L1TCaloLayer1Summary/plugins/L1TCaloLayer1Summary.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Max Zhao <max.zhao@princeton.edu>
//         Created:  Wed, 10 Jul 2024 08:52:01 GMT
//
//

// system include files
#include <memory>
#include <string>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/L1CaloTrigger/interface/CICADA.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "EventFilter/L1TXRawToDigi/interface/UCTDAQRawData.h"
#include "EventFilter/L1TXRawToDigi/interface/UCTAMCRawData.h"

class L1TCaloLayer1Summary : public DQMEDAnalyzer {
public:
  explicit L1TCaloLayer1Summary(const edm::ParameterSet&);
  ~L1TCaloLayer1Summary() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  // void endJob() override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<l1t::CICADABxCollection> caloLayer1CICADAScoreToken_;
  edm::EDGetTokenT<l1t::CICADABxCollection> gtCICADAScoreToken_;
  edm::EDGetTokenT<l1t::CICADABxCollection> simCICADAScoreToken_;
  edm::EDGetTokenT<L1CaloRegionCollection> caloLayer1RegionsToken_;
  edm::EDGetTokenT<L1CaloRegionCollection> simRegionsToken_;
  edm::EDGetTokenT<FEDRawDataCollection> fedRawData_;

  dqm::reco::MonitorElement* histoCaloLayer1CICADAScore;
  dqm::reco::MonitorElement* histoGtCICADAScore;
  dqm::reco::MonitorElement* histoSimCICADAScore;
  dqm::reco::MonitorElement* histoCaloMinusSim;
  dqm::reco::MonitorElement* histoCaloMinusGt;
  dqm::reco::MonitorElement* histoSlot7MinusDaqBxid;
  dqm::reco::MonitorElement* histoCaloRegions;
  dqm::reco::MonitorElement* histoSimRegions;
  dqm::reco::MonitorElement* histoCaloMinusSimRegions;

  // dqm::reco::MonitorElement* bigVectors[30];
  // dqm::reco::MonitorElement* bigVectorsMinus[30];

  std::string histFolder_;
};