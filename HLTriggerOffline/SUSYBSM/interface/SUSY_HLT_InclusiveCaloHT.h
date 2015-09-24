#ifndef SUSY_HLT_InclusiveCaloHT_H
#define SUSY_HLT_InclusiveCaloHT_H

//event
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// MET
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

// Jets
#include "DataFormats/JetReco/interface/CaloJet.h"

// Trigger
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"


class SUSY_HLT_InclusiveCaloHT: public DQMEDAnalyzer{

  public:
  SUSY_HLT_InclusiveCaloHT(const edm::ParameterSet& ps);
  virtual ~SUSY_HLT_InclusiveCaloHT();

  protected:
  void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup);
  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup) ;
  void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup);
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  private:
  //histos booking function
  void bookHistos(DQMStore::IBooker &);
  
  //variables from config file
  edm::EDGetTokenT<reco::CaloMETCollection> theCaloMETCollection_;
  edm::EDGetTokenT<reco::CaloJetCollection> theCaloJetCollection_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResults_;
  edm::EDGetTokenT<trigger::TriggerEvent> theTrigSummary_;

  std::string triggerPath_;
  std::string triggerPathAuxiliaryForHadronic_;
  edm::InputTag triggerFilter_;
  double ptThrJet_;
  double etaThrJet_;
  
  // Histograms
  MonitorElement* h_caloMet;
  MonitorElement* h_caloHT;
  MonitorElement* h_caloJetPt;
  MonitorElement* h_caloJetEta;
  MonitorElement* h_caloJetPhi;
  //MonitorElement* h_triggerJetPt;
  //MonitorElement* h_triggerJetEta;
  //MonitorElement* h_triggerJetPhi;
  //MonitorElement* h_triggerMetPt;
  //MonitorElement* h_triggerMetPhi;
  //MonitorElement* h_triggerHT;
  MonitorElement* h_caloMetTurnOn_num;
  MonitorElement* h_caloMetTurnOn_den;
  MonitorElement* h_caloHTTurnOn_num;
  MonitorElement* h_caloHTTurnOn_den;

};

#endif
