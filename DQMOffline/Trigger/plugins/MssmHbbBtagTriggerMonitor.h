#ifndef DQMOffline_Trigger_MssmHbbBtagTriggerMonitor_H
#define DQMOffline_Trigger_MssmHbbBtagTriggerMonitor_H

/*
  MssmHbbBtagTriggerMonitor DQM code
*/  
//
// Originally created by:  Roberval Walsh
//                         June 2017

#include <string>
#include <vector>
#include <map>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/BTauReco/interface/JetTag.h"


//DataFormats

class GenericTriggerEventFlag;

//
// class declaration
//

class MssmHbbBtagTriggerMonitor : public DQMEDAnalyzer 
{
public:
  MssmHbbBtagTriggerMonitor( const edm::ParameterSet& );
  ~MssmHbbBtagTriggerMonitor();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription & pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription & pset);

protected:


  virtual void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  virtual void dqmBeginRun(edm::Run const& run, edm::EventSetup const& iSetup) override;
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:

  std::string folderName_;
  std::string processname_;
  std::string pathname_;
  std::string triggerobjbtag_;

  edm::InputTag triggerSummaryLabel_;
  edm::InputTag triggerResultsLabel_;
  
  edm::EDGetTokenT<reco::JetTagCollection> offlineCSVPFToken_;
  edm::EDGetTokenT <edm::TriggerResults> triggerResultsToken_;
  edm::EDGetTokenT <trigger::TriggerEvent> triggerSummaryToken_;

  MonitorElement * pt_jet1_;
  MonitorElement * pt_jet2_;

  MonitorElement * pt_probe_;
  MonitorElement * pt_probe_match_;

    
  MonitorElement * discr_offline_btagcsv_jet1_;
  MonitorElement * discr_offline_btagcsv_jet2_;
  
  
  HLTConfigProvider hltConfig_;


};

#endif // DQMOffline_Trigger_MssmHbbBtagTriggerMonitor_H
