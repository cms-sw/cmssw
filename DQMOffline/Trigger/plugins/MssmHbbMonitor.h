#ifndef DQMOffline_Trigger_MssmHbbMonitor_H
#define DQMOffline_Trigger_MssmHbbMonitor_H

/*
  MssmHbbMonitor DQM code
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
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

//DataFormats

struct Binning {
  int nbins;
  double xmin;
  double xmax;
};


//
// class declaration
//

class MssmHbbMonitor : public DQMEDAnalyzer 
{
public:
  MssmHbbMonitor( const edm::ParameterSet& );
  ~MssmHbbMonitor();

protected:


  virtual void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  virtual void dqmBeginRun(edm::Run const& run, edm::EventSetup const& iSetup) override;
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:

  static Binning getHistoPSet  (edm::ParameterSet pset); 

  std::string folderName_;
  std::string processname_;
  std::string pathname_;
  
  double jetPtmin_;
  double jetEtamax_;
  
  Binning jetPtbins_;
  Binning jetEtabins_;
  Binning jetPhibins_;
  Binning jetDRbins_;
  Binning jetBtagbins_;

  Binning muonPtbins_;
  Binning muonEtabins_;
  Binning muonPhibins_;
  
  edm::EDGetTokenT<reco::JetTagCollection> offlineBtagToken_;
  edm::EDGetTokenT<reco::MuonCollection> muonsToken_;
  edm::EDGetTokenT <edm::TriggerResults> triggerResultsToken_;

  MonitorElement * pt_jet1_;
  MonitorElement * pt_jet2_;
  MonitorElement * eta_jet1_;
  MonitorElement * eta_jet2_;
  MonitorElement * phi_jet1_;
  MonitorElement * phi_jet2_;

  MonitorElement * eta_phi_jet1_;
  MonitorElement * eta_phi_jet2_;
  
  
  MonitorElement * discr_offline_btag_jet1_;
  MonitorElement * discr_offline_btag_jet2_;
  
  MonitorElement * deta_jet12_;
  MonitorElement * dphi_jet12_;
  MonitorElement * dr_jet12_;
  
  MonitorElement * pt_muon_;
  MonitorElement * eta_muon_;
  MonitorElement * phi_muon_;
  
  HLTConfigProvider hltConfig_;


};

#endif // DQMOffline_Trigger_MssmHbbMonitor_H
