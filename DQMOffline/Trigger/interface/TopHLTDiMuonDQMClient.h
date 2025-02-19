#ifndef TopHLTDiMuonDQMClient_H
#define TopHLTDiMuonDQMClient_H

/*
 *  $Date: 2010/08/13 09:12:05 $
 *  $Revision: 1.1 $
 *  \author M. Marienfeld - DESY Hamburg
 */

#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class TopHLTDiMuonDQMClient : public edm::EDAnalyzer {

 public:

  TopHLTDiMuonDQMClient( const edm::ParameterSet& );
  ~TopHLTDiMuonDQMClient();

 protected:   

  void beginJob();
  void beginRun(const edm::Run&, const edm::EventSetup&);
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);

  void analyze(const edm::Event&, const edm::EventSetup&);

  void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  void endRun(const edm::Run&, const edm::EventSetup&);
  void endJob();

 private:

  DQMStore * dbe_;
  std::string monitorName_;

  MonitorElement * TriggerEfficiencies;
  MonitorElement * TriggerEfficiencies_sig;
  MonitorElement * TriggerEfficiencies_trig;

  MonitorElement * MuonEfficiency_pT;
  MonitorElement * MuonEfficiency_pT_sig;
  MonitorElement * MuonEfficiency_pT_trig;

  MonitorElement * MuonEfficiency_pT_LOGX;
  MonitorElement * MuonEfficiency_pT_LOGX_sig;
  MonitorElement * MuonEfficiency_pT_LOGX_trig;

  MonitorElement * MuonEfficiency_eta;
  MonitorElement * MuonEfficiency_eta_sig;
  MonitorElement * MuonEfficiency_eta_trig;

  MonitorElement * MuonEfficiency_phi;
  MonitorElement * MuonEfficiency_phi_sig;
  MonitorElement * MuonEfficiency_phi_trig;

};

#endif
