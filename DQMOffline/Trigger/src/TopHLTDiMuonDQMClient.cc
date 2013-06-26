/*
 *  $Date: 2010/08/13 09:11:38 $
 *  $Revision: 1.1 $
 *  \author M. Marienfeld - DESY Hamburg
 */

#include "DQMOffline/Trigger/interface/TopHLTDiMuonDQMClient.h"

using namespace std;
using namespace edm;


TopHLTDiMuonDQMClient::TopHLTDiMuonDQMClient( const edm::ParameterSet& ps ) {

  monitorName_ = ps.getParameter<string>("monitorName");

}


TopHLTDiMuonDQMClient::~TopHLTDiMuonDQMClient() {

}


void TopHLTDiMuonDQMClient::beginJob() {

}


void TopHLTDiMuonDQMClient::beginRun(const edm::Run& r, const edm::EventSetup& context) {

}


void TopHLTDiMuonDQMClient::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) {

}


void TopHLTDiMuonDQMClient::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

}


void TopHLTDiMuonDQMClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) {

}


void TopHLTDiMuonDQMClient::endRun(const edm::Run& r, const edm::EventSetup& context) {

  dbe_ = Service<DQMStore>().operator->();

  TriggerEfficiencies_sig  = dbe_->get(monitorName_+"TriggerEfficiencies_sig");
  TriggerEfficiencies_trig = dbe_->get(monitorName_+"TriggerEfficiencies_trig");
  TriggerEfficiencies      = dbe_->get(monitorName_+"TriggerEfficiencies");

  TriggerEfficiencies_sig->getTH1F()->Sumw2();
  TriggerEfficiencies_trig->getTH1F()->Sumw2();

  TriggerEfficiencies->getTH1F()->Divide(TriggerEfficiencies_sig->getTH1F(),TriggerEfficiencies_trig->getTH1F(),1.,1.,"B");

  MuonEfficiency_pT_sig  = dbe_->get(monitorName_+"MuonEfficiency_pT_sig");
  MuonEfficiency_pT_trig = dbe_->get(monitorName_+"MuonEfficiency_pT_trig");
  MuonEfficiency_pT      = dbe_->get(monitorName_+"MuonEfficiency_pT");

  MuonEfficiency_pT_sig->getTH1F()->Sumw2();
  MuonEfficiency_pT_trig->getTH1F()->Sumw2();

  MuonEfficiency_pT->getTH1F()->Divide(MuonEfficiency_pT_sig->getTH1F(),MuonEfficiency_pT_trig->getTH1F(),1.,1.,"B");

  MuonEfficiency_pT_LOGX_sig  = dbe_->get(monitorName_+"MuonEfficiency_pT_LOGX_sig");
  MuonEfficiency_pT_LOGX_trig = dbe_->get(monitorName_+"MuonEfficiency_pT_LOGX_trig");
  MuonEfficiency_pT_LOGX      = dbe_->get(monitorName_+"MuonEfficiency_pT_LOGX");

  MuonEfficiency_pT_LOGX_sig->getTH1F()->Sumw2();
  MuonEfficiency_pT_LOGX_trig->getTH1F()->Sumw2();

  MuonEfficiency_pT_LOGX->getTH1F()->Divide(MuonEfficiency_pT_LOGX_sig->getTH1F(),MuonEfficiency_pT_LOGX_trig->getTH1F(),1.,1.,"B");

  MuonEfficiency_eta_sig  = dbe_->get(monitorName_+"MuonEfficiency_eta_sig");
  MuonEfficiency_eta_trig = dbe_->get(monitorName_+"MuonEfficiency_eta_trig");
  MuonEfficiency_eta      = dbe_->get(monitorName_+"MuonEfficiency_eta");

  MuonEfficiency_eta_sig->getTH1F()->Sumw2();
  MuonEfficiency_eta_trig->getTH1F()->Sumw2();

  MuonEfficiency_eta->getTH1F()->Divide(MuonEfficiency_eta_sig->getTH1F(),MuonEfficiency_eta_trig->getTH1F(),1.,1.,"B");

  MuonEfficiency_phi_sig  = dbe_->get(monitorName_+"MuonEfficiency_phi_sig");
  MuonEfficiency_phi_trig = dbe_->get(monitorName_+"MuonEfficiency_phi_trig");
  MuonEfficiency_phi      = dbe_->get(monitorName_+"MuonEfficiency_phi");

  MuonEfficiency_phi_sig->getTH1F()->Sumw2();
  MuonEfficiency_phi_trig->getTH1F()->Sumw2();

  MuonEfficiency_phi->getTH1F()->Divide(MuonEfficiency_phi_sig->getTH1F(),MuonEfficiency_phi_trig->getTH1F(),1.,1.,"B");

}


void TopHLTDiMuonDQMClient::endJob() {

}
