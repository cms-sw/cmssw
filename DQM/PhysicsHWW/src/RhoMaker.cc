#include "DQM/PhysicsHWW/interface/RhoMaker.h"

RhoMaker::RhoMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector iCollector){

  Rho_         = iCollector.consumes<double>(iConfig.getParameter<edm::InputTag>("rhoInputTag"));
  wwRho_       = iCollector.consumes<double>(iConfig.getParameter<edm::InputTag>("wwrhoInputTag"));
  wwRhoVor_    = iCollector.consumes<double>(iConfig.getParameter<edm::InputTag>("wwrhovorInputTag"));
  RhoForEGIso_ = iCollector.consumes<double>(iConfig.getParameter<edm::InputTag>("forEGIsoInputTag"));

}

void RhoMaker::SetVars(HWW& hww, const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  hww.Load_evt_rho();
  hww.Load_evt_ww_rho();
  hww.Load_evt_ww_rho_vor();
  hww.Load_evt_kt6pf_foregiso_rho();

  edm::Handle<double> rhoH;
  edm::Handle<double> ww_rhoH;
  edm::Handle<double> ww_rho_vorH;
  edm::Handle<double> kt6pf_foregiso_rhoH;

  bool validToken;

  validToken = iEvent.getByToken( Rho_, rhoH);
  if(!validToken) return;
  hww.evt_rho() = *rhoH;

  validToken = iEvent.getByToken( wwRho_, ww_rhoH);
  if(!validToken) return;
  hww.evt_ww_rho() = *ww_rhoH;

  validToken = iEvent.getByToken( wwRhoVor_ , ww_rho_vorH);
  if(!validToken) return;
  hww.evt_ww_rho_vor() = *ww_rho_vorH;

  validToken = iEvent.getByToken( RhoForEGIso_ , kt6pf_foregiso_rhoH);
  if(!validToken) return;
  hww.evt_kt6pf_foregiso_rho() = *kt6pf_foregiso_rhoH;
}
