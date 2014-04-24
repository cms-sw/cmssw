#include "DQM/PhysicsHWW/interface/PFMETMaker.h"

PFMETMaker::PFMETMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector iCollector){

  PFMET_ = iCollector.consumes<edm::View<reco::PFMET> >(iConfig.getParameter<edm::InputTag>("pfmetInputTag"));

}

void PFMETMaker::SetVars(HWW& hww, const edm::Event& iEvent, const edm::EventSetup& iSetup) {

    hww.Load_evt_pfmet();
    hww.Load_evt_pfmetPhi();

    bool validToken;
  
    edm::Handle<edm::View<reco::PFMET> > met_h;
    validToken = iEvent.getByToken(PFMET_, met_h);
    if(!validToken) return; 

    hww.evt_pfmet()    = ( met_h->front() ).et();
    hww.evt_pfmetPhi() = ( met_h->front() ).phi();
}
