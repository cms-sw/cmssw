//
// Original Author:  Marco Zanetti
//         Created:  Tue Sep  9 15:56:24 CEST 2008



// user include files
#include "CalibMuon/DTCalibration/interface/DTCalibMuonSelection.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/MuonReco/interface/Muon.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

using namespace edm;
using namespace reco;

DTCalibMuonSelection::DTCalibMuonSelection(const edm::ParameterSet& iConfig)
{
  muonList = consumes<MuonCollection>(iConfig.getParameter<edm::InputTag>("src"));
  etaMin = iConfig.getParameter<double>("etaMin");
  etaMax = iConfig.getParameter<double>("etaMax");
  ptMin = iConfig.getParameter<double>("ptMin");
}


DTCalibMuonSelection::~DTCalibMuonSelection() { }


bool DTCalibMuonSelection::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  bool result = false;

  //Retrieve the muons list
  Handle<MuonCollection> MuHandle;
  iEvent.getByToken(muonList,MuHandle);

  for (MuonCollection::const_iterator nmuon = MuHandle->begin(); nmuon != MuHandle->end(); ++nmuon) {

    double ptMuon(0.);
    double etaMuon(-999.);

    if(nmuon->isGlobalMuon()){
      ptMuon = nmuon->globalTrack()->pt();
      etaMuon = nmuon->globalTrack()->eta();
    }
    else continue;

    if(ptMuon > ptMin &&etaMuon > etaMin && etaMuon < etaMax){
      result = true;
      break;
    }

  }

  return result;
}



// ------------ method called once each job just before starting event loop  ------------
void  DTCalibMuonSelection::beginJob() {
}



// ------------ method called once each job just after ending the event loop  ------------
void  DTCalibMuonSelection::endJob() {
}

