#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "Calibration/Tools/plugins/ElectronSqPtTkIsolationProducer.h"
#include "Calibration/Tools/plugins/ElectronSqPtTkIsolation.h"

ElectronSqPtTkIsolationProducer::ElectronSqPtTkIsolationProducer(const edm::ParameterSet& config) : conf_(config)
{
  // use configuration file to setup input/output collection names
  electronProducer_               = conf_.getParameter<edm::InputTag>("electronProducer");
  
  trackProducer_           = conf_.getParameter<edm::InputTag>("trackProducer");

  ptMin_                = conf_.getParameter<double>("ptMin");
  intRadius_            = conf_.getParameter<double>("intRadius");
  extRadius_            = conf_.getParameter<double>("extRadius");
  maxVtxDist_           = conf_.getParameter<double>("maxVtxDist");

  absolut_              = conf_.getParameter<bool>("absolut");

  //register your products
  produces < reco::CandViewDoubleAssociations>();

}

ElectronSqPtTkIsolationProducer::~ElectronSqPtTkIsolationProducer(){}


								       
void ElectronSqPtTkIsolationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // Get the  filtered objects
  edm::Handle< reco::GsfElectronCollection> electronHandle;
  iEvent.getByLabel(electronProducer_,electronHandle);
  
  //get the tracks
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByLabel(trackProducer_,tracks);
  const reco::TrackCollection* trackCollection = tracks.product();
  
  reco::CandViewDoubleAssociations* isoMap = new reco::CandViewDoubleAssociations(reco::CandidateBaseRefProd(reco::GsfElectronRefProd( electronHandle )));
  
  ElectronSqPtTkIsolation myTkIsolation (extRadius_,intRadius_,ptMin_,maxVtxDist_,trackCollection) ;
  
  for(unsigned int i = 0 ; i < electronHandle->size(); ++i ){
    double isoValue = myTkIsolation.getPtTracks(&(electronHandle->at(i)));
    if(absolut_==true){
      isoMap->setValue(i,isoValue);
    }
    else{
      reco::SuperClusterRef sc = (electronHandle->at(i)).superCluster();
      double et = sc.get()->energy()*sin(2*atan(exp(-sc.get()->eta())));
      isoMap->setValue(i,isoValue/et);
    }
  }
  
  std::auto_ptr<reco::CandViewDoubleAssociations> isolMap(isoMap);
  iEvent.put(isolMap);
}
