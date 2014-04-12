//*****************************************************************************
// File:      EgammaElectronTkNumIsolationProducer.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************


// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"


#include "RecoEgamma/EgammaIsolationAlgos/plugins/EgammaElectronTkNumIsolationProducer.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/ElectronTkIsolation.h"

EgammaElectronTkNumIsolationProducer::EgammaElectronTkNumIsolationProducer(const edm::ParameterSet& config) : conf_(config)
{
  // use configuration file to setup input/output collection names
  electronProducer_               = conf_.getParameter<edm::InputTag>("electronProducer");
  
  trackProducer_           = conf_.getParameter<edm::InputTag>("trackProducer");
  beamspotProducer_        = conf_.getParameter<edm::InputTag>("BeamspotProducer");

  ptMin_                = conf_.getParameter<double>("ptMin");
  intRadiusBarrel_      = conf_.getParameter<double>("intRadiusBarrel");
  intRadiusEndcap_      = conf_.getParameter<double>("intRadiusEndcap");
  stripBarrel_          = conf_.getParameter<double>("stripBarrel");
  stripEndcap_          = conf_.getParameter<double>("stripEndcap");
  extRadius_            = conf_.getParameter<double>("extRadius");
  maxVtxDist_           = conf_.getParameter<double>("maxVtxDist");
  drb_                  = conf_.getParameter<double>("maxVtxDistXY");

  //register your products
  produces < edm::ValueMap<int> >();

}

EgammaElectronTkNumIsolationProducer::~EgammaElectronTkNumIsolationProducer(){}


								       
void EgammaElectronTkNumIsolationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // Get the  filtered objects
  edm::Handle< reco::GsfElectronCollection> electronHandle;
  iEvent.getByLabel(electronProducer_,electronHandle);
  
  //get the tracks
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByLabel(trackProducer_,tracks);
  const reco::TrackCollection* trackCollection = tracks.product();

  //prepare product
  std::auto_ptr<edm::ValueMap<int> > isoMap(new edm::ValueMap<int>());
  edm::ValueMap<int>::Filler filler(*isoMap);
  std::vector<int> retV(electronHandle->size(),0);

  //get beamspot 
  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByLabel(beamspotProducer_,beamSpotH);
  reco::TrackBase::Point beamspot = beamSpotH->position();
  
  ElectronTkIsolation myTkIsolation (extRadius_,intRadiusBarrel_,intRadiusEndcap_,stripBarrel_,stripEndcap_,ptMin_,maxVtxDist_,drb_,trackCollection,beamspot) ;
  
  for(unsigned int i = 0 ; i < electronHandle->size(); ++i ){
    int isoValue = myTkIsolation.getNumberTracks(&(electronHandle->at(i)));
    retV[i] = isoValue;
  }

  //fill and insert valuemap
  filler.insert(electronHandle,retV.begin(),retV.end());
  filler.fill();
  iEvent.put(isoMap);
}
