/** \class EgammaHLTElectronTrackIsolationProducers
 *
 *  \author Monica Vazquez Acosta (CERN)
 * 
 * $Id: EgammaHLTElectronTrackIsolationProducers.cc,v 1.10 2012/01/23 12:56:38 sharper Exp $
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTElectronTrackIsolationProducers.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"


#include "RecoEgamma/EgammaIsolationAlgos/interface/ElectronTkIsolation.h"

EgammaHLTElectronTrackIsolationProducers::EgammaHLTElectronTrackIsolationProducers(const edm::ParameterSet& config)
{

  electronProducer_             = config.getParameter<edm::InputTag>("electronProducer");
  trackProducer_                = config.getParameter<edm::InputTag>("trackProducer");
  recoEcalCandidateProducer_ = config.getParameter<edm::InputTag>("recoEcalCandidateProducer"); 
  beamSpotProducer_ = config.getParameter<edm::InputTag>("beamSpotProducer");

  useGsfTrack_ = config.getParameter<bool>("useGsfTrack");
  useSCRefs_ = config.getParameter<bool>("useSCRefs");
  
  egTrkIsoPtMin_                = config.getParameter<double>("egTrkIsoPtMin");
  egTrkIsoConeSize_             = config.getParameter<double>("egTrkIsoConeSize");
  egTrkIsoZSpan_                = config.getParameter<double>("egTrkIsoZSpan");
  egTrkIsoRSpan_                = config.getParameter<double>("egTrkIsoRSpan");
  egTrkIsoVetoConeSizeBarrel_         = config.getParameter<double>("egTrkIsoVetoConeSizeBarrel");
  egTrkIsoVetoConeSizeEndcap_         = config.getParameter<double>("egTrkIsoVetoConeSizeEndcap");
  // egCheckForOtherEleInCone_     = config.getUntrackedParameter<bool>("egCheckForOtherEleInCone",false);
  egTrkIsoStripBarrel_   = config.getParameter<double>("egTrkIsoStripBarrel");
  egTrkIsoStripEndcap_   = config.getParameter<double>("egTrkIsoStripEndcap");

 


  //register your products
  if(useSCRefs_) produces < reco::RecoEcalCandidateIsolationMap >();
  else produces < reco::ElectronIsolationMap >();
}


EgammaHLTElectronTrackIsolationProducers::~EgammaHLTElectronTrackIsolationProducers(){}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EgammaHLTElectronTrackIsolationProducers::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::Handle<reco::ElectronCollection> electronHandle;
  iEvent.getByLabel(electronProducer_,electronHandle);


 // Get the general tracks
  edm::Handle<reco::TrackCollection> trackHandle;
  iEvent.getByLabel(trackProducer_, trackHandle);
  const reco::TrackCollection* trackCollection = trackHandle.product();

  reco::ElectronIsolationMap eleMap;
  reco::RecoEcalCandidateIsolationMap recoEcalCandMap;

  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel(beamSpotProducer_,recoBeamSpotHandle);
  
  const reco::BeamSpot::Point& beamSpotPosition = recoBeamSpotHandle->position(); 

  ElectronTkIsolation isoAlgo(egTrkIsoConeSize_,egTrkIsoVetoConeSizeBarrel_,egTrkIsoVetoConeSizeEndcap_,egTrkIsoStripBarrel_,egTrkIsoStripEndcap_,egTrkIsoPtMin_, egTrkIsoZSpan_ ,egTrkIsoRSpan_,trackCollection,beamSpotPosition);
  
  if(useSCRefs_){
    edm::Handle<reco::RecoEcalCandidateCollection> recoEcalCandHandle;
    iEvent.getByLabel(recoEcalCandidateProducer_,recoEcalCandHandle);
    for(reco::RecoEcalCandidateCollection::const_iterator iRecoEcalCand = recoEcalCandHandle->begin(); iRecoEcalCand != recoEcalCandHandle->end(); iRecoEcalCand++){
      
      reco::RecoEcalCandidateRef recoEcalCandRef(recoEcalCandHandle,iRecoEcalCand-recoEcalCandHandle->begin());
      
      reco::ElectronRef eleRef;
      for(reco::ElectronCollection::const_iterator eleIt = electronHandle->begin(); eleIt != electronHandle->end(); eleIt++){
	if(eleIt->superCluster()==recoEcalCandRef->superCluster()){
	  eleRef = reco::ElectronRef(electronHandle,eleIt - electronHandle->begin());
	  break;
      }
      }
      float isol=999999;
      if(eleRef.isNonnull()){
	const reco::Track* eleTrk = useGsfTrack_ ? &*eleRef->gsfTrack() : &*eleRef->track();
	isol = isoAlgo.getIso(eleTrk).second;
      }
      recoEcalCandMap.insert(recoEcalCandRef,isol);
    }//end reco ecal candidate ref
  }else{ //we are going to loop over electron instead
    for(reco::ElectronCollection::const_iterator iElectron = electronHandle->begin(); iElectron != electronHandle->end(); iElectron++){
      reco::ElectronRef eleRef(reco::ElectronRef(electronHandle,iElectron - electronHandle->begin()));
      const reco::Track* eleTrk = useGsfTrack_ ? &*eleRef->gsfTrack() : &*eleRef->track();
      float isol = isoAlgo.getIso(eleTrk).second;
      eleMap.insert(eleRef, isol);
    }
  }

  if(useSCRefs_){
    std::auto_ptr<reco::RecoEcalCandidateIsolationMap> mapForEvent(new reco::RecoEcalCandidateIsolationMap(recoEcalCandMap));
    iEvent.put(mapForEvent);
  }else{
    std::auto_ptr<reco::ElectronIsolationMap> mapForEvent(new reco::ElectronIsolationMap(eleMap));
    iEvent.put(mapForEvent);
  }
  

}

//define this as a plug-in
//DEFINE_FWK_MODULE(EgammaHLTTrackIsolationProducers);
