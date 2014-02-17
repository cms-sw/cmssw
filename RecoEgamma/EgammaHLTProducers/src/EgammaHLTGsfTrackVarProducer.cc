/** \class EgammaHLTElectronDetaDphiProducer
 *
 *  \author Roberto Covarelli (CERN)
 * 
 * $Id: EgammaHLTGsfTrackVarProducer.cc,v 1.2 2012/02/03 18:48:34 sharper Exp $
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTGsfTrackVarProducer.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
//#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
//#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"//needed?
//#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"//needed?

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "RecoEgamma/EgammaTools/interface/ECALPositionCalculator.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

EgammaHLTGsfTrackVarProducer::EgammaHLTGsfTrackVarProducer(const edm::ParameterSet& config)
{
  recoEcalCandTag_ = config.getParameter<edm::InputTag>("recoEcalCandidateProducer");
  inputCollectionTag_             = config.getParameter<edm::InputTag>("inputCollection");
  beamSpotTag_                   = config.getParameter<edm::InputTag>("beamSpotProducer");
  upperTrackNrToRemoveCut_  = config.getParameter<int>("upperTrackNrToRemoveCut"); //zeros out dEtaIn,dPhiIn if nrTracks>= this
  lowerTrackNrToRemoveCut_  = config.getParameter<int>("lowerTrackNrToRemoveCut"); //zeros out dEtaIn,dPhiIn if nrTracks<= this
 
  
  
  //register your products
  produces < reco::RecoEcalCandidateIsolationMap >( "Deta" ).setBranchAlias( "deta" );
  produces < reco::RecoEcalCandidateIsolationMap >( "Dphi" ).setBranchAlias( "dphi" ); 
}

EgammaHLTGsfTrackVarProducer::~EgammaHLTGsfTrackVarProducer(){}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EgammaHLTGsfTrackVarProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  trackExtrapolator_.setup(iSetup);

  // Get the HLT filtered objects
  edm::Handle<reco::RecoEcalCandidateCollection> recoEcalCandHandle;
  iEvent.getByLabel(recoEcalCandTag_,recoEcalCandHandle);

  edm::Handle<reco::ElectronCollection> electronHandle;
  iEvent.getByLabel(inputCollectionTag_,electronHandle);

  edm::Handle<reco::GsfTrackCollection> gsfTracksHandle;
  if(!electronHandle.isValid()) iEvent.getByLabel (inputCollectionTag_,gsfTracksHandle);

  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel(beamSpotTag_,recoBeamSpotHandle);
  // gets its position
  const reco::BeamSpot& beamSpot = *recoBeamSpotHandle; 

  edm::ESHandle<MagneticField> theMagField;
  iSetup.get<IdealMagneticFieldRecord>().get(theMagField);

  reco::RecoEcalCandidateIsolationMap dEtaMap;
  reco::RecoEcalCandidateIsolationMap dPhiMap;
  
  for(reco::RecoEcalCandidateCollection::const_iterator iRecoEcalCand = recoEcalCandHandle->begin(); iRecoEcalCand != recoEcalCandHandle->end(); iRecoEcalCand++){
    reco::RecoEcalCandidateRef recoEcalCandRef(recoEcalCandHandle,iRecoEcalCand-recoEcalCandHandle->begin());
   
    const reco::SuperClusterRef scRef = recoEcalCandRef->superCluster();
   
    //the idea is that we can take the tracks from properly associated electrons or just take all gsf tracks with that sc as a seed
    std::vector<const reco::GsfTrack*> gsfTracks;
    if(electronHandle.isValid()){
      for(reco::ElectronCollection::const_iterator eleIt = electronHandle->begin(); eleIt != electronHandle->end(); eleIt++){
	if(eleIt->superCluster()==scRef){
	  gsfTracks.push_back(&*eleIt->gsfTrack());
	}
      }
    }else{ 
      for(reco::GsfTrackCollection::const_iterator trkIt =gsfTracksHandle->begin();trkIt!=gsfTracksHandle->end();++trkIt){
	edm::RefToBase<TrajectorySeed> seed = trkIt->extra()->seedRef() ;
	reco::ElectronSeedRef elseed = seed.castTo<reco::ElectronSeedRef>() ;
	edm::RefToBase<reco::CaloCluster> caloCluster = elseed->caloCluster() ;
	reco::SuperClusterRef scRefFromTrk = caloCluster.castTo<reco::SuperClusterRef>() ;
	if(scRefFromTrk==scRef){
	  gsfTracks.push_back(&*trkIt);
	}
      }
      
    }
    float dEtaInValue=999999;
    float dPhiInValue=999999;
    
    if(static_cast<int>(gsfTracks.size())>=upperTrackNrToRemoveCut_){
      dEtaInValue=0;
      dPhiInValue=0;
    }else if(static_cast<int>(gsfTracks.size())<=lowerTrackNrToRemoveCut_){
      dEtaInValue=0;
      dPhiInValue=0;
    }else{
      for(size_t trkNr=0;trkNr<gsfTracks.size();trkNr++){
      
	GlobalPoint scPos(scRef->x(),scRef->y(),scRef->z());
	GlobalPoint trackExtrapToSC = trackExtrapolator_.extrapolateTrackPosToPoint(*gsfTracks[trkNr],scPos);
	EleRelPointPair scAtVtx(scRef->position(),trackExtrapToSC,beamSpot.position());
	
	if(fabs(scAtVtx.dEta())<dEtaInValue) dEtaInValue=fabs(scAtVtx.dEta()); //we are allowing them to come from different tracks
	if(fabs(scAtVtx.dPhi())<dPhiInValue) dPhiInValue=fabs(scAtVtx.dPhi());//we are allowing them to come from different tracks
      }	
    }
    
    dEtaMap.insert(recoEcalCandRef, dEtaInValue);
    dPhiMap.insert(recoEcalCandRef, dPhiInValue);
  }

  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> dEtaMapForEvent(new reco::RecoEcalCandidateIsolationMap(dEtaMap));
  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> dPhiMapForEvent(new reco::RecoEcalCandidateIsolationMap(dPhiMap));
  iEvent.put(dEtaMapForEvent, "Deta" );
  iEvent.put(dPhiMapForEvent, "Dphi" );

}



EgammaHLTGsfTrackVarProducer::TrackExtrapolator::TrackExtrapolator(const EgammaHLTGsfTrackVarProducer::TrackExtrapolator& rhs):
  cacheIDTDGeom_(rhs.cacheIDTDGeom_),
  cacheIDMagField_(rhs.cacheIDMagField_),
  magField_(rhs.magField_),
  trackerHandle_(rhs.trackerHandle_),
  mtsMode_(rhs.mtsMode_)
 
{
  if(rhs.mtsTransform_) mtsTransform_ = new MultiTrajectoryStateTransform(*rhs.mtsTransform_);
  else mtsTransform_ =0;
    
}  

EgammaHLTGsfTrackVarProducer::TrackExtrapolator* EgammaHLTGsfTrackVarProducer::TrackExtrapolator::operator=(const EgammaHLTGsfTrackVarProducer::TrackExtrapolator& rhs)
{ 
  if(this!=&rhs){ //just to ensure we're not copying ourselves
    cacheIDTDGeom_ = rhs.cacheIDTDGeom_;
    cacheIDMagField_ = rhs.cacheIDMagField_;
    magField_ = rhs.magField_;
    trackerHandle_ = rhs.trackerHandle_;
    mtsMode_ = rhs.mtsMode_;
    
    delete mtsTransform_;
    if(rhs.mtsTransform_) mtsTransform_ = new MultiTrajectoryStateTransform(*rhs.mtsTransform_);
    else mtsTransform_ =0;
  }
  return this;
}

void EgammaHLTGsfTrackVarProducer::TrackExtrapolator::setup(const edm::EventSetup& iSetup)
{
  bool updateField(false);
  if (cacheIDMagField_!=iSetup.get<IdealMagneticFieldRecord>().cacheIdentifier()){
    updateField = true;
    cacheIDMagField_=iSetup.get<IdealMagneticFieldRecord>().cacheIdentifier();
    iSetup.get<IdealMagneticFieldRecord>().get(magField_);
  }
  
  bool updateGeometry(false);
  if (cacheIDTDGeom_!=iSetup.get<TrackerDigiGeometryRecord>().cacheIdentifier()){
    updateGeometry = true;
    cacheIDTDGeom_=iSetup.get<TrackerDigiGeometryRecord>().cacheIdentifier();
    iSetup.get<TrackerDigiGeometryRecord>().get(trackerHandle_);
  }
  
  if ( updateField || updateGeometry || !mtsTransform_ ) {
    delete mtsTransform_;
    mtsTransform_ = new MultiTrajectoryStateTransform(trackerHandle_.product(),magField_.product());
  }
}

GlobalPoint EgammaHLTGsfTrackVarProducer::TrackExtrapolator::extrapolateTrackPosToPoint(const reco::GsfTrack& gsfTrack,const GlobalPoint& pointToExtrapTo)
{
  TrajectoryStateOnSurface innTSOS = mtsTransform()->innerStateOnSurface(gsfTrack);
  TrajectoryStateOnSurface posTSOS = mtsTransform()->extrapolatedState(innTSOS,pointToExtrapTo);
  GlobalPoint  extrapolatedPos;
  mtsMode()->positionFromModeCartesian(posTSOS,extrapolatedPos);
  return extrapolatedPos;
}

GlobalVector EgammaHLTGsfTrackVarProducer::TrackExtrapolator::extrapolateTrackMomToPoint(const reco::GsfTrack& gsfTrack,const GlobalPoint& pointToExtrapTo)
{
  TrajectoryStateOnSurface innTSOS = mtsTransform()->innerStateOnSurface(gsfTrack);
  TrajectoryStateOnSurface posTSOS = mtsTransform()->extrapolatedState(innTSOS,pointToExtrapTo);
  GlobalVector  extrapolatedMom;
  mtsMode()->momentumFromModeCartesian(posTSOS,extrapolatedMom);
  return extrapolatedMom;
}

//define this as a plug-in
//DEFINE_FWK_MODULE(EgammaHLTTrackIsolationProducers);
