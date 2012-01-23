/** \class EgammaHLTElectronDetaDphiProducer
 *
 *  \author Roberto Covarelli (CERN)
 * 
 * $Id: EgammaHLTGsfTrackVarProducer.cc,v 1.4 2011/12/19 11:17:28 sani Exp $
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
    reco::GsfTrackRef gsfTrkRef;
    for(reco::ElectronCollection::const_iterator eleIt = electronHandle->begin(); eleIt != electronHandle->end(); eleIt++){
      if(eleIt->superCluster()==scRef){
	gsfTrkRef=eleIt->gsfTrack();
	break;
      }
    }
    if(gsfTrkRef.isNonnull()){
      GlobalPoint scPos(scRef->x(),scRef->y(),scRef->z());
      GlobalPoint trackExtrapToSC = trackExtrapolator_.extrapolateTrackPosToPoint(*gsfTrkRef,scPos);
      EleRelPointPair scAtVtx(scRef->position(),trackExtrapToSC,beamSpot.position());
      
      dEtaMap.insert(recoEcalCandRef, scAtVtx.dEta());
      dPhiMap.insert(recoEcalCandRef, scAtVtx.dPhi());
    }else{ 
      dEtaMap.insert(recoEcalCandRef, 999999);
      dPhiMap.insert(recoEcalCandRef, 999999);
    }
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
