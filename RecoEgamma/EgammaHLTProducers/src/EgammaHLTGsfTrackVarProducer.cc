/** \class EgammaHLTElectronDetaDphiProducer
 *
 *  \author Roberto Covarelli (CERN)
 * 
 * $Id: EgammaHLTGsfTrackVarProducer.cc,v 1.1 2012/01/23 12:56:38 sharper Exp $
 *
 */

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTGsfTrackVarProducer.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "RecoEgamma/EgammaTools/interface/ECALPositionCalculator.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

EgammaHLTGsfTrackVarProducer::EgammaHLTGsfTrackVarProducer(const edm::ParameterSet& config):
  recoEcalCandTag_         (consumes<reco::RecoEcalCandidateCollection>(config.getParameter<edm::InputTag>("recoEcalCandidateProducer"))),
  inputCollectionTag1_     (consumes<reco::ElectronCollection>(config.getParameter<edm::InputTag>("inputCollection"))),
  inputCollectionTag2_     (consumes<reco::GsfTrackCollection>(config.getParameter<edm::InputTag>("inputCollection"))),
  beamSpotTag_             (consumes<reco::BeamSpot>(config.getParameter<edm::InputTag>("beamSpotProducer"))),
  upperTrackNrToRemoveCut_ (config.getParameter<int>("upperTrackNrToRemoveCut")),
  lowerTrackNrToRemoveCut_ (config.getParameter<int>("lowerTrackNrToRemoveCut")),
  useDefaultValuesForBarrel_  (config.getParameter<bool>("useDefaultValuesForBarrel")),
  useDefaultValuesForEndcap_  (config.getParameter<bool>("useDefaultValuesForEndcap"))
{
 
  //register your products
  produces < reco::RecoEcalCandidateIsolationMap >( "Deta" ).setBranchAlias( "deta" );
  produces < reco::RecoEcalCandidateIsolationMap >( "DetaSeed" ).setBranchAlias( "detaseed" );
  produces < reco::RecoEcalCandidateIsolationMap >( "Dphi" ).setBranchAlias( "dphi" );
  produces < reco::RecoEcalCandidateIsolationMap >( "OneOESuperMinusOneOP" );
  produces < reco::RecoEcalCandidateIsolationMap >( "OneOESeedMinusOneOP" );
  produces < reco::RecoEcalCandidateIsolationMap >( "MissingHits" ).setBranchAlias( "missinghits" );
  produces < reco::RecoEcalCandidateIsolationMap >( "Chi2" ).setBranchAlias( "chi2" );
  produces < reco::RecoEcalCandidateIsolationMap >( "ValidHits" ).setBranchAlias( "validhits" );
}

EgammaHLTGsfTrackVarProducer::~EgammaHLTGsfTrackVarProducer()
{}

void EgammaHLTGsfTrackVarProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(("recoEcalCandidateProducer"), edm::InputTag("hltRecoEcalSuperClusterActivityCandidate"));
  desc.add<edm::InputTag>(("inputCollection"), edm::InputTag("hltActivityElectronGsfTracks"));
  desc.add<edm::InputTag>(("beamSpotProducer"), edm::InputTag("hltOnlineBeamSpot"));
  desc.add<int>(("upperTrackNrToRemoveCut"), 9999); 
  desc.add<int>(("lowerTrackNrToRemoveCut"), -1);
  desc.add<bool>(("useDefaultValuesForBarrel"),false);
  desc.add<bool>(("useDefaultValuesForEndcap"),false);
  
  descriptions.add("hltEgammaHLTGsfTrackVarProducer", desc);
}
void EgammaHLTGsfTrackVarProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  trackExtrapolator_.setup(iSetup);
  
  // Get the HLT filtered objects
  edm::Handle<reco::RecoEcalCandidateCollection> recoEcalCandHandle;
  iEvent.getByToken(recoEcalCandTag_,recoEcalCandHandle);

  edm::Handle<reco::ElectronCollection> electronHandle;
  iEvent.getByToken(inputCollectionTag1_,electronHandle);

  edm::Handle<reco::GsfTrackCollection> gsfTracksHandle;
  if(!electronHandle.isValid()) 
    iEvent.getByToken(inputCollectionTag2_, gsfTracksHandle);

  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(beamSpotTag_,recoBeamSpotHandle);
  // gets its position
  const reco::BeamSpot& beamSpot = *recoBeamSpotHandle; 

  edm::ESHandle<MagneticField> theMagField;
  iSetup.get<IdealMagneticFieldRecord>().get(theMagField);

  reco::RecoEcalCandidateIsolationMap dEtaMap(recoEcalCandHandle);
  reco::RecoEcalCandidateIsolationMap dEtaSeedMap(recoEcalCandHandle);
  reco::RecoEcalCandidateIsolationMap dPhiMap(recoEcalCandHandle);
  reco::RecoEcalCandidateIsolationMap oneOverESuperMinusOneOverPMap(recoEcalCandHandle);
  reco::RecoEcalCandidateIsolationMap oneOverESeedMinusOneOverPMap(recoEcalCandHandle);
  reco::RecoEcalCandidateIsolationMap missingHitsMap(recoEcalCandHandle);
  reco::RecoEcalCandidateIsolationMap validHitsMap(recoEcalCandHandle);
  reco::RecoEcalCandidateIsolationMap chi2Map(recoEcalCandHandle);

  for(reco::RecoEcalCandidateCollection::const_iterator iRecoEcalCand = recoEcalCandHandle->begin(); iRecoEcalCand != recoEcalCandHandle->end(); iRecoEcalCand++){
    reco::RecoEcalCandidateRef recoEcalCandRef(recoEcalCandHandle,iRecoEcalCand-recoEcalCandHandle->begin());
   
    const reco::SuperClusterRef scRef = recoEcalCandRef->superCluster();
    bool isBarrel = std::abs(recoEcalCandRef->eta())<1.479;
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

    int validHitsValue = 0;
    float chi2Value = 9999999.;
    float missingHitsValue = 9999999;
    float dEtaInValue=999999;
    float dEtaSeedInValue=999999;
    float dPhiInValue=999999;
    float oneOverESuperMinusOneOverPValue=999999;
    float oneOverESeedMinusOneOverPValue=999999;
    
    const int nrTracks = gsfTracks.size();
    const bool rmCutsDueToNrTracks = nrTracks <= lowerTrackNrToRemoveCut_ ||
                                     nrTracks >= upperTrackNrToRemoveCut_;
    //to use the default values, we require at least one track...
    const bool useDefaultValues = isBarrel ? useDefaultValuesForBarrel_ && nrTracks>=1 : useDefaultValuesForEndcap_ && nrTracks>=1;

    if(rmCutsDueToNrTracks || useDefaultValues){
      dEtaInValue=0;
      dEtaSeedInValue=0;
      dPhiInValue=0;
      missingHitsValue = 0;
      validHitsValue = 100;
      chi2Value = 0;
      oneOverESuperMinusOneOverPValue = 0;
      oneOverESeedMinusOneOverPValue = 0;
    }else{
      for(size_t trkNr=0;trkNr<gsfTracks.size();trkNr++){
      
	GlobalPoint scPos(scRef->x(),scRef->y(),scRef->z());
	GlobalPoint trackExtrapToSC = trackExtrapolator_.extrapolateTrackPosToPoint(*gsfTracks[trkNr],scPos);
	EleRelPointPair scAtVtx(scRef->position(),trackExtrapToSC,beamSpot.position());
	
	float trkP = gsfTracks[trkNr]->p();
	if(scRef->energy()!=0 && trkP!=0){
	  if(fabs(1/scRef->energy() - 1/trkP)<oneOverESuperMinusOneOverPValue) oneOverESuperMinusOneOverPValue =fabs(1/scRef->energy() - 1/trkP);
	}
	if(scRef->seed().isNonnull() && scRef->seed()->energy()!=0 && trkP!=0){
	  if(fabs(1/scRef->seed()->energy() - 1/trkP)<oneOverESeedMinusOneOverPValue) oneOverESeedMinusOneOverPValue =fabs(1/scRef->seed()->energy() - 1/trkP);
	}


	if (gsfTracks[trkNr]->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) < missingHitsValue) 
	  missingHitsValue = gsfTracks[trkNr]->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
	
	if (gsfTracks[trkNr]->numberOfValidHits() < validHitsValue)
	  validHitsValue = gsfTracks[trkNr]->numberOfValidHits();

	if (gsfTracks[trkNr]->numberOfValidHits() < chi2Value)
	  chi2Value = gsfTracks[trkNr]->normalizedChi2();

	if (fabs(scAtVtx.dEta())<dEtaInValue) 
	  dEtaInValue=fabs(scAtVtx.dEta()); //we are allowing them to come from different tracks

	if (fabs(scAtVtx.dEta())<dEtaSeedInValue) 
	  dEtaSeedInValue = fabs(scAtVtx.dEta()-scRef->position().eta()+scRef->seed()->position().eta());

	if (fabs(scAtVtx.dPhi())<dPhiInValue) 
	  dPhiInValue=fabs(scAtVtx.dPhi());//we are allowing them to come from different tracks
      }	
    }
   
    dEtaMap.insert(recoEcalCandRef, dEtaInValue);
    dEtaSeedMap.insert(recoEcalCandRef, dEtaSeedInValue);
    dPhiMap.insert(recoEcalCandRef, dPhiInValue);
    oneOverESuperMinusOneOverPMap.insert(recoEcalCandRef,oneOverESuperMinusOneOverPValue);   
    oneOverESeedMinusOneOverPMap.insert(recoEcalCandRef,oneOverESeedMinusOneOverPValue);
    missingHitsMap.insert(recoEcalCandRef, missingHitsValue);
    validHitsMap.insert(recoEcalCandRef, validHitsValue);
    chi2Map.insert(recoEcalCandRef, chi2Value);
  }

  auto dEtaMapForEvent = std::make_unique<reco::RecoEcalCandidateIsolationMap>(dEtaMap);
  auto dEtaSeedMapForEvent = std::make_unique<reco::RecoEcalCandidateIsolationMap>(dEtaSeedMap);
  auto dPhiMapForEvent = std::make_unique<reco::RecoEcalCandidateIsolationMap>(dPhiMap);
  auto oneOverESuperMinusOneOverPMapForEvent = std::make_unique<reco::RecoEcalCandidateIsolationMap>(oneOverESuperMinusOneOverPMap);
  auto oneOverESeedMinusOneOverPMapForEvent = std::make_unique<reco::RecoEcalCandidateIsolationMap>(oneOverESeedMinusOneOverPMap);
  auto missingHitsForEvent = std::make_unique<reco::RecoEcalCandidateIsolationMap>(missingHitsMap);
  auto validHitsForEvent = std::make_unique<reco::RecoEcalCandidateIsolationMap>(validHitsMap);
  auto chi2ForEvent = std::make_unique<reco::RecoEcalCandidateIsolationMap>(chi2Map);

  iEvent.put(std::move(dEtaMapForEvent), "Deta" );
  iEvent.put(std::move(dEtaSeedMapForEvent), "DetaSeed" );
  iEvent.put(std::move(dPhiMapForEvent), "Dphi" );
  iEvent.put(std::move(oneOverESuperMinusOneOverPMapForEvent), "OneOESuperMinusOneOP");
  iEvent.put(std::move(oneOverESeedMinusOneOverPMapForEvent), "OneOESeedMinusOneOP");
  iEvent.put(std::move(missingHitsForEvent), "MissingHits");
  iEvent.put(std::move(validHitsForEvent), "ValidHits");
  iEvent.put(std::move(chi2ForEvent), "Chi2");
}


EgammaHLTGsfTrackVarProducer::TrackExtrapolator::TrackExtrapolator(const EgammaHLTGsfTrackVarProducer::TrackExtrapolator& rhs):
  cacheIDTDGeom_(rhs.cacheIDTDGeom_),
  cacheIDMagField_(rhs.cacheIDMagField_),
  magField_(rhs.magField_),
  trackerHandle_(rhs.trackerHandle_),
  mtsMode_(rhs.mtsMode_)
 
{
  if(rhs.mtsTransform_) mtsTransform_ = new MultiTrajectoryStateTransform(*rhs.mtsTransform_);
  else mtsTransform_ =nullptr;
    
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
    else mtsTransform_ =nullptr;
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
