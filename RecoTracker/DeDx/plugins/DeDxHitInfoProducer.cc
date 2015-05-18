// -*- C++ -*-
//
// Package:    DeDxHitInfoProducer
// Class:      DeDxHitInfoProducer
// 
/**\class DeDxHitInfoProducer DeDxHitInfoProducer.cc RecoTracker/DeDx/plugins/DeDxHitInfoProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  loic Quertenmont (querten)
//         Created:  Mon Nov 21 14:09:02 CEST 2014
//

#include "RecoTracker/DeDx/plugins/DeDxHitInfoProducer.h"

// system include files


using namespace reco;
using namespace std;
using namespace edm;

DeDxHitInfoProducer::DeDxHitInfoProducer(const edm::ParameterSet& iConfig):
   useTrajectory     ( iConfig.getParameter<bool>    ("useTrajectory")  ),
   usePixel          ( iConfig.getParameter<bool>    ("usePixel")       ),
   useStrip          ( iConfig.getParameter<bool>    ("useStrip")       ),
   MeVperADCPixel    ( iConfig.getParameter<double>  ("MeVperADCPixel") ),
   MeVperADCStrip    ( iConfig.getParameter<double>  ("MeVperADCStrip") ),
   minTrackHits      ( iConfig.getParameter<unsigned>("minTrackHits")   ),
   minTrackPt        ( iConfig.getParameter<double>  ("minTrackPt"  )   ),
   maxTrackEta       ( iConfig.getParameter<double>  ("maxTrackEta" )   ),
   m_calibrationPath ( iConfig.getParameter<string>  ("calibrationPath")),
   useCalibration    ( iConfig.getParameter<bool>    ("useCalibration") ),
   shapetest         ( iConfig.getParameter<bool>    ("shapeTest")      )
{
   produces<reco::DeDxHitInfoCollection >();
   produces<reco::DeDxHitInfoAss >();

   m_tracksTag = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"));
   m_trajTrackAssociationTag   = consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("trajectoryTrackAssociation"));

   if(!usePixel && !useStrip)
   edm::LogError("DeDxHitsProducer") << "No Pixel Hits NOR Strip Hits will be saved.  Running this module is useless";
}


DeDxHitInfoProducer::~DeDxHitInfoProducer(){}

// ------------ method called once each job just before starting event loop  ------------
void  DeDxHitInfoProducer::beginRun(edm::Run const& run, const edm::EventSetup& iSetup)
{
   if(useCalibration && calibGains.size()==0){
      edm::ESHandle<TrackerGeometry> tkGeom;
      iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );
      m_off = tkGeom->offsetDU(GeomDetEnumerators::PixelBarrel); //index start at the first pixel

      DeDxTools::makeCalibrationMap(m_calibrationPath, *tkGeom, calibGains, m_off);
   }
}



void DeDxHitInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{  
  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByToken(m_tracksTag,trackCollectionHandle);
  const TrackCollection& trackCollection(*trackCollectionHandle.product());

  Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
  if(useTrajectory)iEvent.getByToken(m_trajTrackAssociationTag, trajTrackAssociationHandle);

  // creates the output collection
  std::auto_ptr<reco::DeDxHitInfoCollection> resultdedxHitColl(new reco::DeDxHitInfoCollection);

  std::vector<int> indices;

  TrajTrackAssociationCollection::const_iterator cit;
  if(useTrajectory)cit = trajTrackAssociationHandle->begin();
  for(unsigned int j=0;j<trackCollection.size();j++){            
     const reco::Track& track = trackCollection[j];

     //track selection
     if(track.pt()<minTrackPt ||  std::abs(track.eta())>maxTrackEta ||track.numberOfValidHits()<minTrackHits){
        cit++;
        indices.push_back(-1);
        continue;
     }

     reco::DeDxHitInfo hitDeDxInfo;
 
     if(useTrajectory){  //trajectory allows to take into account the local direction of the particle on the module sensor --> muc much better 'dx' measurement
        const edm::Ref<std::vector<Trajectory> > traj = cit->key; cit++;
        const vector<TrajectoryMeasurement> & measurements = traj->measurements();
        for(vector<TrajectoryMeasurement>::const_iterator it = measurements.begin(); it!=measurements.end(); it++){
           const TrajectoryStateOnSurface& trajState=it->updatedState();
           if( !trajState.isValid()) continue;
     
           const TrackingRecHit* recHit=(*it->recHit()).hit();
           if(!recHit)continue;
           const LocalVector& trackDirection = trajState.localDirection();
           float cosine = trackDirection.z()/trackDirection.mag();           

           processHit(recHit, trajState.localMomentum().mag(), cosine, hitDeDxInfo, trajState.localPosition());
        }

     }else{ //assume that the particles trajectory is a straight line originating from the center of the detector  (can be improved)
        for(unsigned int h=0;h<track.recHitsSize();h++){
           const TrackingRecHit* recHit = &(*(track.recHit(h)));
           auto const & thit = static_cast<BaseTrackerRecHit const&>(*recHit);
           if(!thit.isValid())continue;//make sure it's a tracker hit

           const GlobalVector& ModuleNormal = recHit->detUnit()->surface().normalVector();         
           float cosine = (track.px()*ModuleNormal.x()+track.py()*ModuleNormal.y()+track.pz()*ModuleNormal.z())/track.p();
 
           processHit(recHit, track.p(), cosine, hitDeDxInfo, LocalPoint(0.0,0.0));
        } 
     }

     indices.push_back(resultdedxHitColl->size());
     resultdedxHitColl->push_back(hitDeDxInfo);
  }
  ///////////////////////////////////////
 

  edm::OrphanHandle<reco::DeDxHitInfoCollection> dedxHitCollHandle = iEvent.put(resultdedxHitColl);

  //create map passing the handle to the matched collection
  std::auto_ptr<reco::DeDxHitInfoAss> dedxMatch(new reco::DeDxHitInfoAss(dedxHitCollHandle));
  reco::DeDxHitInfoAss::Filler filler(*dedxMatch);  
  filler.insert(trackCollectionHandle, indices.begin(), indices.end()); 
  filler.fill();
  iEvent.put(dedxMatch);
}

void DeDxHitInfoProducer::processHit(const TrackingRecHit* recHit, const float trackMomentum, const float cosine, reco::DeDxHitInfo& hitDeDxInfo,  const LocalPoint& hitLocalPos){
      auto const & thit = static_cast<BaseTrackerRecHit const&>(*recHit);
      if(!thit.isValid())return;

      float cosineAbs = std::max(0.00000001f,std::abs(cosine));//make sure cosine is not 0

      auto const & clus = thit.firstClusterRef();
      if(!clus.isValid())return;

      if(clus.isPixel()){
          if(!usePixel) return;

          auto& detUnit     = *(recHit->detUnit());
          float pathLen     = detUnit.surface().bounds().thickness()/cosineAbs;
          float chargeAbs   = clus.pixelCluster().charge();
          hitDeDxInfo.addHit(chargeAbs, pathLen, thit.geographicalId(), hitLocalPos, clus.pixelCluster() );
       }else if(clus.isStrip() && !thit.isMatched()){
          if(!useStrip) return;

          auto& detUnit     = *(recHit->detUnit());
          int   NSaturating = 0;
          float pathLen     = detUnit.surface().bounds().thickness()/cosineAbs;
          float chargeAbs   = DeDxTools::getCharge(&(clus.stripCluster()),NSaturating, detUnit, calibGains, m_off);
          hitDeDxInfo.addHit(chargeAbs, pathLen, thit.geographicalId(), hitLocalPos, clus.stripCluster() );
       }else if(clus.isStrip() && thit.isMatched()){
          if(!useStrip) return;
          const SiStripMatchedRecHit2D* matchedHit=dynamic_cast<const SiStripMatchedRecHit2D*>(recHit);
          if(!matchedHit)return;

          auto& detUnitM     = *(matchedHit->monoHit().detUnit());
          int   NSaturating = 0;
          float pathLen     = detUnitM.surface().bounds().thickness()/cosineAbs;
          float chargeAbs   = DeDxTools::getCharge(&(matchedHit->monoHit().stripCluster()),NSaturating, detUnitM, calibGains, m_off);
          hitDeDxInfo.addHit(chargeAbs, pathLen, thit.geographicalId(), hitLocalPos, matchedHit->monoHit().stripCluster() );

          auto& detUnitS     = *(matchedHit->stereoHit().detUnit());
          NSaturating = 0;
          pathLen     = detUnitS.surface().bounds().thickness()/cosineAbs;
          chargeAbs   = DeDxTools::getCharge(&(matchedHit->stereoHit().stripCluster()),NSaturating, detUnitS, calibGains, m_off);
          hitDeDxInfo.addHit(chargeAbs, pathLen, thit.geographicalId(), hitLocalPos, matchedHit->stereoHit().stripCluster() );          
       }
}



//define this as a plug-in
DEFINE_FWK_MODULE(DeDxHitInfoProducer);
