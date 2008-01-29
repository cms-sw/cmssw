// -*- C++ -*-
//
// Package:    DeDxHitsProducer
// Class:      DeDxHitsProducer
// 
/**\class DeDxHitsProducer DeDxHitsProducer.cc RecoTracker/DeDxHitsProducer/src/DeDxHitsProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  andrea
//         Created:  Thu May 31 14:09:02 CEST 2007
// $Id: DeDxHitsProducer.cc,v 1.8 2007/08/13 07:52:36 arizzi Exp $
//
//


// system include files
#include <memory>
#include <vector>
#include <map>

#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoTracker/DeDx/interface/DeDxHitsProducer.h"
#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace reco;
using namespace std;
using namespace DeDxTools;

DeDxHitsProducer::DeDxHitsProducer(const edm::ParameterSet& iConfig)
{
   //register your products
   produces<reco::TrackDeDxHitsCollection>();  
   m_tracksTag = iConfig.getParameter<edm::InputTag>("tracks");
   m_refittedTracksTag = iConfig.getParameter<edm::InputTag>("refittedTracks");
//   m_trajectoriesTag   = iConfig.getParameter<edm::InputTag>("trajectories");
   m_trajTrackAssociationTag   = iConfig.getParameter<edm::InputTag>("trajectoryTrackAssociation");
}


DeDxHitsProducer::~DeDxHitsProducer()
{ 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
DeDxHitsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

/*
 //Code for handling calinbration changes in the EventSetup
   const EventSetupRecord & calibRecord= iSetup.get<XXXXXRcd>();
   unsigned long long newCalibrationCacheId= r.cacheIdentifier();
   if(newCalibrationCacheId!=m_calibrationCacheId)
     {
       m_normalizationMap.clear();
       m_calibrationcacheId=newCalibrationCacheId;
     }
*/
   edm::ESHandle<TrackerGeometry> estracker;
   iSetup.get<TrackerDigiGeometryRecord>().get(estracker);
   m_tracker=&(* estracker);


   edm::Handle<reco::TrackCollection> trackCollectionHandle;
   edm::Handle<reco::TrackCollection> refittedTrackCollectionHandle;
//   edm::Handle<vector<Trajectory> > trajectoryCollectionHandle;
   edm::Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
   iEvent.getByLabel(m_tracksTag,trackCollectionHandle);
   iEvent.getByLabel(m_refittedTracksTag,refittedTrackCollectionHandle);
//   iEvent.getByLabel(m_trajectoriesTag,trajectoryCollectionHandle);
   iEvent.getByLabel(m_trajTrackAssociationTag,trajTrackAssociationHandle);
   
   TrackDeDxHitsCollection * outputCollection = new TrackDeDxHitsCollection(reco::TrackRefProd(trackCollectionHandle));
   
   //Loop on tracks and compute normalized hits
   const reco::TrackCollection &tracks=*trackCollectionHandle.product();
   const reco::TrackCollection &refittedTracks=*refittedTrackCollectionHandle.product();
   const TrajTrackAssociationCollection * trajectoryToTrackMap=trajTrackAssociationHandle.product();
    
   //do old to new tracks association
   std::map<const reco::Track *,const reco::Track *> oldToNewTracksMap;
   if(tracks.size() == refittedTracks.size() ) // assume one to one same order collections it should be!
    {
      for(size_t j=0;j< tracks.size(); j++) oldToNewTracksMap[&tracks[j]]=&refittedTracks[j];
    }
   else
    {  
      LogDebug("DeDxHitsProducer") << "Simple matching not possible! " << tracks.size() << " (original) vs " << refittedTracks.size() << " (refitted)" ;
      //new tracks should be less than old tracks, so loop on old and check compatibility
      for(size_t j=0,i=0;j< tracks.size() && i < refittedTracks.size() ; j++)
        {
	   if(compatibleTracks(tracks[j],refittedTracks[i]))
	    {
   	      oldToNewTracksMap[&tracks[j]]=&refittedTracks[i];
	      i++;
	    } 
	    else
	    {
	      //TODO: put a logdebug warning here!
              edm::LogWarning("DeDxHitsProducer") <<  "Lost track found with pt = " << tracks[j].pt() ;
	    }
        }
    }
  
  //reverse the track-trajectory map
   std::map<const reco::Track *,const Trajectory *> trackToTrajectoryMap;
   for(TrajTrackAssociationCollection::const_iterator it = trajectoryToTrackMap->begin(); it!=trajectoryToTrackMap->end(); ++it)
   {
    trackToTrajectoryMap[&(*it->val)]=&(*it->key);
   }


   reco::TrackCollection::const_iterator tk_it = tracks.begin();  
   for(int j=0;tk_it!=tracks.end();tk_it++,j++)  //loop also index j for AssoVector::setValue
   {
     //Get trajectory from the map for the given track
     //call fnction to compute the normalized hits... given track and trajectory
     //angle,detid,rawCharge

     DeDxHitCollection dedxHits; // the output hits for this track

     const Trajectory * trajectory=trackToTrajectoryMap[oldToNewTracksMap[&(*tk_it)]];
     if(trajectory) 
     {  
        vector<DeDxTools::RawHits> hits = trajectoryRawHits(*trajectory);  
        for(size_t i=0; i < hits.size(); i++)
          {
              float pathLen=thickness(hits[i].detId)/std::abs(hits[i].angleCosine);
              float charge=normalize(hits[i].detId,hits[i].charge*std::abs(hits[i].angleCosine)); 
             dedxHits.push_back( DeDxHit( charge, distance(hits[i].detId), pathLen, hits[i].detId) );
          }
     }
     sort(dedxHits.begin(),dedxHits.end(),less<DeDxHit>());
     outputCollection->setValue(j,dedxHits);

    }

   //put in the event the result
   std::auto_ptr<TrackDeDxHitsCollection> hits(outputCollection);
   iEvent.put(hits);


}

// ------------ method called once each job just before starting event loop  ------------
void 
DeDxHitsProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DeDxHitsProducer::endJob() {
//TODO: if verbose level very high, print the detid->calib map
}

double DeDxHitsProducer::thickness(DetId id)
{
 map<DetId,double>::iterator th=m_thicknessMap.find(id);
 if(th!=m_thicknessMap.end())
   return (*th).second;
 else
 {
  double detThickness=1.;
  //compute thickness normalization
  const GeomDetUnit* it = m_tracker->idToDetUnit(DetId(id));
  bool isPixel = dynamic_cast<const PixelGeomDetUnit*>(it)!=0;
  bool isStrip = dynamic_cast<const StripGeomDetUnit*>(it)!=0;
  if (!isPixel && ! isStrip) {
  //FIXME throw exception
edm::LogWarning("DeDxHitsProducer") << "\t\t this detID doesn't seem to belong to the Tracker";
  detThickness = 1.;
  }else{
    detThickness = it->surface().bounds().thickness();
  }

   m_thicknessMap[id]=detThickness;//computed value
   return detThickness;
 }

}


double DeDxHitsProducer::normalization(DetId id)
{
 map<DetId,double>::iterator norm=m_normalizationMap.find(id);
 if(norm!=m_normalizationMap.end()) 
   return (*norm).second;
 else
 {
  double detNormalization=1./thickness(id);
  
//compute other normalization
  const GeomDetUnit* it = m_tracker->idToDetUnit(DetId(id));
  bool isPixel = dynamic_cast<const PixelGeomDetUnit*>(it)!=0;
  bool isStrip = dynamic_cast<const StripGeomDetUnit*>(it)!=0;

  //FIXME: include gain et al calib
   if(isPixel) detNormalization*=3.61e-06;
   if(isStrip) detNormalization*=3.61e-06*250;
 
   m_normalizationMap[id]=detNormalization;//computed value
   return detNormalization;
 }
 
}


double DeDxHitsProducer::distance(DetId id)
{
 map<DetId,double>::iterator dist=m_distanceMap.find(id);
 if(dist!=m_distanceMap.end()) 
   return (*dist).second;
 else
 {
  const GeomDetUnit* it = m_tracker->idToDetUnit(DetId(id));
   float   d=it->position().mag();
   m_distanceMap[id]=d;
   return d;
 }
 
}

bool DeDxHitsProducer::compatibleTracks(const Track & a,const Track & b)
{
  if(a.pt()+b.pt()!=0 && std::abs(a.pt()-b.pt())/(a.pt()+b.pt()) > 1e4) return false;
  
  return true;
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeDxHitsProducer);
