// -*- C++ -*-
//
// Package:    TrackIPProducer
// Class:      TrackIPProducer
// 
/**\class TrackIPProducer TrackIPProducer.cc RecoBTau/TrackIPProducer/src/TrackIPProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Rizzi
//         Created:  Thu Apr  6 09:56:23 CEST 2006
// $Id: TrackIPProducer.cc,v 1.15 2008/02/12 15:37:28 tboccali Exp $
//
//

// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
//#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoBTag/TrackProbability/interface/HistogramProbabilityEstimator.h"
#include "RecoBTag/ImpactParameter/plugins/TrackIPProducer.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"


using namespace std;
using namespace reco;
using namespace edm;

//
// constructors and destructor
//
TrackIPProducer::TrackIPProducer(const edm::ParameterSet& iConfig) : 
  m_config(iConfig),m_probabilityEstimator(0)  {

  m_calibrationCacheId3D= 0;
  m_calibrationCacheId2D= 0;
  
  m_associator = m_config.getParameter<InputTag>("jetTracks");
  m_primaryVertexProducer = m_config.getParameter<InputTag>("primaryVertex");

  m_computeProbabilities = m_config.getParameter<bool>("computeProbabilities"); //FIXME: use or remove
  
  m_cutPixelHits     =  m_config.getParameter<int>("minimumNumberOfPixelHits"); //FIXME: use or remove
  m_cutTotalHits     =  m_config.getParameter<int>("minimumNumberOfHits"); // used
  m_cutMaxTIP        =  m_config.getParameter<double>("maximumTransverseImpactParameter"); // used
  m_cutMinPt         =  m_config.getParameter<double>("minimumTransverseMomentum"); // used
  m_cutMaxDecayLen   =  m_config.getParameter<double>("maximumDecayLength"); //used
  m_cutMaxChiSquared =  m_config.getParameter<double>("maximumChiSquared"); //used
  m_cutMaxLIP        =  m_config.getParameter<double>("maximumLongitudinalImpactParameter"); //used
  m_cutMaxDistToAxis =  m_config.getParameter<double>("maximumDistanceToJetAxis"); //used
  m_directionWithTracks  =  m_config.getParameter<bool>("jetDirectionUsingTracks"); //used
  m_useTrackQuality      =  m_config.getParameter<bool>("useTrackQuality"); //used
  

   produces<reco::TrackIPTagInfoCollection>();

}

TrackIPProducer::~TrackIPProducer()
{
 delete m_probabilityEstimator;
}

//
// member functions
//
// ------------ method called to produce the data  ------------
void
TrackIPProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   if(m_computeProbabilities ) checkEventSetup(iSetup); //Update probability estimator if event setup is changed
 
  //input objects 
   Handle<reco::JetTracksAssociationCollection> jetTracksAssociation;
   iEvent.getByLabel(m_associator,jetTracksAssociation);
   
   Handle<reco::VertexCollection> primaryVertex;
   iEvent.getByLabel(m_primaryVertexProducer,primaryVertex);
   
   edm::ESHandle<TransientTrackBuilder> builder;
   iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",builder);
  //  m_algo.setTransientTrackBuilder(builder.product());

 
   //output collections 
   reco::TrackIPTagInfoCollection * outCollection = new reco::TrackIPTagInfoCollection();

   //use first pv of the collection
   //FIXME: use BeamSpot when pv is missing
   const  Vertex  *pv;
   edm::Ref<VertexCollection> * pvRef;
   bool pvFound = (primaryVertex->size() != 0);
   if(pvFound)
   {
    pv = &(*primaryVertex->begin());
    pvRef = new edm::Ref<VertexCollection>(primaryVertex,0); // we always use the first vertex at the moment
   }
    else 
   { // create a dummy PV
     Vertex::Error e;
     e(0,0)=0.0015*0.0015;
      e(1,1)=0.0015*0.0015;
     e(2,2)=15.*15.;
     Vertex::Point p(0,0,0);
     pv=  new Vertex(p,e,1,1,1);
     pvRef = new edm::Ref<VertexCollection>();
   }
   
   double pvZ=pv->z();
 




   int i=0;
   JetTracksAssociationCollection::const_iterator it = jetTracksAssociation->begin();
   TwoTrackMinimumDistance minDist;
   for(; it != jetTracksAssociation->end(); it++, i++)
     {
        TrackRefVector tracks = it->second;
        math::XYZVector jetMomentum=it->first->momentum()/2.;
        if(m_directionWithTracks) 
         {
           for (TrackRefVector::const_iterator itTrack = tracks.begin(); itTrack != tracks.end(); ++itTrack) {
               if((**itTrack).numberOfValidHits() >= m_cutTotalHits )  //minimal quality cuts
                  jetMomentum+=(**itTrack).momentum();
             }
         }
          else
         {
            jetMomentum=it->first->momentum();
         } 
        GlobalVector direction(jetMomentum.x(),jetMomentum.y(),jetMomentum.z());
        TrackRefVector selectedTracks;
        vector<Measurement1D> ip3Dv,ip2Dv,dLenv,jetDistv;
        vector<float> prob2D,prob3D;
        vector<TrackIPTagInfo::TrackIPData> ipData;

        multimap<float,int> significanceMap; 
        int ind =0;
        for (TrackRefVector::const_iterator itTrack = tracks.begin(); itTrack != tracks.end(); ++itTrack) {
             const Track & track = **itTrack;
             const TransientTrack & transientTrack = builder->build(&(**itTrack));
/*         cout << " pt " <<  track.pt() <<
                 " d0 " <<  fabs(track.d0()) <<
                 " #hit " <<    track.hitPattern().numberOfValidHits()<<
                 " ipZ " <<   fabs(track.dz()-pvZ)<<
                 " chi2 " <<  track.normalizedChi2()<<
                 " #pixel " <<    track.hitPattern().numberOfValidPixelHits()<< endl;
*/
         if(     track.pt() > m_cutMinPt  &&                          // minimum pt
		 // CHANGE: refer to PV
		 //                 fabs(track.d0()) < m_cutMaxTIP &&                // max transverse i.p.
		 fabs(IPTools::signedTransverseImpactParameter(transientTrack, direction, *pv).second.value())
		 < m_cutMaxTIP &&                // max transverse i.p.
		 // end of correction
                 track.hitPattern().numberOfValidHits() >= m_cutTotalHits &&         // min num tracker hits
                 fabs(track.dz()-pvZ) < m_cutMaxLIP &&            // z-impact parameter, loose only to reject PU
                 track.normalizedChi2() < m_cutMaxChiSquared &&   // normalized chi2
                 track.hitPattern().numberOfValidPixelHits() >= m_cutPixelHits //min # pix hits 
           )     // quality cuts
        { 
         //Fill vectors
         //TODO: what if .first is false?
         ip3Dv.push_back(IPTools::signedImpactParameter3D(transientTrack,direction,*pv).second);
         ip2Dv.push_back(IPTools::signedTransverseImpactParameter(transientTrack,direction,*pv).second);
         dLenv.push_back(IPTools::signedDecayLength3D(transientTrack,direction,*pv).second);
         jetDistv.push_back(IPTools::jetTrackDistance(transientTrack,direction,*pv).second);
         TrackIPTagInfo::TrackIPData trackIp;
         trackIp.ip3d=IPTools::signedImpactParameter3D(transientTrack,direction,*pv).second;
         trackIp.ip2d=IPTools::signedTransverseImpactParameter(transientTrack,direction,*pv).second;
         TrajectoryStateOnSurface closest = IPTools::closestApproachToJet(transientTrack.impactPointState(), *pv, direction,transientTrack.field());
         if(closest.isValid())  trackIp.closestToJetAxis=closest.globalPosition();
         //TODO:cross check if it is the same using other methods
         trackIp.distanceToJetAxis=IPTools::jetTrackDistance(transientTrack,direction,*pv).second.value();
       
         significanceMap.insert(pair<float,int>(trackIp.ip3d.significance(), ind++) ); 

         ipData.push_back(trackIp);
         selectedTracks.push_back(*itTrack);
        
         if(m_computeProbabilities) {
              //probability with 3D ip
              pair<bool,double> probability =  m_probabilityEstimator->probability(m_useTrackQuality , 0,ipData.back().ip3d.significance(),track,*(it->first),*pv);
              if(probability.first)  prob3D.push_back(probability.second); else  prob3D.push_back(-1.); 
              
              //probability with 2D ip
              probability =  m_probabilityEstimator->probability(m_useTrackQuality ,1,ipData.back().ip2d.significance(),track,*(it->first),*pv);
              if(probability.first)  prob2D.push_back(probability.second); else  prob2D.push_back(-1.); 

          } 
    
         } // quality cuts if
     
      } //track loop
       
       if(ipData.size() >  1)
       {
        multimap<float,int>::iterator last=significanceMap.end();
        last--;
        int first=last->second;
        last--;
        int second=last->second;
       
        for(int n=0;n< ipData.size();n++)
        {
               int use;
               if(n==first) use = second; else use = first;
               TrajectoryStateOnSurface trackState1 =  builder->build(selectedTracks[n]).impactPointState();
               TrajectoryStateOnSurface trackState2 =  builder->build(selectedTracks[use]).impactPointState();
	       minDist.calculate(trackState1,trackState2);
               std::pair<GlobalPoint,GlobalPoint> points = minDist.points();
               float distance = ( points.first - points.second ).mag();
               ipData[n].closestToFirstTrack=points.first;
               ipData[n].distanceToFirstTrack=distance;

        }
       }
       TrackIPTagInfo tagInfo(ipData,prob2D,prob3D,selectedTracks,
                              edm::Ref<JetTracksAssociationCollection>(jetTracksAssociation,i),
                              *pvRef);
       outCollection->push_back(tagInfo); 
     }
 
    std::auto_ptr<reco::TrackIPTagInfoCollection> result(outCollection);
   iEvent.put(result);
 
   if(!pvFound) delete pv; //dummy pv deleted
   delete pvRef;
}


#include "CondFormats/BTauObjects/interface/TrackProbabilityCalibration.h"
#include "RecoBTag/XMLCalibration/interface/CalibrationInterface.h"
#include "CondFormats/DataRecord/interface/BTagTrackProbability2DRcd.h"
#include "CondFormats/DataRecord/interface/BTagTrackProbability3DRcd.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"


void TrackIPProducer::checkEventSetup(const EventSetup & iSetup)
 {
using namespace edm;
using namespace edm::eventsetup;
   const EventSetupRecord & re2D= iSetup.get<BTagTrackProbability2DRcd>();
   const EventSetupRecord & re3D= iSetup.get<BTagTrackProbability3DRcd>();
   unsigned long long cacheId2D= re2D.cacheIdentifier();
   unsigned long long cacheId3D= re3D.cacheIdentifier();

   if(cacheId2D!=m_calibrationCacheId2D || cacheId3D!=m_calibrationCacheId3D  )  //Calibration changed
   {
     //iSetup.get<BTagTrackProbabilityRcd>().get(calib);
     ESHandle<TrackProbabilityCalibration> calib2DHandle;
     iSetup.get<BTagTrackProbability2DRcd>().get(calib2DHandle);
     ESHandle<TrackProbabilityCalibration> calib3DHandle;
     iSetup.get<BTagTrackProbability3DRcd>().get(calib3DHandle);

     const TrackProbabilityCalibration *  ca2D= calib2DHandle.product();
     const TrackProbabilityCalibration *  ca3D= calib3DHandle.product();

     if(m_probabilityEstimator) delete m_probabilityEstimator;  
     m_probabilityEstimator=new HistogramProbabilityEstimator(ca3D,ca2D);

   }
   m_calibrationCacheId3D=cacheId3D;
   m_calibrationCacheId2D=cacheId2D;
}



