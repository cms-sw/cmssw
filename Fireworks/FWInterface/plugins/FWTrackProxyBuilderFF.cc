// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWTrackProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Nov 25 14:42:13 EST 2008
// $Id: FWTrackProxyBuilder.cc,v 1.15 2012/06/27 20:26:29 amraktad Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveBrowser.h"
#include "TEveTrack.h"
#include "TEvePointSet.h"
#include "TEveCompound.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWMagField.h"


#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Common/interface/EventBase.h"

class FWTrackProxyBuilderFF : public FWProxyBuilderBase {

public:
   FWTrackProxyBuilderFF();
   virtual ~FWTrackProxyBuilderFF();

   REGISTER_PROXYBUILDER_METHODS();
   
   virtual void setItem(const FWEventItem* iItem);

private:
   FWTrackProxyBuilderFF(const FWTrackProxyBuilderFF&); // stop default

   const FWTrackProxyBuilderFF& operator=(const FWTrackProxyBuilderFF&); // stop default

   void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);
   TEveTrackPropagator* m_trackerPropagator;
};

FWTrackProxyBuilderFF::FWTrackProxyBuilderFF(): m_trackerPropagator(0)
{
   m_trackerPropagator = new TEveTrackPropagator();
   m_trackerPropagator->SetStepper( TEveTrackPropagator::kRungeKutta );
   m_trackerPropagator->SetDelta(0.01);
   m_trackerPropagator->SetMaxOrbs(0.7);
   m_trackerPropagator->IncDenyDestroy();
}

FWTrackProxyBuilderFF::~FWTrackProxyBuilderFF()
{
   m_trackerPropagator->DecDenyDestroy();
}

void FWTrackProxyBuilderFF::setItem(const FWEventItem* iItem)
{
   FWProxyBuilderBase::setItem(iItem);
   if (iItem)
   {
      m_trackerPropagator->SetMagFieldObj(context().getField(), false);

      iItem->getConfig()->assertParam("Rnr TrajectoryMeasurement", false);
   }
}

void
FWTrackProxyBuilderFF::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext* vc)
{    
   const reco::TrackCollection * tracks = 0;
   iItem->get( tracks );
   if( tracks == 0 ) return;
   
   const TrajTrackAssociationCollection* TrajToTrackMap;
   try {
      const edm::EventBase* event = item()->getEvent();
      edm::InputTag tag(item()->moduleLabel(), item()->productInstanceLabel(), item()->processName());
      edm::Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
      event->getByLabel(tag, trajTrackAssociationHandle);
      TrajToTrackMap = &*trajTrackAssociationHandle;
      // printf("====================================== %s map %p \n",item()->name().c_str(), (void*)trajTrackAssociationHandle.product() );
   }
   catch (cms::Exception &exception)
   {
      std::cout << exception.what() << std::endl;
      return;
   }
   
 
   bool rnrPathMarks = item()->getConfig()->value<bool>("Rnr TrajectoryMeasurement");
   if (m_trackerPropagator->GetRnrReferences() != rnrPathMarks ) m_trackerPropagator->SetRnrReferences(rnrPathMarks);

   unsigned track_index = 0;
   for(TrajTrackAssociationCollection::const_iterator it = TrajToTrackMap->begin(); it!=TrajToTrackMap->end(); ++it, ++track_index) 
   {
      const reco::Track track = *it->val;
      const Trajectory  traj  = *it->key;
      
      TEveCompound* comp = createCompound();
      setupAddElement( comp, product );
      
      
      if (item()->modelInfo(track_index).displayProperties().isVisible() == false)   continue;

      TEveRecTrack ts;
      ts.fBeta = 1.;
      ts.fSign = track.charge();
      ts.fP.Set(track.px(), track.py(), track.pz());
      ts.fV.Set(track.vx(), track.vy(), track.vz());
      
      //printf("rc->fSign = %d; \n", ts.fSign );
      //printf("rc->fP.Set( %f, %f, %f); \n",  track.px(), track.py(), track.pz());
      //printf("rc->fV.Set(%f, %f, %f); \n",  track.vx(), track.vy(), track.vz());

      TEveTrack* eveTrack = new TEveTrack( &ts, m_trackerPropagator);
      
      // add path-mark from trajectory
      std::vector<TrajectoryMeasurement> measurements = traj.measurements();
      for(std::vector<TrajectoryMeasurement>::const_reverse_iterator measurement_it = measurements.rbegin(); measurement_it!=measurements.rend(); measurement_it++)
      {
         TrajectoryStateOnSurface trajState = measurement_it->updatedState();
         if( !trajState.isValid() ) continue;
         
         /*
           printf("addTrackMesure(track, %f, %f, %f, %f, %f, %f );\n", 
           trajState.globalPosition().x(),trajState.globalPosition().y(), trajState.globalPosition().z(),
           trajState.globalMomentum().x(),trajState.globalMomentum().y(), trajState.globalMomentum().z());
         */

         eveTrack->AddPathMark( TEvePathMark( TEvePathMark::kReference,
                                              TEveVector(trajState.globalPosition().x(),trajState.globalPosition().y(), trajState.globalPosition().z()),
                                              TEveVector(trajState.globalMomentum().x(),trajState.globalMomentum().y(), trajState.globalMomentum().z())));
         
         // eveTrack->AddPathMark( TEvePathMark( TEvePathMark::kDaughter,TEveVector(trajState.globalPosition().x(),trajState.globalPosition().y(), trajState.globalPosition().z())));         
      }
      
      eveTrack->MakeTrack();         
      setupAddElement(eveTrack, comp);      
   }
   //gEve->GetBrowser()->MapWindow();
}

REGISTER_FWPROXYBUILDER(FWTrackProxyBuilderFF, reco::TrackCollection, "TracksFF", FWViewType::kAll3DBits | FWViewType::kAllRPZBits);


