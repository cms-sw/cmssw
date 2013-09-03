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
// $Id: FWTrackProxyBuilderFF.cc,v 1.4 2012/10/22 20:08:20 amraktad Exp $
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
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
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
   virtual bool visibilityModelChanges(const FWModelId&, TEveElement*, FWViewType::EType, const FWViewContext*);
   
private:
   FWTrackProxyBuilderFF(const FWTrackProxyBuilderFF&); // stop default
   const FWTrackProxyBuilderFF& operator=(const FWTrackProxyBuilderFF&); // stop default

   void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);
   void buildTrack(TrajTrackAssociationCollection::const_iterator it, TEveCompound* comp);

   
   TEveTrackPropagator* m_trackerPropagator;
   const TrajTrackAssociationCollection* m_trajToTrackMap;
};

FWTrackProxyBuilderFF::FWTrackProxyBuilderFF(): 
m_trackerPropagator(0),
m_trajToTrackMap(0)
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
   
   try {
      const edm::EventBase* event = item()->getEvent();
      edm::InputTag tag(item()->moduleLabel(), item()->productInstanceLabel(), item()->processName());
      edm::Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
      event->getByLabel(tag, trajTrackAssociationHandle);
      m_trajToTrackMap = &*trajTrackAssociationHandle;
   }
   catch (cms::Exception &exception)
   {
      m_trajToTrackMap =0;
      std::cout << exception.what() << std::endl;
      return;
   }
   
 
   bool rnrPathMarks = item()->getConfig()->value<bool>("Rnr TrajectoryMeasurement");
   if (m_trackerPropagator->GetRnrReferences() != rnrPathMarks ) m_trackerPropagator->SetRnrReferences(rnrPathMarks);

   unsigned track_index = 0;
   for(TrajTrackAssociationCollection::const_iterator it = m_trajToTrackMap->begin(); it!=m_trajToTrackMap->end(); ++it, ++track_index) 
   {
      TEveCompound* comp = createCompound();
      setupAddElement( comp, product );
      
      if (item()->modelInfo(track_index).displayProperties().isVisible())
         buildTrack(it, comp);
   }
   
  gEve->GetBrowser()->MapWindow();
}

void FWTrackProxyBuilderFF::buildTrack(TrajTrackAssociationCollection::const_iterator it, TEveCompound* comp)
{
   const reco::Track track = *it->val;
   const Trajectory  traj  = *it->key;
   
   TEveRecTrack ts;
   ts.fBeta = 1.;
   ts.fSign = track.charge();
   ts.fP.Set(track.px(), track.py(), track.pz());
   ts.fV.Set(track.vx(), track.vy(), track.vz());
   TEveTrack* eveTrack = new TEveTrack( &ts, m_trackerPropagator);
   
   // path-marks from a trajectory
   std::vector<TrajectoryMeasurement> measurements = traj.measurements();
   std::vector<TrajectoryMeasurement>::iterator measurements_it = measurements.begin();
   std::vector<TrajectoryMeasurement>::reverse_iterator measurements_rit = measurements.rbegin();
   for (size_t t=0; t != measurements.size(); ++t, ++measurements_it, ++measurements_rit)
   {
      TrajectoryStateOnSurface trajState = (traj.direction() == alongMomentum) ? measurements_it->updatedState() : measurements_rit->updatedState() ;

      if( !trajState.isValid() ) continue;

      eveTrack->AddPathMark( TEvePathMark( TEvePathMark::kReference,
                                           TEveVector(trajState.globalPosition().x(),trajState.globalPosition().y(), trajState.globalPosition().z()),
                                           TEveVector(trajState.globalMomentum().x(),trajState.globalMomentum().y(), trajState.globalMomentum().z())));
   }

   eveTrack->MakeTrack();         
   setupAddElement(eveTrack, comp);      
}

bool FWTrackProxyBuilderFF::visibilityModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                             FWViewType::EType viewType, const FWViewContext* vc)
{
   const FWEventItem::ModelInfo& info = iId.item()->modelInfo(iId.index());
   bool returnValue = false;
   if (info.displayProperties().isVisible() && iCompound->NumChildren()==0 && m_trajToTrackMap)
   {
      TrajTrackAssociationCollection::const_iterator it = m_trajToTrackMap->begin();
      std::advance(it, iId.index());
      buildTrack(it, (TEveCompound*)iCompound);
      returnValue=true;
   }
   return returnValue;
}

REGISTER_FWPROXYBUILDER(FWTrackProxyBuilderFF, reco::TrackCollection, "TracksFF", FWViewType::kAll3DBits | FWViewType::kAllRPZBits);


