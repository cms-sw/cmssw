
/*
 *  FWTrackingParticleProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 9/9/10.
 *
 */

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimGeneral/TrackingAnalysis/interface/SimHitTPAssociationProducer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "DataFormats/DetId/interface/DetId.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Common/interface/EventBase.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/interface/FWParameters.h"

#include "TEveTrack.h"
#include "TEveCompound.h"
#include "TEveManager.h"
#include "TEveBrowser.h"
#include "TEveTrackPropagator.h"


class FWTrackingParticleProxyBuilderFF : public FWProxyBuilderBase
{
public:

   FWTrackingParticleProxyBuilderFF( void ):m_assocList(0) {} 
   virtual ~FWTrackingParticleProxyBuilderFF( void ) {}

   void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) override;
  
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWTrackingParticleProxyBuilderFF( const FWTrackingParticleProxyBuilderFF& );
   const FWTrackingParticleProxyBuilderFF& operator=( const FWTrackingParticleProxyBuilderFF& );


   const SimHitTPAssociationProducer::SimHitTPAssociationList* m_assocList;

   void getAssocList();
};

//______________________________________________________________________________



void
FWTrackingParticleProxyBuilderFF::getAssocList()
{
   edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssoc; 
   const edm::Event* event = (const edm::Event*)item()->getEvent();

   // AMT todo: check if there is any other way getting the list other than this
   //           ifnot, set proces name as a configurable parameter
   try {
      event->getByLabel("xxx", simHitsTPAssoc);
   }
   catch (const std::exception& e) {
      std::cerr << " Can't get asociation list " << e.what() <<  std::endl;
      return;
   }   

      m_assocList = &*simHitsTPAssoc;
}
//______________________________________________________________________________


void
FWTrackingParticleProxyBuilderFF::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext* vc)
{
   const edm::Event* event = (const edm::Event*)item()->getEvent();
   if (item()->getEvent() == 0 ) {
      return;
   }

   getAssocList();
   if (!m_assocList) return;

   gEve->GetBrowser()->MapWindow();
   
   const FWGeometry *geom = item()->getGeom();
   float local[3];
   float localDir[3];
   float global[3] = { 0.0, 0.0, 0.0 };
   float globalDir[3] = { 0.0, 0.0, 0.0 };


   context().getTrackPropagator()->SetRnrReferences(true);
   edm::Handle<TrackingParticleCollection> tpch;
   edm::InputTag coltag(item()->moduleLabel(), item()->productInstanceLabel(), item()->processName());
   event->getByLabel(coltag, tpch);

   unsigned int tpIdx = 0;
   for (TrackingParticleCollection::const_iterator it = tpch->begin(); it != tpch->end(); ++it, ++tpIdx) 
   {
      TEveCompound* comp = createCompound();
      setupAddElement( comp, product );
      if (iItem->modelInfo(tpIdx).displayProperties().isVisible() == false)
      {
         continue;
      }

      const TrackingParticle& iData = *it;
      TEveRecTrack t;
      t.fBeta = 1.0;
      t.fP = TEveVector( iData.px(), iData.py(), iData.pz() );
      t.fV = TEveVector( iData.vx(), iData.vy(), iData.vz() );
      t.fSign = iData.charge();

      TEveTrack* track = new TEveTrack(&t, context().getTrackPropagator());
      setupAddElement( track, comp );

      if( t.fSign == 0 )
         track->SetLineStyle( 7 );

      TEvePointSet* pointSet = new TEvePointSet;
      setupAddElement( pointSet, comp );


      TrackingParticleRef tpr(tpch, tpIdx);
      int alistIdx = 0;
      std::pair<TrackingParticleRef, TrackPSimHitRef> clusterTPpairWithDummyTP(tpr,TrackPSimHitRef());
      auto range = std::equal_range(m_assocList->begin(), m_assocList->end(), clusterTPpairWithDummyTP, SimHitTPAssociationProducer::simHitTPAssociationListGreater);      
      // printf("TrackingParticle[%d] matches %d hits\n",tpIdx,(int)(range.second-range.first ));
      for (SimHitTPAssociationProducer::SimHitTPAssociationList::const_iterator ai = range.first; ai != range.second; ++ai, ++alistIdx)
      {
               TrackPSimHitRef phit = ai->second;
               local[0] = phit->localPosition().x();
               local[1] = phit->localPosition().y();
               local[2] = phit->localPosition().z();
               localDir[0] = phit->momentumAtEntry().x();
               localDir[1] = phit->momentumAtEntry().y();
               localDir[2] = phit->momentumAtEntry().z();
               geom->localToGlobal( phit->detUnitId(), local, global );
               geom->localToGlobal( phit->detUnitId(), localDir, globalDir );
               pointSet->SetNextPoint( global[0], global[1], global[2] );
               track->AddPathMark( TEvePathMark( TEvePathMark::kReference, TEveVector( global[0], global[1], global[2] ),
                                                 TEveVector( globalDir[0], globalDir[1], globalDir[2] )));
      }


      track->MakeTrack();
   }

}

REGISTER_FWPROXYBUILDER( FWTrackingParticleProxyBuilderFF, TrackingParticleCollection, "TrackingParticlesFF", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
