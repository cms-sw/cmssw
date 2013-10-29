
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

   void getAssocList1();
   void getAssocList2();
};

//______________________________________________________________________________



void
FWTrackingParticleProxyBuilderFF::getAssocList1()
{
   edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssoc; 
   const edm::Event* event = (const edm::Event*)item()->getEvent();
   try {
      event->getByLabel("xxx", simHitsTPAssoc);
   }
   catch (const std::exception& e) {
      std::cout << "===== ERROR #1 " << e.what() <<  std::endl;
   }   

   if (simHitsTPAssoc.isValid()) {
      m_assocList = &*simHitsTPAssoc;
   }
   //else 
   {
      printf("HAVE ASSOC LIST SIZE %d \n", (int)simHitsTPAssoc->size() );
   }
}
//______________________________________________________________________________


void
FWTrackingParticleProxyBuilderFF::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext* vc)
{
   const edm::Event* event = (const edm::Event*)item()->getEvent();
   if (item()->getEvent() == 0 ) {
      return;
   }

   getAssocList1();
   fflush(stdout);

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

      /*
        TrackingParticleRef tpr(tpch, tpIdx);
        std::pair<TrackingParticleRef, TrackPSimHitRef> clusterTPpairWithDummyTP(tpr,TrackPSimHitRef());//SimHit is dummy: for simHitTPAssociationListGreater 
        auto range = std::equal_range(m_assocList->begin(), m_assocList->end(), 
        clusterTPpairWithDummyTP, SimHitTPAssociationProducer::simHitTPAssociationListGreater);
      */

       int alistIdx = 0;
      for (SimHitTPAssociationProducer::SimHitTPAssociationList::const_iterator ai = m_assocList->begin(); ai != m_assocList->end(); ++ai, ++alistIdx)
      {
         const edm::PSimHitContainer* l = ai->second.product();
         const TrackingParticle* tp = ai->first.get();
         printf("assoc list entry %d has %d TrackingParticles in TrackingParticleRef \n", alistIdx, (int)ai->first.product()->size());
         if (&iData == tp) {
            printf("TrackingParticle[%d] matches [%d] associantion pair, the pair has %d hits\n",tpIdx, alistIdx, (int)l->size());
            for (edm::PSimHitContainer::const_iterator hi = l->begin(); hi != l->end(); ++hi)
            {
               const PSimHit& phit = (*hi);

               if (phit.trackId() != tpIdx)
                  continue;

               local[0] = phit.localPosition().x();
               local[1] = phit.localPosition().y();
               local[2] = phit.localPosition().z();
               localDir[0] = phit.momentumAtEntry().x();
               localDir[1] = phit.momentumAtEntry().y();
               localDir[2] = phit.momentumAtEntry().z();
               geom->localToGlobal( phit.detUnitId(), local, global );
               geom->localToGlobal( phit.detUnitId(), localDir, globalDir );
               pointSet->SetNextPoint( global[0], global[1], global[2] );
               track->AddPathMark( TEvePathMark( TEvePathMark::kReference, TEveVector( global[0], global[1], global[2] ),
                                                 TEveVector( globalDir[0], globalDir[1], globalDir[2] )));
            }
         }
      }

      track->MakeTrack();
   }

}

REGISTER_FWPROXYBUILDER( FWTrackingParticleProxyBuilderFF, TrackingParticleCollection, "TrackingParticlesFF", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
