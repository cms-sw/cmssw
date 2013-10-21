
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

#include "DataFormats/DetId/interface/DetId.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Common/interface/EventBase.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/interface/FWParameters.h"

#include "TEveTrack.h"
#include "TEveCompound.h"
// #include <boost/exception.hpp>

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
      event->getByLabel("simHitTPAssocProducer", simHitsTPAssoc);
   }
   catch (const std::exception& e) {
      std::cout << "===== ERROR #1 " << e.what() <<  std::endl;
   }   

   if (simHitsTPAssoc.isValid()) {
      m_assocList = &*simHitsTPAssoc;
   }
}
//______________________________________________________________________________


void
FWTrackingParticleProxyBuilderFF::getAssocList2()
{ 
   edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssoc; 
   const edm::Event* event = (const edm::Event*)item()->getEvent();
 
   try {
      edm::ParameterSet confProcess;
      bool ok = event->getProcessParameterSet("DISPLAY", confProcess);
      printf("OK for DISPLAY process PS [%d], dump: \n", ok);
      // confProcess.dump();

     

      // now try to get the simhit ParameterSet
      const edm::ParameterSet& conf_ = confProcess.getParameterSet("xxx");
      // conf_.dump();
      
      const std::vector<std::string> pnames =  conf_.getParameterNames();
      for (std::vector<std::string>::const_iterator it = pnames.begin(); it != pnames.end(); ++it)
         std::cout << *it << std::endl;

      edm::InputTag _simHitTpMapTag(conf_.getParameter<edm::InputTag>("g4SimHits"));
      event->getByLabel( _simHitTpMapTag, simHitsTPAssoc);

   }
   catch (const std::exception& e) {
      std::cout << "======= ERROR #2 " << e.what() <<  std::endl;
   }

   if (simHitsTPAssoc.isValid()) {
      m_assocList = &*simHitsTPAssoc;
   }
   else {
      printf("HAVCE LIST SIZE %d \n", (int)simHitsTPAssoc->size() );
   }
}
//______________________________________________________________________________


void
FWTrackingParticleProxyBuilderFF::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext* vc)
{
   //getAssocList1();
   if (!m_assocList)
      getAssocList2();

   
   const TrackingParticleCollection * tracks = 0;
   iItem->get( tracks );
   for (TrackingParticleCollection::const_iterator it = tracks->begin(); it != tracks->end(); ++it)
   {
      TEveCompound* comp = createCompound();
      setupAddElement( comp, product );
      continue;

      const TrackingParticle& iData = *it;
      TEveRecTrack t;
      t.fBeta = 1.0;
      t.fP = TEveVector( iData.px(), iData.py(), iData.pz() );
      t.fV = TEveVector( iData.vx(), iData.vy(), iData.vz() );
      t.fSign = iData.charge();
  
      TEveTrack* track = new TEveTrack(&t, context().getTrackPropagator());
      if( t.fSign == 0 )
         track->SetLineStyle( 7 );
      track->MakeTrack();
      setupAddElement( track, comp );
   }

}

REGISTER_FWPROXYBUILDER( FWTrackingParticleProxyBuilderFF, TrackingParticleCollection, "TrackingParticlesFF", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
