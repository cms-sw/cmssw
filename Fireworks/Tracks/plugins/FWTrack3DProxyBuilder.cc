// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWTrack3DProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Nov 25 14:42:13 EST 2008
// $Id: FWTrack3DProxyBuilder.cc,v 1.12 2009/12/11 21:18:45 dmytro Exp $
//

// system include files
#include "TEveTrack.h"
#define protected public
#include "TEveTrackPropagator.h"
#undef protected

// user include files
#include "Fireworks/Core/interface/register_dataproxybuilder_macro.h"
#include "Fireworks/Core/interface/FW3DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWMagField.h"
#include "Fireworks/Tracks/interface/estimate_field.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/TrackReco/interface/Track.h"

class FWTrack3DProxyBuilder : public FW3DSimpleProxyBuilderTemplate<reco::Track> {

public:
   FWTrack3DProxyBuilder();
   virtual ~FWTrack3DProxyBuilder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWTrack3DProxyBuilder(const FWTrack3DProxyBuilder&); // stop default

   const FWTrack3DProxyBuilder& operator=(const FWTrack3DProxyBuilder&); // stop default

   void build(const reco::Track& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
   // ---------- member data --------------------------------

   FWEvePtr<TEveTrackPropagator> m_trackerPropagator;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWTrack3DProxyBuilder::FWTrack3DProxyBuilder():
   m_trackerPropagator( new TEveTrackPropagator)
{
   m_trackerPropagator->IncRefCount();
   m_trackerPropagator->IncDenyDestroy();
   m_trackerPropagator->SetMaxR( 850 );
   m_trackerPropagator->SetMaxZ( 1100 );
   m_trackerPropagator->SetMaxStep(1);
}

FWTrack3DProxyBuilder::~FWTrack3DProxyBuilder()
{
   m_trackerPropagator->DecRefCount();
}

void
FWTrack3DProxyBuilder::build(const reco::Track& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   if(context().getField()->getAutodetect()) {
      if ( fabs( iData.eta() ) < 2.0 && iData.pt() > 0.5 && iData.pt() < 30 ) {
         double estimate = fw::estimate_field(iData,true);
         if ( estimate >= 0 ) context().getField()->guessField(estimate);
      }
   }

   // workaround for missing GetFieldObj() in TEveTrackPropagator, default stepper is kHelix
   if (m_trackerPropagator->GetStepper() == TEveTrackPropagator::kHelix) {
      m_trackerPropagator->SetStepper(TEveTrackPropagator::kRungeKutta);
      m_trackerPropagator->SetMagFieldObj(context().getField());
   }

   TEveTrackPropagator* propagator =   (!iData.extra().isAvailable()) ? m_trackerPropagator.get() : context().getTrackPropagator();

   TEveTrack* trk = fireworks::prepareTrack( iData, propagator, item()->defaultDisplayProperties().color() );
   trk->MakeTrack();
   oItemHolder.AddElement( trk );
}

//
// static member functions
//
REGISTER_FW3DDATAPROXYBUILDER(FWTrack3DProxyBuilder,reco::Track,"Tracks");
