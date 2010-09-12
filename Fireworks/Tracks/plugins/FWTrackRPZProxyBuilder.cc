// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWTrackRPZProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Nov 25 14:42:13 EST 2008
// $Id: FWTrackRPZProxyBuilder.cc,v 1.13 2010/01/21 21:02:13 amraktad Exp $
//

// system include files
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "DataFormats/TrackReco/interface/Track.h"

// user include files
#include "Fireworks/Tracks/interface/estimate_field.h"
#include "Fireworks/Core/interface/FWRPZSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/register_dataproxybuilder_macro.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilderBaseFactory.h"
#include "Fireworks/Core/interface/FWMagField.h"

#include "Fireworks/Tracks/interface/TrackUtils.h"


class FWTrackRPZProxyBuilder : public FWRPZSimpleProxyBuilderTemplate<reco::Track> {

public:
   FWTrackRPZProxyBuilder();
   virtual ~FWTrackRPZProxyBuilder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWTrackRPZProxyBuilder(const FWTrackRPZProxyBuilder&); // stop default

   const FWTrackRPZProxyBuilder& operator=(const FWTrackRPZProxyBuilder&); // stop default

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
FWTrackRPZProxyBuilder::FWTrackRPZProxyBuilder() :
   m_trackerPropagator( new TEveTrackPropagator)
{
   m_trackerPropagator->IncRefCount();
   m_trackerPropagator->IncDenyDestroy();
   m_trackerPropagator->SetMaxR( 850 );
   m_trackerPropagator->SetMaxZ( 1100 );
   m_trackerPropagator->SetMaxStep(1);
}

FWTrackRPZProxyBuilder::~FWTrackRPZProxyBuilder()
{
   m_trackerPropagator->DecRefCount();
}

void
FWTrackRPZProxyBuilder::build(const reco::Track& iData, unsigned int iIndex,TEveElement& oItemHolder) const
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
REGISTER_FWRPZDATAPROXYBUILDERBASE(FWTrackRPZProxyBuilder,reco::Track,"Tracks");
