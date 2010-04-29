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
// $Id: FWTrackProxyBuilder.cc,v 1.6 2010/04/21 11:10:08 amraktad Exp $
//

// system include files
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWMagField.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/TrackReco/interface/Track.h"

class FWTrackProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Track> {

public:
   FWTrackProxyBuilder();
   virtual ~FWTrackProxyBuilder();

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWTrackProxyBuilder(const FWTrackProxyBuilder&); // stop default

   const FWTrackProxyBuilder& operator=(const FWTrackProxyBuilder&); // stop default

   void build(const reco::Track& iData, unsigned int iIndex,TEveElement& oItemHolder);

   FWEvePtr<TEveTrackPropagator> m_trackerPropagator;
};

FWTrackProxyBuilder::FWTrackProxyBuilder()
   : m_trackerPropagator( new TEveTrackPropagator )
{
   m_trackerPropagator->IncRefCount();
   m_trackerPropagator->IncDenyDestroy();
   m_trackerPropagator->SetMaxR( 850 );
   m_trackerPropagator->SetMaxZ( 1100 );
   m_trackerPropagator->SetDelta(0.01);
}

FWTrackProxyBuilder::~FWTrackProxyBuilder()
{
   m_trackerPropagator->DecRefCount();
}

void
FWTrackProxyBuilder::build( const reco::Track& iData, unsigned int iIndex,TEveElement& oItemHolder ) 
{
   if( context().getField()->getAutodetect() ) {
      if( fabs( iData.eta() ) < 2.0 && iData.pt() > 0.5 && iData.pt() < 30 ) {
	 double estimate = fireworks::estimateField(iData,true);
         if( estimate >= 0 ) context().getField()->guessField(estimate);
      }
   }

   // workaround for missing GetFieldObj() in TEveTrackPropagator, default stepper is kHelix
   if( m_trackerPropagator->GetStepper() == TEveTrackPropagator::kHelix ) {
      m_trackerPropagator->SetStepper( TEveTrackPropagator::kRungeKutta );
      m_trackerPropagator->SetMagFieldObj( context().getField(), false);
   }

   TEveTrackPropagator* propagator = ( !iData.extra().isAvailable() ) ? m_trackerPropagator.get() : context().getTrackPropagator();

   TEveTrack* trk = fireworks::prepareTrack( iData, propagator );
   trk->MakeTrack();
   setupAddElement(trk, &oItemHolder);
}

//
// static member functions
//
REGISTER_FWPROXYBUILDER(FWTrackProxyBuilder, reco::Track, "Tracks", FWViewType::kAll3DBits | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
