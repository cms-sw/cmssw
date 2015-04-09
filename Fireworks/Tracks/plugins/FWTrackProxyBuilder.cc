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
//

// system include files
#include "TEveTrack.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/interface/FWMagField.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Tracks/interface/estimate_field.h"

#include "DataFormats/TrackReco/interface/Track.h"

class FWTrackProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Track> {

public:
   FWTrackProxyBuilder();
   virtual ~FWTrackProxyBuilder();

   REGISTER_PROXYBUILDER_METHODS();
  
   virtual void setItem(const FWEventItem* iItem);
private:
   FWTrackProxyBuilder(const FWTrackProxyBuilder&); // stop default

   const FWTrackProxyBuilder& operator=(const FWTrackProxyBuilder&); // stop default

   using FWSimpleProxyBuilderTemplate<reco::Track>::build;
   void build(const reco::Track& iData, unsigned int iIndex,TEveElement& oItemHolder, const FWViewContext*) override;
};

FWTrackProxyBuilder::FWTrackProxyBuilder()
{
}

FWTrackProxyBuilder::~FWTrackProxyBuilder()
{
}

void
FWTrackProxyBuilder::setItem(const FWEventItem* iItem)
{
   FWProxyBuilderBase::setItem(iItem);
   
   if (iItem) {
      iItem->getConfig()->assertParam("LineWidth", long(1), long(1), long(4));
   }
}

void
FWTrackProxyBuilder::build( const reco::Track& iData, unsigned int iIndex,TEveElement& oItemHolder , const FWViewContext*) 
{
   if( context().getField()->getSource() == FWMagField::kNone ) {
      if( fabs( iData.eta() ) < 2.0 && iData.pt() > 0.5 && iData.pt() < 30 ) {
	 double estimate = fw::estimate_field( iData, true );
         if( estimate >= 0 ) context().getField()->guessField( estimate );
      }
   }

   TEveTrackPropagator* propagator = ( !iData.extra().isAvailable() ) ?  context().getTrackerTrackPropagator() : context().getTrackPropagator();

   TEveTrack* trk = fireworks::prepareTrack( iData, propagator );
   trk->MakeTrack();

   // Line width can be cached as a member. Set in virtual builder::itemChanged()
   int width = item()->getConfig()->value<long>("LineWidth");
   trk->SetLineWidth(width);

   setupAddElement(trk, &oItemHolder);
}

//
// static member functions
//
REGISTER_FWPROXYBUILDER(FWTrackProxyBuilder, reco::Track, "Tracks", FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
