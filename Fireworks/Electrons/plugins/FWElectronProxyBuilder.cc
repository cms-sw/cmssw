// -*- C++ -*-
//
// Package:     Electrons
// Class  :     FWElectronProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 14:17:03 EST 2008
// $Id: FWElectronProxyBuilder.cc,v 1.3 2010/04/14 11:53:41 yana Exp $
//
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWEveScalableStraightLineSet.h"

#include "Fireworks/Candidates/interface/CandidateUtils.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include "TEveTrack.h"

class FWElectronProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::GsfElectron> {

public:
   FWElectronProxyBuilder() {}
   virtual ~FWElectronProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWElectronProxyBuilder(const FWElectronProxyBuilder&); // stop default

   const FWElectronProxyBuilder& operator=(const FWElectronProxyBuilder&); // stop default

   virtual void build(const reco::GsfElectron& iData, unsigned int iIndex, TEveElement& oItemHolder) const;
};

void
FWElectronProxyBuilder::build(const reco::GsfElectron& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
   TEveTrack* track(0);
   if( iData.gsfTrack().isAvailable() )
      track = fireworks::prepareTrack( *(iData.gsfTrack()),
				       context().getTrackPropagator());
   else
      track = fireworks::prepareCandidate( iData,
					   context().getTrackPropagator());
   track->MakeTrack();
   oItemHolder.AddElement( track );
}

class FWElectronGlimpseProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::GsfElectron> {

public:
   FWElectronGlimpseProxyBuilder() {}
   virtual ~FWElectronGlimpseProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWElectronGlimpseProxyBuilder(const FWElectronGlimpseProxyBuilder&); // stop default

   const FWElectronGlimpseProxyBuilder& operator=(const FWElectronGlimpseProxyBuilder&); // stop default

   virtual void build(const reco::GsfElectron& iData, unsigned int iIndex, TEveElement& oItemHolder) const;
};

void
FWElectronGlimpseProxyBuilder::build( const reco::GsfElectron& iData, unsigned int iIndex,TEveElement& oItemHolder ) const
{
   FWEveScalableStraightLineSet* marker = new FWEveScalableStraightLineSet("", "");
   marker->SetLineWidth(2);
   fireworks::addStraightLineSegment( marker, &iData, 1.0 );
   oItemHolder.AddElement(marker);
   //add to scaler at end so that it can scale the line after all ends have been added
   // FIXME: It's not a part of a standard FWSimpleProxyBuilderTemplate: the scaler is not set!
//    assert(scaler());
//    scaler()->addElement(marker);
}

REGISTER_FWPROXYBUILDER(FWElectronProxyBuilder, reco::GsfElectron, "Electrons", FWViewType::k3DBit | FWViewType::kRPZBit);
REGISTER_FWPROXYBUILDER(FWElectronGlimpseProxyBuilder, reco::GsfElectron, "Electrons", FWViewType::kGlimpseBit);
