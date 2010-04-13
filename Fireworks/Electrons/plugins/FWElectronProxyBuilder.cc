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
// $Id: FWElectronProxyBuilder.cc,v 1.7 2010/03/04 21:19:03 chrjones Exp $
//
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "Fireworks/Candidates/interface/CandidateUtils.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Electrons/interface/makeSuperCluster.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include "TEveTrack.h"

class TEveTrackPropagator;

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
   if ( iData.gsfTrack().isAvailable() )
      track = fireworks::prepareTrack( *(iData.gsfTrack()),
				       context().getTrackPropagator(),
				       item()->defaultDisplayProperties().color() );
   else
      track = fireworks::prepareCandidate( iData,
					   context().getTrackPropagator(),
					   item()->defaultDisplayProperties().color() );
   track->MakeTrack();
   oItemHolder.AddElement( track );
}

class FWElectronRhoPhiProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::GsfElectron> {

public:
   FWElectronRhoPhiProxyBuilder() {}
   virtual ~FWElectronRhoPhiProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWElectronRhoPhiProxyBuilder(const FWElectronRhoPhiProxyBuilder&); // stop default

   const FWElectronRhoPhiProxyBuilder& operator=(const FWElectronRhoPhiProxyBuilder&); // stop default

   virtual void build(const reco::GsfElectron& iData, unsigned int iIndex, TEveElement& oItemHolder) const;
};

void
FWElectronRhoPhiProxyBuilder::build(const reco::GsfElectron& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
   fireworks::makeRhoPhiSuperCluster(*item(),
                                     iData.superCluster(),
				     iData.phi(),
				     oItemHolder);
   TEveTrack* track(0);
   if ( iData.gsfTrack().isAvailable() )
      track = fireworks::prepareTrack( *(iData.gsfTrack()),
				       context().getTrackPropagator(),
				       item()->defaultDisplayProperties().color() );
   else
      track = fireworks::prepareCandidate( iData,
					   context().getTrackPropagator(),
					   item()->defaultDisplayProperties().color() );
   track->MakeTrack();
   oItemHolder.AddElement( track );
}

class FWElectronRhoZProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::GsfElectron> {

public:
   FWElectronRhoZProxyBuilder() {}
   virtual ~FWElectronRhoZProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWElectronRhoZProxyBuilder(const FWElectronRhoZProxyBuilder&); // stop default

   const FWElectronRhoZProxyBuilder& operator=(const FWElectronRhoZProxyBuilder&); // stop default

   virtual void build(const reco::GsfElectron& iData, unsigned int iIndex, TEveElement& oItemHolder) const;
};

void
FWElectronRhoZProxyBuilder::build(const reco::GsfElectron& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
   fireworks::makeRhoZSuperCluster(*item(),
                                   iData.superCluster(),
				   iData.phi(),
				   oItemHolder);
   TEveTrack* track(0);
   if ( iData.gsfTrack().isAvailable() )
      track = fireworks::prepareTrack( *(iData.gsfTrack()),
				       context().getTrackPropagator(),
				       item()->defaultDisplayProperties().color() );
   else
      track = fireworks::prepareCandidate( iData,
					   context().getTrackPropagator(),
					   item()->defaultDisplayProperties().color() );
   track->MakeTrack();
   oItemHolder.AddElement( track );
}

REGISTER_FWPROXYBUILDER(FWElectronProxyBuilder, reco::GsfElectron, "Electrons", FWViewType::k3DBit);
REGISTER_FWPROXYBUILDER(FWElectronRhoPhiProxyBuilder, reco::GsfElectron, "Electrons", FWViewType::kRhoPhiBit);
REGISTER_FWPROXYBUILDER(FWElectronRhoZProxyBuilder, reco::GsfElectron, "Electrons", FWViewType::kRhoZBit);
