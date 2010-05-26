// -*- C++ -*-
//
// Package:     Electrons
// Class  :     FWElectron3DProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 14:17:03 EST 2008
// $Id: FWElectron3DProxyBuilder.cc,v 1.5 2009/10/04 12:13:19 dmytro Exp $
//

#include "Fireworks/Core/interface/FW3DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"


class TEveTrack;
class TEveTrackPropagator;

class FWElectron3DProxyBuilder : public FW3DSimpleProxyBuilderTemplate<reco::GsfElectron> {

public:
   FWElectron3DProxyBuilder();
   virtual ~FWElectron3DProxyBuilder() {}

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWElectron3DProxyBuilder(const FWElectron3DProxyBuilder&); // stop default

   const FWElectron3DProxyBuilder& operator=(const FWElectron3DProxyBuilder&); // stop default

   virtual void build(const reco::GsfElectron& iData, unsigned int iIndex,TEveElement& oItemHolder) const;

   // ---------- member data --------------------------------
};

FWElectron3DProxyBuilder::FWElectron3DProxyBuilder()
{
}

//
// member functions
//
void
FWElectron3DProxyBuilder::build(const reco::GsfElectron& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   TEveTrack* track(0);
   if ( iData.gsfTrack().isAvailable() )
      track = fireworks::prepareTrack( *(iData.gsfTrack()),
                                       context().getTrackPropagator(),
                                       item()->defaultDisplayProperties().color() );
   else
      track = fireworks::prepareTrack( iData,
                                       context().getTrackPropagator(),
				       item()->defaultDisplayProperties().color() );
   track->MakeTrack();
   oItemHolder.AddElement( track );
}

REGISTER_FW3DDATAPROXYBUILDER(FWElectron3DProxyBuilder,std::vector<reco::GsfElectron>,"Electrons");
