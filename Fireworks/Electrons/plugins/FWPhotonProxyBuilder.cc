// -*- C++ -*-
//
// Package:     Photons
// Class  :     FWPhotonProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Nov 26 14:52:01 EST 2008
// $Id: FWPhotonProxyBuilder.cc,v 1.6 2010/04/23 21:02:00 amraktad Exp $
//
#include "TEveCompound.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWViewType.h"

#include "Fireworks/Electrons/interface/makeSuperCluster.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"


class FWPhotonProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Photon> {

public:
   FWPhotonProxyBuilder() {}

   virtual ~FWPhotonProxyBuilder() {}

   virtual bool haveSingleProduct() const { return false; }

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPhotonProxyBuilder(const FWPhotonProxyBuilder&); // stop default
   const FWPhotonProxyBuilder& operator=(const FWPhotonProxyBuilder&); // stop default

   virtual void buildViewType(const reco::Photon& iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type );
};

void
FWPhotonProxyBuilder::buildViewType(const reco::Photon& photon, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type )
{ 
   if( type == FWViewType::kRhoPhi )
      fireworks::makeRhoPhiSuperCluster(this,
                                        photon.superCluster(),
                                        photon.phi(),
                                        oItemHolder);
   else if( type == FWViewType::kRhoZ )
      fireworks::makeRhoZSuperCluster(this,
                                      photon.superCluster(),
                                      photon.phi(),
                                      oItemHolder);
}

REGISTER_FWPROXYBUILDER( FWPhotonProxyBuilder, reco::Photon, "Photons", FWViewType::kAllRPZBits );
