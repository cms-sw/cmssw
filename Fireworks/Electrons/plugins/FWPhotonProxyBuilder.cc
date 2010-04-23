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
// $Id: FWPhotonProxyBuilder.cc,v 1.5 2010/04/21 10:39:25 amraktad Exp $
//
#include "TEveCompound.h"

#include "Fireworks/Core/interface/FWProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWViewType.h"

#include "Fireworks/Electrons/interface/makeSuperCluster.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"


class FWPhotonProxyBuilder : public FWProxyBuilderTemplate<reco::Photon> {

public:
   FWPhotonProxyBuilder() {}

   virtual ~FWPhotonProxyBuilder() {}

   virtual bool haveSingleProduct() const { return false; }

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPhotonProxyBuilder(const FWPhotonProxyBuilder&); // stop default
   const FWPhotonProxyBuilder& operator=(const FWPhotonProxyBuilder&); // stop default

   virtual void buildViewType( const FWEventItem* iItem, TEveElementList* product, FWViewType::EType type );
};

void
FWPhotonProxyBuilder::buildViewType( const FWEventItem* iItem, TEveElementList* product, FWViewType::EType type )
{ 
   for (int i = 0; i < static_cast<int>(iItem->size()); ++i)
   {    
      const reco::Photon &photon = modelData(i);
      TEveCompound* comp = createCompound();

      if( type == FWViewType::kRhoPhi )
	 fireworks::makeRhoPhiSuperCluster(this,
					   photon.superCluster(),
					   photon.phi(),
					   *comp);
      else if( type == FWViewType::kRhoZ )
         fireworks::makeRhoZSuperCluster(this,
                                         photon.superCluster(),
                                         photon.phi(),
                                         *comp);
      setupAddElement(comp, product);
   }
}

REGISTER_FWPROXYBUILDER( FWPhotonProxyBuilder, reco::PhotonCollection, "Photons", FWViewType::kAllRPZBits );
