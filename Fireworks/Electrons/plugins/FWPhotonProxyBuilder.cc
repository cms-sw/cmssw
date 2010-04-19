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
// $Id: FWPhotonProxyBuilder.cc,v 1.2 2010/04/19 08:20:15 yana Exp $
//
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWViewType.h"

#include "Fireworks/Electrons/interface/makeSuperCluster.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

class FWPhotonProxyBuilder : public FWProxyBuilderBase {

public:
   FWPhotonProxyBuilder() {}

   virtual ~FWPhotonProxyBuilder() {}

   virtual bool hasSingleProduct() { return false; }

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPhotonProxyBuilder(const FWPhotonProxyBuilder&); // stop default
   const FWPhotonProxyBuilder& operator=(const FWPhotonProxyBuilder&); // stop default

   virtual void buildViewType( const FWEventItem* iItem, TEveElementList* product, FWViewType::EType type );
};

void
FWPhotonProxyBuilder::buildViewType( const FWEventItem* iItem, TEveElementList* product, FWViewType::EType type )
{
   reco::PhotonCollection const * photons = 0;
   iItem->get( photons );
   if( photons == 0 ) return;

   Int_t idx = 0;
   for( reco::PhotonCollection::const_iterator it = photons->begin(), itEnd = photons->end(); it != itEnd; ++it, ++idx)
   { 
      const char* name = Form( "Photon %d", idx );
      TEveElementList* comp = new TEveElementList( name, name );
      if( type == FWViewType::kRhoPhi )
	 fireworks::makeRhoPhiSuperCluster(*item(),
					   (*it).superCluster(),
					   (*it).phi(),
					   *comp);
      else if( type == FWViewType::kRhoZ )
         fireworks::makeRhoZSuperCluster(*item(),
                                         (*it).superCluster(),
                                         (*it).phi(),
                                         *comp);
      product->AddElement(comp);
   }
}

REGISTER_FWPROXYBUILDER( FWPhotonProxyBuilder, reco::PhotonCollection, "Photons", FWViewType::kAllRPZBits );
