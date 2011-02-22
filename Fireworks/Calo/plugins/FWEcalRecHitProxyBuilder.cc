/*
 *  FWEcalRecHitProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 5/28/10.
 *  Copyright 2010 FNAL. All rights reserved.
 *
 */
#include "TEveDigitSet.h"
#include "Fireworks/Core/interface/FWDigitSetProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

class FWEcalRecHitProxyBuilder : public FWDigitSetProxyBuilder
{
public:
   FWEcalRecHitProxyBuilder() {}
   virtual ~FWEcalRecHitProxyBuilder() {}
	
   REGISTER_PROXYBUILDER_METHODS();
	
private:
   FWEcalRecHitProxyBuilder( const FWEcalRecHitProxyBuilder& );
   const FWEcalRecHitProxyBuilder& operator=( const FWEcalRecHitProxyBuilder& );

   virtual void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* );	
};

//______________________________________________________________________________

void FWEcalRecHitProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*)
{
   const EcalRecHitCollection* collection = 0;
   iItem->get( collection );
   if (! collection)
      return;

   TEveBoxSet* boxSet = addBoxSetToProduct(product);
   for (std::vector<EcalRecHit>::const_iterator it = collection->begin() ; it != collection->end(); ++it)
   {
      const float* corners = item()->getGeom()->getCorners((*it).detid());
      if (corners == 0) 
         continue;

      Float_t scale = 10.0;
      bool reflect = false;
      if (EcalSubdetector( (*it).detid().subdetId() ) == EcalPreshower)
      {
         scale = 1000.0; 	// FIXME: The scale should be taken form somewhere else
         reflect = corners[2] < 0;
      }

      std::vector<float> scaledCorners(24);
      fireworks::energyTower3DCorners(corners, (*it).energy() * scale,  scaledCorners, reflect);

      addBox(boxSet, &scaledCorners[0]);
   }
}

REGISTER_FWPROXYBUILDER( FWEcalRecHitProxyBuilder, EcalRecHitCollection, "Ecal RecHit", FWViewType::kISpyBit );
