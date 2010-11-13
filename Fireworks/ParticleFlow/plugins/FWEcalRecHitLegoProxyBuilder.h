#ifndef _FWECALRECHITLEGOPROXYBUILDER_H_
#define _FWECALRECHITLEGOPROXYBUILDER_H_

//
// Package:           Particle Flow
// Class:              FWEcalRecHitLegoProxyBuilder
// Original Author:      Simon Harris
// $Id: FWEcalRecHitLegoProxyBuilder.h,v 1.2 2010/09/15 18:38:09 amraktad Exp $
//

#include "Fireworks/Core/interface/FWProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/Context.h"
#include "TEveCompound.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "Fireworks/ParticleFlow/plugins/LegoRecHit.h"



class FWEcalRecHitLegoProxyBuilder : public FWProxyBuilderTemplate<EcalRecHit>
{
private:
   // Disable default copy constructor
   FWEcalRecHitLegoProxyBuilder( const FWEcalRecHitLegoProxyBuilder& );
   // Disable default assignment operator
   const FWEcalRecHitLegoProxyBuilder& operator=( const FWEcalRecHitLegoProxyBuilder& );

 /*************************************************************\
(                       MEMBER FUNCTIONS                        )
 \*************************************************************/

   float calculateET( const std::vector<TEveVector> &corners, float E );

 /***************************************************************\
(                   CONSTRUCTOR(S)/DESTRUCTOR                     )
 \***************************************************************/
public:
   FWEcalRecHitLegoProxyBuilder(){}
   virtual ~FWEcalRecHitLegoProxyBuilder(){}

      static std::string typeOfBuilder() { return "simple#"; }

     virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);
   virtual void localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                  FWViewType::EType viewType, const FWViewContext* vc);

   REGISTER_PROXYBUILDER_METHODS();

};
#endif
