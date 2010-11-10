#ifndef _FWPFECALRECHITRPPROXYBUILDER_H_
#define _FWPFECALRECHITRPPROXYBUILDER_H_

#include "TEveCompound.h"

// User include files
#include "Fireworks/Core/interface/FWProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "Fireworks/ParticleFlow/plugins/FWPFRhoPhiRecHit.h"


//-----------------------------------------------------------------------------
// FWPFEcalRecHitRPProxyBuilder
//-----------------------------------------------------------------------------
class FWPFEcalRecHitRPProxyBuilder : public FWProxyBuilderTemplate<EcalRecHit>
{
   private:
      FWPFEcalRecHitRPProxyBuilder( const FWPFEcalRecHitRPProxyBuilder& );               // Stop default
      const FWPFEcalRecHitRPProxyBuilder& operator=( const FWPFEcalRecHitRPProxyBuilder& );   // Stop default

      TEveVector calculateCentre( const float *corners );

      std::vector<FWPFRhoPhiRecHit*> towers;

   public:
      static std::string typeOfBuilder() { return "simple#"; }

      FWPFEcalRecHitRPProxyBuilder(){}
      virtual ~FWPFEcalRecHitRPProxyBuilder(){}

      virtual void build( const FWEventItem *iItem, TEveElementList *product, const FWViewContext* );

      virtual bool havePerViewProduct( FWViewType::EType ) const { return true; }
      virtual void scaleProduct( TEveElementList *parent, FWViewType::EType, const FWViewContext *vc );
      virtual void cleanLocal() { towers.clear(); }   // This needs to also take care of all children

      REGISTER_PROXYBUILDER_METHODS();
};
#endif
