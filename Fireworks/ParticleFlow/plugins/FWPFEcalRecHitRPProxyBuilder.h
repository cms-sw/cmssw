#ifndef _FWPFECALRECHITRPPROXYBUILDER_H_
#define _FWPFECALRECHITRPPROXYBUILDER_H_

#include "TEveCompound.h"

// User include files
#include "Fireworks/Core/interface/FWProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/Context.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "Fireworks/ParticleFlow/plugins/FWPFRhoPhiRecHit.h"

#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"


//-----------------------------------------------------------------------------
// FWPFEcalRecHitRPProxyBuilder
//-----------------------------------------------------------------------------
class FWPFEcalRecHitRPProxyBuilder : public FWProxyBuilderTemplate<EcalRecHit>
{
   public:
      static std::string typeOfBuilder() { return "simple#"; }

   // -------------------- Constructor(s)/Destructors --------------------------
      FWPFEcalRecHitRPProxyBuilder(){}
      virtual ~FWPFEcalRecHitRPProxyBuilder(){}

      virtual void build( const FWEventItem *iItem, TEveElementList *product, const FWViewContext* );

      virtual bool havePerViewProduct( FWViewType::EType ) const { return true; }
      virtual void scaleProduct( TEveElementList *parent, FWViewType::EType, const FWViewContext *vc );
      virtual void cleanLocal();

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFEcalRecHitRPProxyBuilder( const FWPFEcalRecHitRPProxyBuilder& );                    // Stop default
      const FWPFEcalRecHitRPProxyBuilder& operator=( const FWPFEcalRecHitRPProxyBuilder& );   // Stop default

   // ------------------------- Member Functions -------------------------------
      TEveVector calculateCentre( const float *corners );
      float      calculateEt( const TEveVector &centre, float E );

   // --------------------------- Data Members ---------------------------------
      std::vector<FWPFRhoPhiRecHit*> m_towers;
};
#endif
