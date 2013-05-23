#ifndef _FWPFECALRECHITRPPROXYBUILDER_H_
#define _FWPFECALRECHITRPPROXYBUILDER_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFEcalRecHitRPProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//

// System include files
#include "TEveCompound.h"

// User include files
#include "Fireworks/Core/interface/FWProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/Context.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "Fireworks/ParticleFlow/plugins/FWPFRhoPhiRecHit.h"
#include "Fireworks/ParticleFlow/interface/FWPFUtils.h"

#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"

//-----------------------------------------------------------------------------
// FWPFEcalRecHitRPProxyBuilder
//-----------------------------------------------------------------------------
class FWPFEcalRecHitRPProxyBuilder : public FWProxyBuilderTemplate<EcalRecHit>
{
   public:
      static std::string typeOfBuilder() { return "simple#"; }

   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFEcalRecHitRPProxyBuilder(){ m_pfUtils = new FWPFUtils(); }
      virtual ~FWPFEcalRecHitRPProxyBuilder(){ delete m_pfUtils; }

   // --------------------- Member Functions --------------------------
      virtual void build( const FWEventItem *iItem, TEveElementList *product, const FWViewContext* );

      virtual bool havePerViewProduct( FWViewType::EType ) const { return true; }
      virtual void scaleProduct( TEveElementList *parent, FWViewType::EType, const FWViewContext *vc );
      virtual void cleanLocal();

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFEcalRecHitRPProxyBuilder( const FWPFEcalRecHitRPProxyBuilder& );                    // Stop default
      const FWPFEcalRecHitRPProxyBuilder& operator=( const FWPFEcalRecHitRPProxyBuilder& );   // Stop default

   // --------------------- Member Functions --------------------------
      TEveVector calculateCentre( const float *corners );
      float      calculateEt( const TEveVector &centre, float E );

   // ----------------------- Data Members ----------------------------
      std::vector<FWPFRhoPhiRecHit*> m_towers;
      FWPFUtils *m_pfUtils;
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
