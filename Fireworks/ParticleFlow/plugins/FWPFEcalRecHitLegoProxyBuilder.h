#ifndef _FWPFEcalRecHitLegoProxyBuilder_H_
#define _FWPFEcalRecHitLegoProxyBuilder_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFEcalRecHitLegoProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//

// System include files
#include <cmath>
#include "TEveCompound.h"

// User include files
#include "Fireworks/Core/interface/FWProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/Context.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "Fireworks/ParticleFlow/interface/FWPFLegoRecHit.h"
#include "Fireworks/ParticleFlow/interface/FWPFMaths.h"

#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"

//-----------------------------------------------------------------------------
// FWPFEcalRecHitLegoProxyBuilder
//-----------------------------------------------------------------------------

class FWPFEcalRecHitLegoProxyBuilder : public FWProxyBuilderTemplate<EcalRecHit>
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFEcalRecHitLegoProxyBuilder() : m_maxEnergy(0), m_maxEt(0), m_maxEnergyLog(0), m_maxEtLog(0) {}
      ~FWPFEcalRecHitLegoProxyBuilder() override{}

      static std::string typeOfBuilder() { return "simple#"; }

   // --------------------- Member Functions --------------------------
      void build( const FWEventItem *iItem, TEveElementList *product, const FWViewContext* ) override;

      void scaleProduct( TEveElementList *parent, FWViewType::EType, const FWViewContext *vc ) override;
      bool havePerViewProduct( FWViewType::EType ) const override { return true; }
      void cleanLocal() override;

      // Needed by FWPFLegoRecHit
      TEveVector  calculateCentre( const std::vector<TEveVector> & corners ) const;
      float getMaxValLog(bool et) const { return et ? m_maxEtLog : m_maxEnergyLog; }
      float getMaxVal( bool et ) const { return et ? m_maxEt : m_maxEnergy; }

      REGISTER_PROXYBUILDER_METHODS();

   protected:

   // --------------------- Member Functions --------------------------
      void localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                     FWViewType::EType viewType, const FWViewContext* vc) override;

   private:
      // Disable default copy constructor
      FWPFEcalRecHitLegoProxyBuilder( const FWPFEcalRecHitLegoProxyBuilder& ) = delete;
      // Disable default assignment operator
      const FWPFEcalRecHitLegoProxyBuilder& operator=( const FWPFEcalRecHitLegoProxyBuilder& ) = delete;

   // ----------------------- Data Members ----------------------------
      float m_maxEnergy;
      float m_maxEt;
      float m_maxEnergyLog;
      float m_maxEtLog;
      std::vector<FWPFLegoRecHit*> m_recHits;
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
