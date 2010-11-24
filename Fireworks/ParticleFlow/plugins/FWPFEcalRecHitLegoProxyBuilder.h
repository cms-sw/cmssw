#ifndef _FWPFEcalRecHitLegoProxyBuilder_H_
#define _FWPFEcalRecHitLegoProxyBuilder_H_

//
// Package:             Particle Flow
// Class:               FWPFEcalRecHitLegoProxyBuilder
// Original Author:     Simon Harris
//

#include <math.h>

#include "TEveScalableStraightLineSet.h"

// User include files
#include "Fireworks/Core/interface/FWProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/Context.h"
#include "TEveCompound.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "Fireworks/ParticleFlow/plugins/FWPFLegoRecHit.h"

#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"


class FWPFEcalRecHitLegoProxyBuilder : public FWProxyBuilderTemplate<EcalRecHit>
{
private:
   // Disable default copy constructor
   FWPFEcalRecHitLegoProxyBuilder( const FWPFEcalRecHitLegoProxyBuilder& );
   // Disable default assignment operator
   const FWPFEcalRecHitLegoProxyBuilder& operator=( const FWPFEcalRecHitLegoProxyBuilder& );

   // ------------------------- Member Functions -------------------------------
   float calculateEt( const TEveVector &centre, float E );

   // --------------------------- Data Members ---------------------------------
   float m_maxEnergyLog;
   float m_maxEtLog;
   std::vector<FWPFLegoRecHit*> m_recHits;
protected:

   virtual void localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                  FWViewType::EType viewType, const FWViewContext* vc);
public:
   // -------------------- Constructor(s)/Destructors --------------------------
   FWPFEcalRecHitLegoProxyBuilder(){}
   virtual ~FWPFEcalRecHitLegoProxyBuilder(){}

   static std::string typeOfBuilder() { return "simple#"; }

   virtual void build( const FWEventItem *iItem, TEveElementList *product, const FWViewContext* );
   virtual bool visibilityModelChanges(const FWModelId&, TEveElement*, FWViewType::EType, const FWViewContext*);


   virtual void scaleProduct( TEveElementList *parent, FWViewType::EType, const FWViewContext *vc );
   virtual bool havePerViewProduct( FWViewType::EType ) const { return true; }
   virtual void cleanLocal();

   // needed by LegoRecHit
   TEveVector  calculateCentre( const std::vector<TEveVector> & corners ) const;
   float getMaxValLog(bool et) const { return et ? m_maxEtLog : m_maxEnergyLog; }

   REGISTER_PROXYBUILDER_METHODS();

};
#endif
