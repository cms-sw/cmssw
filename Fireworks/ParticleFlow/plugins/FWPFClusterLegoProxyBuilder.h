#ifndef _FWPFCLUSTERLEGOPROXYBUILDER_H_
#define _FWPFCLUSTERLEGOPROXYBUILDER_H_

//
// Package:             Particle Flow
// Class:               FWPFClusterLegoProxyBuilder
// Original Author:     Simon Harris
//

#include <math.h>

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "Fireworks/ParticleFlow/plugins/FWPFLegoCandidate.h"
#include "Fireworks/Core/interface/FWProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/Context.h"
#include "TEveCompound.h"
#include "TEveBox.h"


///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Base ProxyBuilder
///////////////////////////////////////////////////////////////////////////////////////////////////////////
class FWPFClusterLegoProxyBuilder : public FWProxyBuilderTemplate<reco::PFCluster>
{
    public:
      static std::string typeOfBuilder() { return "simple#"; }

      // -------------------- Constructor(s)/Destructors --------------------------
      FWPFClusterLegoProxyBuilder(){}
      virtual ~FWPFClusterLegoProxyBuilder(){}

      // ------------------------- member functions -------------------------------
      virtual void scaleProduct( TEveElementList *parent, FWViewType::EType, const FWViewContext *vc );
      virtual bool havePerViewProduct(FWViewType::EType) const { return true; }
      virtual void localModelChanges( const FWModelId &iId, TEveElement *iCompound,
                                        FWViewType::EType viewType, const FWViewContext *vc );
   
      REGISTER_PROXYBUILDER_METHODS();

   protected:
      // ------------------------- member functions -------------------------------
      void  sharedBuild( const reco::PFCluster &iData, TEveCompound *itemHolder, const FWViewContext *vc );
      float calculateEt( const reco::PFCluster &cluster, float E );

   private:
      // Disable default copy constructor
      FWPFClusterLegoProxyBuilder( const FWPFClusterLegoProxyBuilder& );
      // Disable default assignment operator
      const FWPFClusterLegoProxyBuilder& operator=( const FWPFClusterLegoProxyBuilder& );
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// ECAL
///////////////////////////////////////////////////////////////////////////////////////////////////////////
class FWPFEcalClusterLegoProxyBuilder : public FWPFClusterLegoProxyBuilder
{
   public:
      FWPFEcalClusterLegoProxyBuilder(){}
      virtual ~FWPFEcalClusterLegoProxyBuilder(){}

      virtual void build( const FWEventItem *iItem, TEveElementList *product, const FWViewContext* );

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFEcalClusterLegoProxyBuilder( const FWPFEcalClusterLegoProxyBuilder& );
      const FWPFEcalClusterLegoProxyBuilder& operator=( const FWPFEcalClusterLegoProxyBuilder& );
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// HCAL
///////////////////////////////////////////////////////////////////////////////////////////////////////////
class FWPFHcalClusterLegoProxyBuilder : public FWPFClusterLegoProxyBuilder
{
   public:
      FWPFHcalClusterLegoProxyBuilder(){}
      virtual ~FWPFHcalClusterLegoProxyBuilder(){}

      virtual void build( const FWEventItem *iItem, TEveElementList *product, const FWViewContext* );

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFHcalClusterLegoProxyBuilder( const FWPFHcalClusterLegoProxyBuilder& );
      const FWPFHcalClusterLegoProxyBuilder& operator=( const FWPFHcalClusterLegoProxyBuilder& );
};
#endif
