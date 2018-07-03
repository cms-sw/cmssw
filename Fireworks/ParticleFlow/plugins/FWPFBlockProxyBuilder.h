#ifndef _FWPFBLOCKPROXYBUILDER_H_
#define _FWPFBLOCKPROXYBUILDER_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFBlockProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//       Created:    14/02/2011
//


// System include files
#include <cmath>
#include "TEveScalableStraightLineSet.h"

// User include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/Context.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "Fireworks/ParticleFlow/interface/FWPFTrackUtils.h"
#include "Fireworks/ParticleFlow/interface/FWPFClusterRPZUtils.h"


//-----------------------------------------------------------------------------
// FWPFBlockProxyBuilder
//-----------------------------------------------------------------------------
class FWPFBlockProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFBlock>
{
   public:
      enum BuilderType { BASE=0, ECAL=1, HCAL=2 };

   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFBlockProxyBuilder() : e_builderType(BASE) {}
      ~FWPFBlockProxyBuilder() override{}

      REGISTER_PROXYBUILDER_METHODS();

   protected:
   // --------------------- Member Functions --------------------------
      void           setupTrackElement( const reco::PFBlockElement&, TEveElement&, const FWViewContext*, FWViewType::EType );
      void           setupClusterElement( const reco::PFBlockElement&, TEveElement&, const FWViewContext*, 
                                          FWViewType::EType, float r );
      void           clusterSharedBuild( const reco::PFCluster&, TEveElement&, const FWViewContext* );

      using FWProxyBuilderBase::havePerViewProduct;
      bool   havePerViewProduct( FWViewType::EType ) const override { return true; }

      using FWProxyBuilderBase::haveSingleProduct;
      bool   haveSingleProduct() const override { return false; } // different view types

       using FWProxyBuilderBase::scaleProduct;
      void   scaleProduct( TEveElementList *parent, FWViewType::EType, const FWViewContext *vc ) override;

      using FWProxyBuilderBase::cleanLocal;
      void   cleanLocal() override { m_clusters.clear(); }
      
      using FWSimpleProxyBuilderTemplate<reco::PFBlock>::buildViewType;
      void   buildViewType( const reco::PFBlock&, unsigned int, TEveElement&, FWViewType::EType, const FWViewContext* ) override;

    
   // ----------------------- Data Members ----------------------------
      BuilderType e_builderType;
      std::vector<ScalableLines> m_clusters;

   private:
      FWPFBlockProxyBuilder( const FWPFBlockProxyBuilder& );
      const FWPFBlockProxyBuilder& operator=( const FWPFBlockProxyBuilder& ); 
};
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_


//-----------------------------------------------------------------------------
// FWPFBlockEcalProxyBuilder
//-----------------------------------------------------------------------------
class FWPFBlockEcalProxyBuilder : public FWPFBlockProxyBuilder
{
   public:
      FWPFBlockEcalProxyBuilder(){ e_builderType = ECAL; }
      ~FWPFBlockEcalProxyBuilder() override{}

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFBlockEcalProxyBuilder( const FWPFBlockEcalProxyBuilder& ) = delete;
      const FWPFBlockEcalProxyBuilder& operator=( const FWPFBlockEcalProxyBuilder& ) = delete;
};
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_


//-----------------------------------------------------------------------------
// FWPFBlockHcalProxyBuilder
//-----------------------------------------------------------------------------
class FWPFBlockHcalProxyBuilder : public FWPFBlockProxyBuilder
{
   public:
      FWPFBlockHcalProxyBuilder(){ e_builderType = HCAL; }
      ~FWPFBlockHcalProxyBuilder() override{}

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFBlockHcalProxyBuilder( const FWPFBlockHcalProxyBuilder& ) = delete;
      const FWPFBlockHcalProxyBuilder& operator=( const FWPFBlockHcalProxyBuilder& ) = delete;
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
