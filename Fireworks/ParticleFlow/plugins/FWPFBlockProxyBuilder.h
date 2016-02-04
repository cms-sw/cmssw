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
#include <math.h>
#include "TEveScalableStraightLineSet.h"
#include "TEveCompound.h"

// User include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/Context.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "Fireworks/ParticleFlow/interface/FWPFTrackUtils.h"
#include "Fireworks/ParticleFlow/interface/FWPFClusterRPZUtils.h"
#include "FWPFLegoCandidate.h"

//-----------------------------------------------------------------------------
// FWPFBlockProxyBuilder
//-----------------------------------------------------------------------------
class FWPFBlockProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFBlock>
{
   public:
      enum BuilderType { BASE=0, ECAL=1, HCAL=2 };

   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFBlockProxyBuilder() : e_builderType(BASE) {}
      virtual ~FWPFBlockProxyBuilder(){}

      REGISTER_PROXYBUILDER_METHODS();

   protected:
   // --------------------- Member Functions --------------------------
      float          calculateEt( const TEveVector &cluster, float e );
      void           setupTrackElement( const reco::PFBlockElement&, TEveElement&, const FWViewContext*, FWViewType::EType );
      void           setupClusterElement( const reco::PFBlockElement&, TEveElement&, const FWViewContext*, 
                                          FWViewType::EType, float r );
      void           clusterSharedBuild( const reco::PFCluster&, TEveElement&, const FWViewContext* );

      virtual bool   havePerViewProduct( FWViewType::EType ) const { return true; }
      virtual bool   haveSingleProduct() const { return false; } // different view types
      virtual void   buildViewType( const reco::PFBlock&, unsigned int, TEveElement&, FWViewType::EType, const FWViewContext* );
      virtual void   scaleProduct( TEveElementList *parent, FWViewType::EType, const FWViewContext *vc );
      virtual void   cleanLocal() { m_clusters.clear(); }
      
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
      virtual ~FWPFBlockEcalProxyBuilder(){}

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFBlockEcalProxyBuilder( const FWPFBlockEcalProxyBuilder& );
      const FWPFBlockEcalProxyBuilder& operator=( const FWPFBlockEcalProxyBuilder& );
};
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_


//-----------------------------------------------------------------------------
// FWPFBlockHcalProxyBuilder
//-----------------------------------------------------------------------------
class FWPFBlockHcalProxyBuilder : public FWPFBlockProxyBuilder
{
   public:
      FWPFBlockHcalProxyBuilder(){ e_builderType = HCAL; }
      virtual ~FWPFBlockHcalProxyBuilder(){}

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFBlockHcalProxyBuilder( const FWPFBlockHcalProxyBuilder& );
      const FWPFBlockHcalProxyBuilder& operator=( const FWPFBlockHcalProxyBuilder& );
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
