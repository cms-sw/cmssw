#ifndef _FWPFCLUSTERLEGOPROXYBUILDER_H_
#define _FWPFCLUSTERLEGOPROXYBUILDER_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFClusterLegoProxyBuilder, FWPFEcalClusterLegoProxyBuilder, FWPFHcalClusterLegoProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//

// System include files
#include <math.h>
#include "TEveBox.h"

// User include files
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "Fireworks/ParticleFlow/interface/FWPFMaths.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/Context.h"

//-----------------------------------------------------------------------------
// FWPFClusterLegoProxyBuilder
//-----------------------------------------------------------------------------

class FWPFClusterLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFCluster>
{
    public:
      static std::string typeOfBuilder() { return "simple#"; }

   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFClusterLegoProxyBuilder(){}
      virtual ~FWPFClusterLegoProxyBuilder(){}

   // --------------------- Member Functions --------------------------
      virtual void scaleProduct( TEveElementList *parent, FWViewType::EType, const FWViewContext *vc );
      virtual bool havePerViewProduct(FWViewType::EType) const { return true; }
      virtual void localModelChanges( const FWModelId &iId, TEveElement *el,
                                        FWViewType::EType viewType, const FWViewContext *vc );
   
      REGISTER_PROXYBUILDER_METHODS();

   protected:
   // --------------------- Member Functions --------------------------
      void sharedBuild( const reco::PFCluster&, TEveElement&, const FWViewContext* );
      float calculateEt( const reco::PFCluster &cluster, float E );

   private:
      // Disable default copy constructor
      FWPFClusterLegoProxyBuilder( const FWPFClusterLegoProxyBuilder& );
      // Disable default assignment operator
      const FWPFClusterLegoProxyBuilder& operator=( const FWPFClusterLegoProxyBuilder& );
};
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_

//-----------------------------------------------------------------------------
// FWPFEcalClusterLegoProxyBuilder
//-----------------------------------------------------------------------------

class FWPFEcalClusterLegoProxyBuilder : public FWPFClusterLegoProxyBuilder
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFEcalClusterLegoProxyBuilder(){}
      virtual ~FWPFEcalClusterLegoProxyBuilder(){}

   // --------------------- Member Functions --------------------------
      virtual void build( const reco::PFCluster&, unsigned int, TEveElement&, const FWViewContext* );

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFEcalClusterLegoProxyBuilder( const FWPFEcalClusterLegoProxyBuilder& );
      const FWPFEcalClusterLegoProxyBuilder& operator=( const FWPFEcalClusterLegoProxyBuilder& );
};
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_

//-----------------------------------------------------------------------------
// FWPFHcalClusterLegoProxyBuilder
//-----------------------------------------------------------------------------

class FWPFHcalClusterLegoProxyBuilder : public FWPFClusterLegoProxyBuilder
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFHcalClusterLegoProxyBuilder(){}
      virtual ~FWPFHcalClusterLegoProxyBuilder(){}

   // --------------------- Member Functions --------------------------
      virtual void build( const reco::PFCluster&, unsigned int, TEveElement&, const FWViewContext* );

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFHcalClusterLegoProxyBuilder( const FWPFHcalClusterLegoProxyBuilder& );
      const FWPFHcalClusterLegoProxyBuilder& operator=( const FWPFHcalClusterLegoProxyBuilder& );
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
