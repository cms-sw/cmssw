#ifndef _FWPFCLUSTERRPZPROXYBUILDER_H_
#define _FWPFCLUSTERRPZPROXYBUILDER_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFClusterRPZProxyBuilder, FWPFEcalClusterRPZProxyBuilder, FWPFHcalClusterRPZProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//

// System include files
#include "TEveScalableStraightLineSet.h"

// User include files
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/ParticleFlow/interface/FWPFUtils.h"

//-----------------------------------------------------------------------------
// FWPFClusterRPZProxyBuilder
//-----------------------------------------------------------------------------

class FWPFClusterRPZProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFCluster>
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFClusterRPZProxyBuilder(){ m_pfUtils = new FWPFUtils(); }
      virtual ~FWPFClusterRPZProxyBuilder(){ delete m_pfUtils; }

   // --------------------- Member Functions --------------------------
      virtual void build( const reco::PFCluster &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc );
      virtual void scaleProduct( TEveElementList *parent, FWViewType::EType, const FWViewContext *vc );
      virtual bool havePerViewProduct( FWViewType::EType ) const { return true; }
      virtual void cleanLocal() { m_clusters.clear(); }

      REGISTER_PROXYBUILDER_METHODS();

   protected:
   // ----------------------- Data Members ----------------------------
      struct ScalableLines
      {
         ScalableLines( TEveScalableStraightLineSet *ls, float et, float e, const FWViewContext *vc ) :
         m_ls(ls), m_et(et), m_energy(e), m_vc(vc){}

         TEveScalableStraightLineSet *m_ls;
         float m_et, m_energy;
         const FWViewContext *m_vc;
      };
      std::vector<ScalableLines> m_clusters;
      FWPFUtils *m_pfUtils;

   // --------------------- Member Functions --------------------------
      float calculateEt( const reco::PFCluster &cluster, float E );
      virtual void sharedBuild( const reco::PFCluster &cluster, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc, float radius );

   private:
      FWPFClusterRPZProxyBuilder( const FWPFClusterRPZProxyBuilder& );                    // Disable default
      const FWPFClusterRPZProxyBuilder& operator=( const FWPFClusterRPZProxyBuilder& );   // Disable default
};
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_


//-----------------------------------------------------------------------------
// FWPFEcalClusterRPZProxyBuilder
//-----------------------------------------------------------------------------

class FWPFEcalClusterRPZProxyBuilder : public FWPFClusterRPZProxyBuilder
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFEcalClusterRPZProxyBuilder(){}
      virtual ~FWPFEcalClusterRPZProxyBuilder(){}

   // --------------------- Member Functions --------------------------
      virtual void build( const reco::PFCluster &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc );

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFEcalClusterRPZProxyBuilder( const FWPFEcalClusterRPZProxyBuilder& );
      const FWPFEcalClusterRPZProxyBuilder& operator=( const FWPFEcalClusterRPZProxyBuilder& );
};
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_


//-----------------------------------------------------------------------------
// FWPFHcalClusterRPZProxyBuilder
//-----------------------------------------------------------------------------

class FWPFHcalClusterRPZProxyBuilder : public FWPFClusterRPZProxyBuilder
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWPFHcalClusterRPZProxyBuilder(){}
      virtual ~FWPFHcalClusterRPZProxyBuilder(){}

   // --------------------- Member Functions --------------------------
      virtual void build( const reco::PFCluster &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc );

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFHcalClusterRPZProxyBuilder( const FWPFHcalClusterRPZProxyBuilder& );
      const FWPFHcalClusterRPZProxyBuilder& operator=( const FWPFHcalClusterRPZProxyBuilder& );
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
