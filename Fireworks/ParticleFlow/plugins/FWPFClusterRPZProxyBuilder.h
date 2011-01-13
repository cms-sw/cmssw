#ifndef _FWPFCLUSTERRPZPROXYBUILDER_H_
#define _FWPFCLUSTERRPZPROXYBUILDER_H_

#include "TEveScalableStraightLineSet.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Core/interface/FWViewContext.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Base ProxyBuilder
///////////////////////////////////////////////////////////////////////////////////////////////////////////
class FWPFClusterRPZProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFCluster>
{
   public:
        // -------------------- Constructor(s)/Destructors --------------------------
      FWPFClusterRPZProxyBuilder(){}
      virtual ~FWPFClusterRPZProxyBuilder(){}

       // ------------------------- member functions -------------------------------
      virtual void build( const reco::PFCluster &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc );
      virtual void scaleProduct( TEveElementList *parent, FWViewType::EType, const FWViewContext *vc );
      virtual bool havePerViewProduct( FWViewType::EType ) const { return true; }
      virtual void cleanLocal() { m_clusters.clear(); }

      REGISTER_PROXYBUILDER_METHODS();

   protected:
      struct ScalableLines
      {
         ScalableLines( TEveScalableStraightLineSet *ls, float et, float e, const FWViewContext *vc ) :
         m_ls(ls), m_et(et), m_energy(e), m_vc(vc){}

         TEveScalableStraightLineSet *m_ls;
         float m_et, m_energy;
         const FWViewContext *m_vc;
      };
      std::vector<ScalableLines> m_clusters;

       // ------------------------- member functions -------------------------------
      float calculateEt( const reco::PFCluster &cluster, float E );
      virtual void sharedBuild( const reco::PFCluster &cluster, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc, float radius );

   private:
      FWPFClusterRPZProxyBuilder( const FWPFClusterRPZProxyBuilder& );                    // Disable default
      const FWPFClusterRPZProxyBuilder& operator=( const FWPFClusterRPZProxyBuilder& );   // Disable default
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// ECAL
///////////////////////////////////////////////////////////////////////////////////////////////////////////
class FWPFEcalClusterRPZProxyBuilder : public FWPFClusterRPZProxyBuilder
{
   public:
      FWPFEcalClusterRPZProxyBuilder(){}
      virtual ~FWPFEcalClusterRPZProxyBuilder(){}

      virtual void build( const reco::PFCluster &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc );

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFEcalClusterRPZProxyBuilder( const FWPFEcalClusterRPZProxyBuilder& );
      const FWPFEcalClusterRPZProxyBuilder& operator=( const FWPFEcalClusterRPZProxyBuilder& );
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// HCAL
///////////////////////////////////////////////////////////////////////////////////////////////////////////
class FWPFHcalClusterRPZProxyBuilder : public FWPFClusterRPZProxyBuilder
{
   public:
      FWPFHcalClusterRPZProxyBuilder(){}
      virtual ~FWPFHcalClusterRPZProxyBuilder(){}

      virtual void build( const reco::PFCluster &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc );

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFHcalClusterRPZProxyBuilder( const FWPFHcalClusterRPZProxyBuilder& );
      const FWPFHcalClusterRPZProxyBuilder& operator=( const FWPFHcalClusterRPZProxyBuilder& );
};
#endif
