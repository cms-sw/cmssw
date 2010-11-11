#ifndef _FWPFCLUSTERRPPROXYBUILDER_H_
#define _FWPFCLUSTERRPPROXYBUILDER_H_

#include "TEveScalableStraightLineSet.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Core/interface/FWViewContext.h"

class FWPFClusterRPProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFCluster>
{
   private:
      struct ScalableLines
      {
         ScalableLines( TEveScalableStraightLineSet *ls, float et, float e, const FWViewContext *vc ) :
         m_ls(ls), m_et(et), m_energy(e), m_vc(vc){}

         TEveScalableStraightLineSet *m_ls;
         float m_et, m_energy;
         const FWViewContext *m_vc;
      };
      std::vector<ScalableLines> m_clusters;

      FWPFClusterRPProxyBuilder( const FWPFClusterRPProxyBuilder& );                       // Disable default
      const FWPFClusterRPProxyBuilder& operator=( const FWPFClusterRPProxyBuilder& );      // Disable default

        // ------------------------- member functions -------------------------------
      float calculateEt( const reco::PFCluster &cluster );

   public:
      static std::string typeOfBuilder() { return "simple#"; }

        // -------------------- Constructor(s)/Destructors --------------------------
      FWPFClusterRPProxyBuilder(){}
      virtual ~FWPFClusterRPProxyBuilder(){}

       // ------------------------- member functions -------------------------------
      virtual void build( const reco::PFCluster &iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext *vc );
      virtual void scaleProduct( TEveElementList *parent, FWViewType::EType, const FWViewContext *vc );
      virtual bool havePerViewProduct( FWViewType::EType ) const { return true; }
      virtual void cleanLocal() { m_clusters.clear(); }

      REGISTER_PROXYBUILDER_METHODS();
};
#endif
