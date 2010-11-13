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




class FWPFClusterLegoProxyBuilder : public FWProxyBuilderTemplate<reco::PFCluster>
{
   private:
      // Disable default copy constructor
      FWPFClusterLegoProxyBuilder( const FWPFClusterLegoProxyBuilder& );
      // Disable default assignment operator
      const FWPFClusterLegoProxyBuilder& operator=( const FWPFClusterLegoProxyBuilder& );

      // ------------------------- member functions -------------------------------
      float calculateET( const reco::PFCluster &cluster );

    public:
      static std::string typeOfBuilder() { return "simple#"; }

      // -------------------- Constructor(s)/Destructors --------------------------
      FWPFClusterLegoProxyBuilder(){}
      virtual ~FWPFClusterLegoProxyBuilder(){}

      // ------------------------- member functions -------------------------------
      virtual void build( const FWEventItem *iItem, TEveElementList *product, const FWViewContext* );
      virtual void scaleProduct( TEveElementList *parent, FWViewType::EType, const FWViewContext *vc );
      virtual bool havePerViewProduct(FWViewType::EType) const { return true; }
      virtual void localModelChanges( const FWModelId &iId, TEveElement *iCompound,
                                        FWViewType::EType viewType, const FWViewContext *vc );
   
      REGISTER_PROXYBUILDER_METHODS();

};
#endif
