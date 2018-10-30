#ifndef _FWCANDIDATEHGCALLEGOPROXYBUILDER_H_
#define _FWCANDIDATEHGCALLEGOPROXYBUILDER_H_

// -*- C++ -*-
//
// Package:     Candidates
// Class  :     FWCandidateHGCalLegoProxyBuilder
//
//

// User include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "DataFormats/ParticleFlowReco/interface/HGCalMultiCluster.h"

#include "Fireworks/Candidates/interface/FWLegoCandidate.h"

#include <cmath>

//-----------------------------------------------------------------------------
// FWCandidateHGCalLegoProxyBuilder
//-----------------------------------------------------------------------------
class FWCandidateHGCalLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::HGCalMultiCluster>
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWCandidateHGCalLegoProxyBuilder(){}
      ~FWCandidateHGCalLegoProxyBuilder() override{}

   // --------------------- Member Functions --------------------------
      bool havePerViewProduct( FWViewType::EType ) const override { return true; }
      void scaleProduct( TEveElementList*, FWViewType::EType, const FWViewContext* ) override;
      void localModelChanges( const FWModelId&, TEveElement*, FWViewType::EType,
                                 const FWViewContext* ) override;

      REGISTER_PROXYBUILDER_METHODS();

   private:
   // ----------------------- Data Members ----------------------------
      FWCandidateHGCalLegoProxyBuilder( const FWCandidateHGCalLegoProxyBuilder& ) = delete;
      const FWCandidateHGCalLegoProxyBuilder& operator=( const FWCandidateHGCalLegoProxyBuilder& ) = delete;

   // --------------------- Member Functions --------------------------
      using FWSimpleProxyBuilderTemplate<reco::HGCalMultiCluster>::build;
      void build( const reco::HGCalMultiCluster&, unsigned int, TEveElement&, const FWViewContext* ) override;
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_


//______________________________________________________________________________
void
FWCandidateHGCalLegoProxyBuilder::scaleProduct( TEveElementList *parent, FWViewType::EType type,
                                           const FWViewContext *vc )
{
   for( TEveElement::List_i i = parent->BeginChildren(); i != parent->EndChildren(); ++i )
   {
      if( (*i)->HasChildren() )
      {
         TEveElement *el = (*i)->FirstChild();  // There is only one child
         FWLegoCandidate *candidate = dynamic_cast<FWLegoCandidate*> (el);
         candidate->updateScale( vc, context() );
      }
   }
}

//______________________________________________________________________________
void
FWCandidateHGCalLegoProxyBuilder::localModelChanges( const FWModelId &iId, TEveElement *parent,
                                       FWViewType::EType type, const FWViewContext *vc )
{
   // Line set marker is nto the same color as line, have to fix it here
   if( (parent)->HasChildren() )
   {
      TEveElement *el = (parent)->FirstChild();       // There is only one child
      FWLegoCandidate *candidate = dynamic_cast<FWLegoCandidate*> (el);
      candidate->SetMarkerColor( item()->modelInfo( iId.index() ).displayProperties().color() );
      candidate->ElementChanged();
   }
}

//______________________________________________________________________________
void
FWCandidateHGCalLegoProxyBuilder::build( const reco::HGCalMultiCluster &iData, unsigned int iIndex,
                              TEveElement &oItemHolder, const FWViewContext *vc )
{
  const auto & clusters = iData.clusters();

  for (const auto & c : clusters)
    {
      auto pt = c->energy()/std::cosh(c->eta());
      FWLegoCandidate *candidate = new FWLegoCandidate( vc, context(), c->energy(),
							pt, pt,
							c->eta(), c->phi() );

      candidate->SetMarkerColor( item()->defaultDisplayProperties().color() );
      context().voteMaxEtAndEnergy( pt, c->energy() );
      setupAddElement( candidate, &oItemHolder );
    }
}

//______________________________________________________________________________
REGISTER_FWPROXYBUILDER( FWCandidateHGCalLegoProxyBuilder, reco::HGCalMultiCluster, "HGCal Multiclusters Lego",
                         FWViewType::kLegoBit | FWViewType::kLegoHF );
