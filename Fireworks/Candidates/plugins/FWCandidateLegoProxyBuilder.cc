#ifndef _FWCANDIDATELEGOPROXYBUILDER_H_
#define _FWCANDIDATELEGOPROXYBUILDER_H_

// -*- C++ -*-
//
// Package:     Candidates
// Class  :     FWCandidateLegoProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//       Created:    24/06/2011
//

// User include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "Fireworks/Candidates/interface/FWLegoCandidate.h"

//-----------------------------------------------------------------------------
// FWCandidateLegoProxyBuilder
//-----------------------------------------------------------------------------
class FWCandidateLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Candidate>
{
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
      FWCandidateLegoProxyBuilder(){}
      ~FWCandidateLegoProxyBuilder() override{}

   // --------------------- Member Functions --------------------------
      bool havePerViewProduct( FWViewType::EType ) const override { return true; }
      void scaleProduct( TEveElementList*, FWViewType::EType, const FWViewContext* ) override;
      void localModelChanges( const FWModelId&, TEveElement*, FWViewType::EType,
                                 const FWViewContext* ) override;

      REGISTER_PROXYBUILDER_METHODS();

   private:
   // ----------------------- Data Members ----------------------------
      FWCandidateLegoProxyBuilder( const FWCandidateLegoProxyBuilder& ) = delete;
      const FWCandidateLegoProxyBuilder& operator=( const FWCandidateLegoProxyBuilder& ) = delete;

   // --------------------- Member Functions --------------------------
      using FWSimpleProxyBuilderTemplate<reco::Candidate>::build;
      void build( const reco::Candidate&, unsigned int, TEveElement&, const FWViewContext* ) override;
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_


//______________________________________________________________________________
void
FWCandidateLegoProxyBuilder::scaleProduct( TEveElementList *parent, FWViewType::EType type,
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
FWCandidateLegoProxyBuilder::localModelChanges( const FWModelId &iId, TEveElement *parent,
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
FWCandidateLegoProxyBuilder::build( const reco::Candidate &iData, unsigned int iIndex,
                              TEveElement &oItemHolder, const FWViewContext *vc )
{
   FWLegoCandidate *candidate = new FWLegoCandidate( vc, context(), iData.energy(),
                                                         iData.et(), iData.pt(),
                                                         iData.eta(), iData.phi() );

   candidate->SetMarkerColor( item()->defaultDisplayProperties().color() );
   context().voteMaxEtAndEnergy( iData.et(), iData.energy() );

   setupAddElement( candidate, &oItemHolder );
}

//______________________________________________________________________________
REGISTER_FWPROXYBUILDER( FWCandidateLegoProxyBuilder, reco::Candidate, "Candidates",
                         FWViewType::kLegoBit | FWViewType::kLegoPFECALBit );
