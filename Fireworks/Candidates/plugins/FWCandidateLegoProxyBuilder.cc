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
      virtual ~FWCandidateLegoProxyBuilder(){}

   // --------------------- Member Functions --------------------------
      virtual bool havePerViewProduct( FWViewType::EType ) const { return true; }
      virtual void scaleProduct( TEveElementList*, FWViewType::EType, const FWViewContext* );
      virtual void localModelChanges( const FWModelId&, TEveElement*, FWViewType::EType,
                                 const FWViewContext* );

      REGISTER_PROXYBUILDER_METHODS();

   private:
   // ----------------------- Data Members ----------------------------
      FWCandidateLegoProxyBuilder( const FWCandidateLegoProxyBuilder& );
      const FWCandidateLegoProxyBuilder& operator=( const FWCandidateLegoProxyBuilder& );

   // --------------------- Member Functions --------------------------
      void build( const reco::Candidate&, unsigned int, TEveElement&, const FWViewContext* );
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
      const FWDisplayProperties &dp = item()->modelInfo( iId.index() ).displayProperties();
      candidate->SetMarkerColor( dp.color() );
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
