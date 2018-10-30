// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWCandidatesLegoProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Colin Bernet
//         Created:  Fri May 28 14:54:19 CEST 2010
// Edited:           sharris, Wed 10 Feb 2011, 13:00
//

// User include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "Fireworks/Candidates/interface/FWLegoCandidate.h"
#include "Fireworks/ParticleFlow/interface/setTrackTypePF.h"   // NB: This has to come after FWLegoCandidate include

//-----------------------------------------------------------------------------
// FWCandidate3DProxyBuilder
//-----------------------------------------------------------------------------

class FWPFCandidatesLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFCandidate> 
{
   public:
      FWPFCandidatesLegoProxyBuilder();
      ~FWPFCandidatesLegoProxyBuilder() override;

   // --------------------- Member Functions --------------------------
      bool havePerViewProduct(FWViewType::EType) const override { return true; }
      void scaleProduct(TEveElementList* parent, FWViewType::EType, const FWViewContext* vc) override;
      void localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                     FWViewType::EType viewType, const FWViewContext* vc) override;

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFCandidatesLegoProxyBuilder(const FWPFCandidatesLegoProxyBuilder&) = delete; // stop default
      const FWPFCandidatesLegoProxyBuilder& operator=(const FWPFCandidatesLegoProxyBuilder&) = delete; // stop default
      
   // --------------------- Member Functions --------------------------
      using FWSimpleProxyBuilderTemplate<reco::PFCandidate> ::build;
      void build(const reco::PFCandidate&, unsigned int, TEveElement&, const FWViewContext*) override;
};
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_


//
// constructors and destructor
//
//______________________________________________________________________________
FWPFCandidatesLegoProxyBuilder::FWPFCandidatesLegoProxyBuilder(){}
FWPFCandidatesLegoProxyBuilder::~FWPFCandidatesLegoProxyBuilder(){}

//
// member functions
//
//______________________________________________________________________________
void 
FWPFCandidatesLegoProxyBuilder::build(const reco::PFCandidate &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc)
{
   FWLegoCandidate *candidate = new FWLegoCandidate( vc, context(), iData.energy(), iData.et(), iData.pt(),
                                                         iData.eta(), iData.phi() );
   candidate->SetMarkerColor( item()->defaultDisplayProperties().color() );
   fireworks::setTrackTypePF( iData, candidate );

   context().voteMaxEtAndEnergy( iData.et(), iData.et() );

   setupAddElement( candidate, &oItemHolder );
}

//______________________________________________________________________________
void
FWPFCandidatesLegoProxyBuilder::scaleProduct(TEveElementList* parent, FWViewType::EType type, const FWViewContext* vc)
{
   for (TEveElement::List_i i = parent->BeginChildren(); i!= parent->EndChildren(); ++i)
   {
      if ((*i)->HasChildren())
      {
         TEveElement* el = (*i)->FirstChild();  // there is only one child added in this proxy builder
         FWLegoCandidate *candidate = dynamic_cast<FWLegoCandidate*> (el);
         candidate->updateScale(vc, context());
      }
   }
}

//______________________________________________________________________________
void
FWPFCandidatesLegoProxyBuilder::localModelChanges(const FWModelId& iId, TEveElement* parent,
                                                  FWViewType::EType viewType, const FWViewContext* vc)
{
   // line set marker is not same color as line, have to fix it here
   if ((parent)->HasChildren())
   {
      TEveElement* el = (parent)->FirstChild();  // we know there is only one child added in this proxy builder
      FWLegoCandidate *candidate = dynamic_cast<FWLegoCandidate*> (el);
      candidate->SetMarkerColor( item()->modelInfo(iId.index()).displayProperties().color());
      candidate->ElementChanged();
   }  
}

//______________________________________________________________________________
REGISTER_FWPROXYBUILDER(FWPFCandidatesLegoProxyBuilder, reco::PFCandidate, "PF Candidates", FWViewType::kLegoPFECALBit | FWViewType::kLegoBit );
