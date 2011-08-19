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
#include "FWPFLegoCandidate.h"
#include "Fireworks/ParticleFlow/interface/setTrackTypePF.h"   // NB: This has to come after FWPFLegoCandidate include

//-----------------------------------------------------------------------------
// FWCandidate3DProxyBuilder
//-----------------------------------------------------------------------------

class FWPFCandidatesLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFCandidate> 
{
   public:
      FWPFCandidatesLegoProxyBuilder();
      virtual ~FWPFCandidatesLegoProxyBuilder();

   // --------------------- Member Functions --------------------------
      virtual bool havePerViewProduct(FWViewType::EType) const { return true; }
      virtual void scaleProduct(TEveElementList* parent, FWViewType::EType, const FWViewContext* vc);
      virtual void localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                     FWViewType::EType viewType, const FWViewContext* vc);

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFCandidatesLegoProxyBuilder(const FWPFCandidatesLegoProxyBuilder&); // stop default
      const FWPFCandidatesLegoProxyBuilder& operator=(const FWPFCandidatesLegoProxyBuilder&); // stop default
      
   // --------------------- Member Functions --------------------------
      void build(const reco::PFCandidate&, unsigned int, TEveElement&, const FWViewContext*);
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
   FWPFLegoCandidate *candidate = new FWPFLegoCandidate( vc, context(), iData.energy(), iData.et(), iData.pt(),
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
         FWPFLegoCandidate *candidate = dynamic_cast<FWPFLegoCandidate*> (el);
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
      FWPFLegoCandidate *candidate = dynamic_cast<FWPFLegoCandidate*> (el);
      const FWDisplayProperties& dp = item()->modelInfo(iId.index()).displayProperties();
      candidate->SetMarkerColor( dp.color());
      candidate->ElementChanged();
   }  
}

//______________________________________________________________________________
REGISTER_FWPROXYBUILDER(FWPFCandidatesLegoProxyBuilder, reco::PFCandidate, "PF Candidates", FWViewType::kLegoPFECALBit | FWViewType::kLegoBit );
