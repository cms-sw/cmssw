// -*- C++ -*-
//
// Package: Fireworks
// Class  : FWPFJetLegoProxyBuilder

/*

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author: Colin Bernet
//         Created: Fri May 28 15:33:06 2010 
//
//


// system include files

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "Fireworks/ParticleFlow/interface/FWLegoEvePFCandidate.h"
#include "Fireworks/ParticleFlow/src/FWPFScale.h"

#include "Fireworks/ParticleFlow/interface/setTrackTypePF.h"

// forward declarations

class FWPFJetLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFJet> {
public:
   FWPFJetLegoProxyBuilder();
   virtual ~FWPFJetLegoProxyBuilder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

   virtual bool havePerViewProduct(FWViewType::EType) const { return true; }

   virtual void scaleProduct(TEveElementList* parent, FWViewType::EType, const FWViewContext* vc);


   virtual void localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                  FWViewType::EType viewType, const FWViewContext* vc);

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPFJetLegoProxyBuilder(const FWPFJetLegoProxyBuilder&); // stop default
   const FWPFJetLegoProxyBuilder& operator=(const FWPFJetLegoProxyBuilder&); // stop default
   
   void build(const reco::PFJet&, unsigned int, TEveElement&, const FWViewContext*);

   // ---------- member data --------------------------------
};

//
// constructors and destructor
//
FWPFJetLegoProxyBuilder::FWPFJetLegoProxyBuilder()
{
}

FWPFJetLegoProxyBuilder::~FWPFJetLegoProxyBuilder()
{
}

//
// member functions
//
void 
FWPFJetLegoProxyBuilder::build(const reco::PFJet& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* vc)
{
   std::vector<reco::PFCandidatePtr > consts = iData.getPFConstituents();

   typedef std::vector<reco::PFCandidatePtr >::const_iterator IC;
 
   for(IC ic=consts.begin();
       ic!=consts.end(); ++ic) {

      const reco::PFCandidatePtr pfCandPtr = *ic;
      FWLegoEvePFCandidate* evePFCandidate = new FWLegoEvePFCandidate( *pfCandPtr, vc, context());

      evePFCandidate->SetLineWidth(3);
      evePFCandidate->SetMarkerColor(item()->defaultDisplayProperties().color());
      fireworks::setTrackTypePF(  (*pfCandPtr), evePFCandidate); 
      setupAddElement( evePFCandidate, &oItemHolder );
   }
}

void
FWPFJetLegoProxyBuilder::scaleProduct(TEveElementList* parent, FWViewType::EType type, const FWViewContext* vc)
{
   // loop items in product
   for (TEveElement::List_i i = parent->BeginChildren(); i!= parent->EndChildren(); ++i)
   {
      if ((*i)->HasChildren())
      {
         // loop elements for the reco::PFJet item
         for (TEveElement::List_i j = (*i)->BeginChildren(); j != (*i)->EndChildren(); ++j)
         {
            FWLegoEvePFCandidate* cand = dynamic_cast<FWLegoEvePFCandidate*> (*j);  
            cand->updateScale( vc, context());
         }
      }
   }
}

void
FWPFJetLegoProxyBuilder::localModelChanges(const FWModelId& iId, TEveElement* parent,
                                                  FWViewType::EType viewType, const FWViewContext* vc)
{
   // line set marker is not same color as line, have to fix it here
   if ((parent)->HasChildren())
   {
      for (TEveElement::List_i j = parent->BeginChildren(); j != parent->EndChildren(); ++j)
      {
         FWLegoEvePFCandidate* cand = dynamic_cast<FWLegoEvePFCandidate*> (*j); 
         const FWDisplayProperties& dp = item()->modelInfo(iId.index()).displayProperties();
         cand->SetMarkerColor( dp.color());
         cand->ElementChanged();
      }
   }  
}

REGISTER_FWPROXYBUILDER(FWPFJetLegoProxyBuilder, reco::PFJet, "PFJet", FWViewType::kLegoBit);
