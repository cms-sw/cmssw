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
   FWViewEnergyScale* scale = vc->getEnergyScale("ParticleFlow");
   if (!scale)
   {
      scale = new FWPFScale();
      vc->addScale("ParticleFlow", scale);
   }
   for(IC ic=consts.begin();
       ic!=consts.end(); ++ic) {

      const reco::PFCandidatePtr pfCandPtr = *ic;
      scale->setVal( pfCandPtr->et());
      FWLegoEvePFCandidate* evePFCandidate = new FWLegoEvePFCandidate( *pfCandPtr );

      evePFCandidate->SetLineWidth(3);
      evePFCandidate->SetMarkerColor(item()->defaultDisplayProperties().color());
      evePFCandidate->SetMarkerSize(0.01); 
      fireworks::setTrackTypePF(  (*pfCandPtr), evePFCandidate); 
      setupAddElement( evePFCandidate, &oItemHolder );
   }
}

void
FWPFJetLegoProxyBuilder::scaleProduct(TEveElementList* parent, FWViewType::EType type, const FWViewContext* vc)
{
   for (TEveElement::List_i i = parent->BeginChildren(); i!= parent->EndChildren(); ++i)
   {
      if ((*i)->HasChildren())
      {
         TEveElement* el = (*i)->FirstChild();  // there is only one child added in this proxy builder
         FWLegoEvePFCandidate* cand = dynamic_cast<FWLegoEvePFCandidate*> (el);  
         cand->UpdateScale(vc->getEnergyScale("ParticleFlow")->getVal());
      }
   }
}

REGISTER_FWPROXYBUILDER(FWPFJetLegoProxyBuilder, reco::PFJet, "PFJet", FWViewType::kLegoBit);
