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

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "Fireworks/ParticleFlow/interface/FWLegoEvePFCandidate.h"

// forward declarations

class FWPFJetLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFJet> {
public:
   FWPFJetLegoProxyBuilder();
   virtual ~FWPFJetLegoProxyBuilder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

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
FWPFJetLegoProxyBuilder::build(const reco::PFJet& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*)
{
   std::vector<reco::PFCandidatePtr > consts = iData.getPFConstituents();

   typedef std::vector<reco::PFCandidatePtr >::const_iterator IC;

   for(IC ic=consts.begin();
       ic!=consts.end(); ++ic) {

      const reco::PFCandidatePtr pfCandPtr = *ic;

      FWLegoEvePFCandidate* evePFCandidate = new FWLegoEvePFCandidate( *pfCandPtr );

      evePFCandidate->SetLineWidth(3);
      setupAddElement( evePFCandidate, &oItemHolder );
   }
}

//
// const member functions
//

//
// static member functions
//

REGISTER_FWPROXYBUILDER(FWPFJetLegoProxyBuilder, reco::PFJet, "PFJet", FWViewType::kLegoBit);
