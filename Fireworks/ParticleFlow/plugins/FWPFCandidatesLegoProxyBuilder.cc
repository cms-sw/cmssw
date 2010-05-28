// -*- C++ -*-
//
// Package: Fireworks
// Class  : FWPFCandidatesLegoProxyBuilder

/*

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author: Colin Bernet
//         Created: Fri May 28 14:54:08 2010 
//
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "Fireworks/ParticleFlow/interface/FWLegoEvePFCandidate.h"

// forward declarations

class FWPFCandidatesLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFCandidate> {
public:
   FWPFCandidatesLegoProxyBuilder();
   virtual ~FWPFCandidatesLegoProxyBuilder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

   REGISTER_PROXYBUILDER_METHODS();
private:
   FWPFCandidatesLegoProxyBuilder(const FWPFCandidatesLegoProxyBuilder&); // stop default
   const FWPFCandidatesLegoProxyBuilder& operator=(const FWPFCandidatesLegoProxyBuilder&); // stop default
   
   void build(const reco::PFCandidate&, unsigned int, TEveElement&, const FWViewContext*);

   // ---------- member data --------------------------------
};

//
// constructors and destructor
//
FWPFCandidatesLegoProxyBuilder::FWPFCandidatesLegoProxyBuilder()
{
}

FWPFCandidatesLegoProxyBuilder::~FWPFCandidatesLegoProxyBuilder()
{
}

//
// member functions
//
void 
FWPFCandidatesLegoProxyBuilder::build(const reco::PFCandidate& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*)
{
  FWLegoEvePFCandidate* evePFCandidate = new FWLegoEvePFCandidate( iData );
  setupAddElement( evePFCandidate, &oItemHolder );
}

//
// const member functions
//

//
// static member functions
//

REGISTER_FWPROXYBUILDER(FWPFCandidatesLegoProxyBuilder, reco::PFCandidate, "PFCandidates", FWViewType::kLegoBit);
