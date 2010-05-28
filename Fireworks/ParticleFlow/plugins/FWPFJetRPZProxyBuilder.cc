// -*- C++ -*-
//
// Package: Fireworks
// Class  : FWPFJetRPZProxyBuilder

/*

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author: ColinBernet
//         Created: Fri May 28 16:02:09 2010 
//
//

// system include files

// user include files

#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveVSDStructs.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/ParticleFlow/interface/setTrackTypePF.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

// forward declarations

class FWPFJetRPZProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFJet> {
public:
   FWPFJetRPZProxyBuilder();
   virtual ~FWPFJetRPZProxyBuilder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPFJetRPZProxyBuilder(const FWPFJetRPZProxyBuilder&); // stop default
   const FWPFJetRPZProxyBuilder& operator=(const FWPFJetRPZProxyBuilder&); // stop default
   
   void build(const reco::PFJet&, unsigned int, TEveElement&, const FWViewContext*);

   // ---------- member data --------------------------------
};

//
// constructors and destructor
//
FWPFJetRPZProxyBuilder::FWPFJetRPZProxyBuilder()
{
}

FWPFJetRPZProxyBuilder::~FWPFJetRPZProxyBuilder()
{
}

//
// member functions
//
void 
FWPFJetRPZProxyBuilder::build(const reco::PFJet& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*)
{
   std::vector<reco::PFCandidatePtr > consts = iData.getPFConstituents();

   typedef std::vector<reco::PFCandidatePtr >::const_iterator IC;

   for(IC ic=consts.begin();
       ic!=consts.end(); ++ic) {

      const reco::PFCandidatePtr pfCandPtr = *ic;

      TEveRecTrack t;
      t.fBeta = 1.;
      t.fP = TEveVector( pfCandPtr->px(), pfCandPtr->py(), pfCandPtr->pz() );
      t.fV = TEveVector( pfCandPtr->vertex().x(), pfCandPtr->vertex().y(), pfCandPtr->vertex().z() );
      t.fSign = pfCandPtr->charge();
      TEveTrack* trk = new TEveTrack(&t, context().getTrackPropagator());
      trk->MakeTrack();
      trk->SetLineWidth(3);

      fireworks::setTrackTypePF( *pfCandPtr, trk );

      setupAddElement( trk, &oItemHolder );
   }
}

//
// const member functions
//

//
// static member functions
//

REGISTER_FWPROXYBUILDER(FWPFJetRPZProxyBuilder, reco::PFJet, "PFJet", FWViewType::kAllRPZBits);
