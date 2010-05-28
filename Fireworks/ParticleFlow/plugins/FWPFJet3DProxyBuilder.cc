// -*- C++ -*-
//
// Package: Fireworks
// Class  : FWPFJet3DProxyBuilder

/*

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author: Colin Bernet
//         Created: Fri May 28 15:46:39 2010 
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

class FWPFJet3DProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFJet> {
public:
   FWPFJet3DProxyBuilder();
   virtual ~FWPFJet3DProxyBuilder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------
   /*
   static void buildJet3D(const FWEventItem* iItem,
                          const reco::PFJet* iData,
                          TEveElement& tList);
   */
   // ---------- member functions ---------------------------


   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPFJet3DProxyBuilder(const FWPFJet3DProxyBuilder&); // stop default
   const FWPFJet3DProxyBuilder& operator=(const FWPFJet3DProxyBuilder&); // stop default
   
   void build(const reco::PFJet&, unsigned int, TEveElement&, const FWViewContext*);

   // ---------- member data --------------------------------
};

//
// constructors and destructor
//
FWPFJet3DProxyBuilder::FWPFJet3DProxyBuilder()
{
}

FWPFJet3DProxyBuilder::~FWPFJet3DProxyBuilder()
{
}

//
// member functions
//
void 
FWPFJet3DProxyBuilder::build(const reco::PFJet& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*)
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
/*
void
FWPFJet3DProxyBuilder::buildJet3D(const FWEventItem* iItem,
                                  const reco::PFJet* jet,
                                  TEveElement& container)
{

   FW3DEveJet* cone = new FW3DEveJet(*jet,"jet","jet");
   cone->SetPickable(kTRUE);
   cone->SetMainColor( container.GetMainColor() );
   cone->SetMainTransparency(75);

  container.AddElement( cone );
}
*/
//
// const member functions
//

//
// static member functions
//

REGISTER_FWPROXYBUILDER(FWPFJet3DProxyBuilder, reco::PFJet, "PFJet", FWViewType::kAll3DBits);
