// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWCandidate3DProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Colin Bernet
//         Created:  Fri May 28 15:58:19 CEST 2010
// Edited:           sharris, Wed 9 Feb 2011, 17:34
//

// System include files
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

// User include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "Fireworks/Core/src/CmsShowMain.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/ParticleFlow/interface/setTrackTypePF.h"

//-----------------------------------------------------------------------------
// FWPFCandidate3DProxyBuilder
//-----------------------------------------------------------------------------

class FWPFCandidate3DProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFCandidate> {
public:
  // ---------------- Constructor(s)/Destructor ----------------------
  FWPFCandidate3DProxyBuilder() {}
  ~FWPFCandidate3DProxyBuilder() override;

  REGISTER_PROXYBUILDER_METHODS();

private:
  FWPFCandidate3DProxyBuilder(const FWPFCandidate3DProxyBuilder&) = delete;                   // Stop default
  const FWPFCandidate3DProxyBuilder& operator=(const FWPFCandidate3DProxyBuilder&) = delete;  // Stop default

  // --------------------- Member Functions --------------------------
  using FWSimpleProxyBuilderTemplate<reco::PFCandidate>::build;
  void build(const reco::PFCandidate& iData,
             unsigned int iIndex,
             TEveElement& oItemHolder,
             const FWViewContext*) override;
};
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_

//______________________________________________________________________________
FWPFCandidate3DProxyBuilder::~FWPFCandidate3DProxyBuilder() {}

//______________________________________________________________________________
void FWPFCandidate3DProxyBuilder::build(const reco::PFCandidate& iData,
                                        unsigned int iIndex,
                                        TEveElement& oItemHolder,
                                        const FWViewContext*) {
  TEveRecTrack t;
  t.fBeta = 1.;
  t.fP = TEveVector(iData.px(), iData.py(), iData.pz());
  t.fV = TEveVector(iData.vertex().x(), iData.vertex().y(), iData.vertex().z());
  t.fSign = iData.charge();
  TEveTrack* trk = new TEveTrack(&t, context().getTrackPropagator());

  trk->MakeTrack();

  fireworks::setTrackTypePF(iData, trk);
  setupAddElement(trk, &oItemHolder);
}

//______________________________________________________________________________
REGISTER_FWPROXYBUILDER(FWPFCandidate3DProxyBuilder,
                        reco::PFCandidate,
                        "PF Candidates",
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
