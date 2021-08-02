#ifndef _FWPFTRACK3DPROXYBUILDER_H_
#define _FWPFTRACK3DPROXYBUILDER_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFTrack3DProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//

// User include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Tracks/interface/estimate_field.h"
#include "Fireworks/ParticleFlow/interface/FWPFTrackUtils.h"

//-----------------------------------------------------------------------------
// FWPFTrack3DProxyBuilder
//-----------------------------------------------------------------------------
class FWPFTrack3DProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Track> {
public:
  // ---------------- Constructor(s)/Destructor ----------------------
  FWPFTrack3DProxyBuilder() {}
  ~FWPFTrack3DProxyBuilder() override {}

  using FWSimpleProxyBuilderTemplate<reco::Track>::build;
  void build(const reco::Track& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* vc) override;
  REGISTER_PROXYBUILDER_METHODS();

  FWPFTrack3DProxyBuilder(const FWPFTrack3DProxyBuilder&) = delete;
  const FWPFTrack3DProxyBuilder& operator=(const FWPFTrack3DProxyBuilder&) = delete;

  // --------------------- Member Functions --------------------------
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
