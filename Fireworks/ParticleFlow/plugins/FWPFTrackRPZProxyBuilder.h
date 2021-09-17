#ifndef _FWPFTRACKRPZLEGOPROXYBUILDER_H_
#define _FWPFTRACKRPZLEGOPROXYBUILDER_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFTrackRPZProxyBuilder
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
// FWPFTrackRPZProxyBuilder
//-----------------------------------------------------------------------------
class FWPFTrackRPZProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Track> {
public:
  // ---------------- Constructor(s)/Destructor ----------------------
  FWPFTrackRPZProxyBuilder() {}
  ~FWPFTrackRPZProxyBuilder() override {}
  using FWSimpleProxyBuilderTemplate<reco::Track>::build;
  void build(const reco::Track& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* vc) override;
  REGISTER_PROXYBUILDER_METHODS();

  FWPFTrackRPZProxyBuilder(const FWPFTrackRPZProxyBuilder&) = delete;
  const FWPFTrackRPZProxyBuilder& operator=(const FWPFTrackRPZProxyBuilder&) = delete;

  // --------------------- Member Functions --------------------------
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
