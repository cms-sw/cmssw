#ifndef _FWPFPATJET3DPROXYBUILDER__
#define _FWPFPATJET3DPROXYBUILDER__

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFPatJet3DProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//

// System include files
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveVSDStructs.h"

// User include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/ParticleFlow/interface/setTrackTypePF.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

//-----------------------------------------------------------------------------
// FWPFPatJet3DProxyBuilder
//-----------------------------------------------------------------------------
template <class T>
class FWPFPatJet3DProxyBuilder : public FWSimpleProxyBuilderTemplate<T> {
public:
  // ---------------- Constructor(s)/Destructor ----------------------
  FWPFPatJet3DProxyBuilder();
  ~FWPFPatJet3DProxyBuilder() override;

  FWPFPatJet3DProxyBuilder(const FWPFPatJet3DProxyBuilder&) = delete;                   // Stop default
  const FWPFPatJet3DProxyBuilder& operator=(const FWPFPatJet3DProxyBuilder&) = delete;  // Stop default

private:
  // --------------------- Member Functions --------------------------
  using FWSimpleProxyBuilderTemplate<T>::build;
  void build(const T&, unsigned int, TEveElement&, const FWViewContext*) override;
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
