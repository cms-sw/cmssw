#ifndef _FWPFPATJETLEGOPROXYBUILDER_H_
#define _FWPFPATJETLEGOPROXYBUILDER_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFPatJetLegoProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//

// User include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "Fireworks/ParticleFlow/interface/setTrackTypePF.h"

//-----------------------------------------------------------------------------
// FWPFPatJetLegoProxyBuilder
//-----------------------------------------------------------------------------
template <class T>
class FWPFPatJetLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<T> {
public:
  // ---------------- Constructor(s)/Destructor ----------------------
  FWPFPatJetLegoProxyBuilder();
  ~FWPFPatJetLegoProxyBuilder() override;

  // --------------------- Member Functions --------------------------
  using FWProxyBuilderBase::havePerViewProduct;
  bool havePerViewProduct(FWViewType::EType) const override { return true; }

  using FWProxyBuilderBase::scaleProduct;
  void scaleProduct(TEveElementList* parent, FWViewType::EType, const FWViewContext* vc) override;

  using FWProxyBuilderBase::localModelChanges;
  void localModelChanges(const FWModelId& iId,
                         TEveElement* iCompound,
                         FWViewType::EType viewType,
                         const FWViewContext* vc) override;

  using FWSimpleProxyBuilderTemplate<T>::build;
  void build(const T&, unsigned int, TEveElement&, const FWViewContext*) override;

  FWPFPatJetLegoProxyBuilder(const FWPFPatJetLegoProxyBuilder&) = delete;             //stop default
  const FWPFPatJetLegoProxyBuilder& operator=(FWPFPatJetLegoProxyBuilder&) = delete;  //stop default

  // --------------------- Member Functions --------------------------
};
#endif  // FWPFPATJETLEGOPROXYBUILDER
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
