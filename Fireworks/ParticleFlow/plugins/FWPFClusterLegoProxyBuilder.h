#ifndef _FWPFCLUSTERLEGOPROXYBUILDER_H_
#define _FWPFCLUSTERLEGOPROXYBUILDER_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFClusterLegoProxyBuilder, FWPFEcalClusterLegoProxyBuilder, FWPFHcalClusterLegoProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//

// System include files
#include <cmath>
#include "TEveBox.h"

// User include files
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "Fireworks/ParticleFlow/interface/FWPFMaths.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/Context.h"

//-----------------------------------------------------------------------------
// FWPFClusterLegoProxyBuilder
//-----------------------------------------------------------------------------

class FWPFClusterLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFCluster> {
public:
  static std::string typeOfBuilder() { return "simple#"; }

  // ---------------- Constructor(s)/Destructor ----------------------
  FWPFClusterLegoProxyBuilder() {}
  ~FWPFClusterLegoProxyBuilder() override {}

  // --------------------- Member Functions --------------------------
  using FWSimpleProxyBuilderTemplate<reco::PFCluster>::scaleProduct;
  void scaleProduct(TEveElementList* parent, FWViewType::EType, const FWViewContext* vc) override;
  using FWSimpleProxyBuilderTemplate<reco::PFCluster>::havePerViewProduct;
  bool havePerViewProduct(FWViewType::EType) const override { return true; }
  using FWSimpleProxyBuilderTemplate<reco::PFCluster>::localModelChanges;
  void localModelChanges(const FWModelId& iId,
                         TEveElement* el,
                         FWViewType::EType viewType,
                         const FWViewContext* vc) override;

  REGISTER_PROXYBUILDER_METHODS();

protected:
  // --------------------- Member Functions --------------------------
  void sharedBuild(const reco::PFCluster&, TEveElement&, const FWViewContext*);
  float calculateEt(const reco::PFCluster& cluster, float E);

private:
  // Disable default copy constructor
  FWPFClusterLegoProxyBuilder(const FWPFClusterLegoProxyBuilder&);
  // Disable default assignment operator
  const FWPFClusterLegoProxyBuilder& operator=(const FWPFClusterLegoProxyBuilder&);
};
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_

//-----------------------------------------------------------------------------
// FWPFEcalClusterLegoProxyBuilder
//-----------------------------------------------------------------------------

class FWPFEcalClusterLegoProxyBuilder : public FWPFClusterLegoProxyBuilder {
public:
  // ---------------- Constructor(s)/Destructor ----------------------
  FWPFEcalClusterLegoProxyBuilder() {}
  ~FWPFEcalClusterLegoProxyBuilder() override {}

  // --------------------- Member Functions --------------------------
  using FWSimpleProxyBuilderTemplate<reco::PFCluster>::build;
  void build(const reco::PFCluster&, unsigned int, TEveElement&, const FWViewContext*) override;

  REGISTER_PROXYBUILDER_METHODS();

  FWPFEcalClusterLegoProxyBuilder(const FWPFEcalClusterLegoProxyBuilder&) = delete;
  const FWPFEcalClusterLegoProxyBuilder& operator=(const FWPFEcalClusterLegoProxyBuilder&) = delete;
};
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_

//-----------------------------------------------------------------------------
// FWPFHcalClusterLegoProxyBuilder
//-----------------------------------------------------------------------------

class FWPFHcalClusterLegoProxyBuilder : public FWPFClusterLegoProxyBuilder {
public:
  // ---------------- Constructor(s)/Destructor ----------------------
  FWPFHcalClusterLegoProxyBuilder() {}
  ~FWPFHcalClusterLegoProxyBuilder() override {}

  // --------------------- Member Functions --------------------------
  using FWSimpleProxyBuilderTemplate<reco::PFCluster>::build;
  void build(const reco::PFCluster&, unsigned int, TEveElement&, const FWViewContext*) override;

  REGISTER_PROXYBUILDER_METHODS();

  FWPFHcalClusterLegoProxyBuilder(const FWPFHcalClusterLegoProxyBuilder&) = delete;
  const FWPFHcalClusterLegoProxyBuilder& operator=(const FWPFHcalClusterLegoProxyBuilder&) = delete;
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
