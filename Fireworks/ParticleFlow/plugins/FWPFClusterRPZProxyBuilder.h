#ifndef _FWPFCLUSTERRPZPROXYBUILDER_H_
#define _FWPFCLUSTERRPZPROXYBUILDER_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFClusterRPZProxyBuilder, FWPFEcalClusterRPZProxyBuilder, FWPFHcalClusterRPZProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//

// User include files
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/ParticleFlow/interface/FWPFGeom.h"
#include "Fireworks/ParticleFlow/interface/FWPFClusterRPZUtils.h"

//-----------------------------------------------------------------------------
// FWPFClusterRPZProxyBuilder
//-----------------------------------------------------------------------------

class FWPFClusterRPZProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFCluster> {
public:
  // ---------------- Constructor(s)/Destructor ----------------------
  FWPFClusterRPZProxyBuilder();
  ~FWPFClusterRPZProxyBuilder() override;

  // --------------------- Member Functions --------------------------
  using FWSimpleProxyBuilderTemplate<reco::PFCluster>::build;
  void build(const reco::PFCluster &iData,
             unsigned int iIndex,
             TEveElement &oItemHolder,
             const FWViewContext *vc) override;
  using FWSimpleProxyBuilderTemplate<reco::PFCluster>::scaleProduct;
  void scaleProduct(TEveElementList *parent, FWViewType::EType, const FWViewContext *vc) override;
  using FWSimpleProxyBuilderTemplate<reco::PFCluster>::havePerViewProduct;
  bool havePerViewProduct(FWViewType::EType) const override { return true; }
  using FWSimpleProxyBuilderTemplate<reco::PFCluster>::cleanLocal;
  void cleanLocal() override { m_clusters.clear(); }

  REGISTER_PROXYBUILDER_METHODS();

protected:
  // ----------------------- Data Members ----------------------------
  std::vector<ScalableLines> m_clusters;
  FWPFClusterRPZUtils *m_clusterUtils;

  // --------------------- Member Functions --------------------------
  virtual void sharedBuild(const reco::PFCluster &cluster,
                           unsigned int iIndex,
                           TEveElement &oItemHolder,
                           const FWViewContext *vc,
                           float radius);

public:
  FWPFClusterRPZProxyBuilder(const FWPFClusterRPZProxyBuilder &) = delete;                   // Disable default
  const FWPFClusterRPZProxyBuilder &operator=(const FWPFClusterRPZProxyBuilder &) = delete;  // Disable default
};
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_

//-----------------------------------------------------------------------------
// FWPFEcalClusterRPZProxyBuilder
//-----------------------------------------------------------------------------

class FWPFEcalClusterRPZProxyBuilder : public FWPFClusterRPZProxyBuilder {
public:
  // ---------------- Constructor(s)/Destructor ----------------------
  FWPFEcalClusterRPZProxyBuilder() {}
  ~FWPFEcalClusterRPZProxyBuilder() override {}

  // --------------------- Member Functions --------------------------
  using FWSimpleProxyBuilderTemplate<reco::PFCluster>::build;
  void build(const reco::PFCluster &iData,
             unsigned int iIndex,
             TEveElement &oItemHolder,
             const FWViewContext *vc) override;

  REGISTER_PROXYBUILDER_METHODS();

  FWPFEcalClusterRPZProxyBuilder(const FWPFEcalClusterRPZProxyBuilder &) = delete;
  const FWPFEcalClusterRPZProxyBuilder &operator=(const FWPFEcalClusterRPZProxyBuilder &) = delete;
};
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_

//-----------------------------------------------------------------------------
// FWPFHcalClusterRPZProxyBuilder
//-----------------------------------------------------------------------------

class FWPFHcalClusterRPZProxyBuilder : public FWPFClusterRPZProxyBuilder {
public:
  // ---------------- Constructor(s)/Destructor ----------------------
  FWPFHcalClusterRPZProxyBuilder() {}
  ~FWPFHcalClusterRPZProxyBuilder() override {}

  // --------------------- Member Functions --------------------------
  using FWSimpleProxyBuilderTemplate<reco::PFCluster>::build;
  void build(const reco::PFCluster &iData,
             unsigned int iIndex,
             TEveElement &oItemHolder,
             const FWViewContext *vc) override;

  REGISTER_PROXYBUILDER_METHODS();

  FWPFHcalClusterRPZProxyBuilder(const FWPFHcalClusterRPZProxyBuilder &) = delete;
  const FWPFHcalClusterRPZProxyBuilder &operator=(const FWPFHcalClusterRPZProxyBuilder &) = delete;
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
