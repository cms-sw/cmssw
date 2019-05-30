#ifndef Fireworks_Calo_FWCaloTowerProxyBuilder_h
#define Fireworks_Calo_FWCaloTowerProxyBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloTowerProxyBuilderBase
//
/**\class FWCaloTowerProxyBuilderBase FWCaloTowerProxyBuilderBase.h Fireworks/Calo/interface/FWCaloTowerProxyBuilderBase.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Wed Dec  3 11:28:08 EST 2008
//

#include "Rtypes.h"
#include <string>

#include "Fireworks/Calo/interface/FWCaloDataHistProxyBuilder.h"
#include "Fireworks/Calo/src/FWFromTEveCaloDataSelector.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDefs.h"

class FWHistSliceSelector;
//
// base
//

class FWCaloTowerProxyBuilderBase : public FWCaloDataHistProxyBuilder {
public:
  FWCaloTowerProxyBuilderBase();
  ~FWCaloTowerProxyBuilderBase() override;

  virtual double getEt(const CaloTower&) const = 0;

protected:
  void fillCaloData() override;
  FWHistSliceSelector* instantiateSliceSelector() override;
  void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) override;

private:
  FWCaloTowerProxyBuilderBase(const FWCaloTowerProxyBuilderBase&) = delete;                   // stop default
  const FWCaloTowerProxyBuilderBase& operator=(const FWCaloTowerProxyBuilderBase&) = delete;  // stop default

  const CaloTowerCollection* m_towers;
};

//
// Ecal
//

class FWECalCaloTowerProxyBuilder : public FWCaloTowerProxyBuilderBase {
public:
  FWECalCaloTowerProxyBuilder() {}
  ~FWECalCaloTowerProxyBuilder() override {}

  // ---------- const member functions ---------------------

  double getEt(const CaloTower& iTower) const override { return iTower.emEt(); }

  REGISTER_PROXYBUILDER_METHODS();

private:
  FWECalCaloTowerProxyBuilder(const FWECalCaloTowerProxyBuilder&) = delete;                   // stop default
  const FWECalCaloTowerProxyBuilder& operator=(const FWECalCaloTowerProxyBuilder&) = delete;  // stop default
};

//
// Hcal
//

class FWHCalCaloTowerProxyBuilder : public FWCaloTowerProxyBuilderBase {
public:
  FWHCalCaloTowerProxyBuilder() {}
  ~FWHCalCaloTowerProxyBuilder() override {}

  // ---------- const member functions ---------------------

  double getEt(const CaloTower& iTower) const override { return iTower.hadEt(); }

  REGISTER_PROXYBUILDER_METHODS();

private:
  FWHCalCaloTowerProxyBuilder(const FWHCalCaloTowerProxyBuilder&) = delete;  // stop default

  const FWHCalCaloTowerProxyBuilder& operator=(const FWHCalCaloTowerProxyBuilder&) = delete;  // stop default
};

//
// HCal Outer
//

class FWHOCaloTowerProxyBuilder : public FWCaloTowerProxyBuilderBase {
public:
  FWHOCaloTowerProxyBuilder() {}
  ~FWHOCaloTowerProxyBuilder() override {}

  // ---------- const member functions ---------------------

  double getEt(const CaloTower& iTower) const override { return iTower.outerEt(); }

  REGISTER_PROXYBUILDER_METHODS();

private:
  FWHOCaloTowerProxyBuilder(const FWHOCaloTowerProxyBuilder&) = delete;                   // stop default
  const FWHOCaloTowerProxyBuilder& operator=(const FWHOCaloTowerProxyBuilder&) = delete;  // stop default
};

#endif
