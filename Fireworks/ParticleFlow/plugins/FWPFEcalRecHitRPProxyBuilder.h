#ifndef _FWPFECALRECHITRPPROXYBUILDER_H_
#define _FWPFECALRECHITRPPROXYBUILDER_H_

// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWPFEcalRecHitRPProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Simon Harris
//

// System include files
#include "TEveCompound.h"

// User include files
#include "Fireworks/Core/interface/FWProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/Context.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "Fireworks/ParticleFlow/interface/FWPFRhoPhiRecHit.h"
#include "Fireworks/ParticleFlow/interface/FWPFGeom.h"
#include "Fireworks/ParticleFlow/interface/FWPFMaths.h"

#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"

//-----------------------------------------------------------------------------
// FWPFEcalRecHitRPProxyBuilder
//-----------------------------------------------------------------------------
class FWPFEcalRecHitRPProxyBuilder : public FWProxyBuilderTemplate<EcalRecHit> {
public:
  static std::string typeOfBuilder() { return "simple#"; }

  // ---------------- Constructor(s)/Destructor ----------------------
  FWPFEcalRecHitRPProxyBuilder() {}
  ~FWPFEcalRecHitRPProxyBuilder() override {}

  // --------------------- Member Functions --------------------------
  void build(const FWEventItem *iItem, TEveElementList *product, const FWViewContext *) override;

  bool havePerViewProduct(FWViewType::EType) const override { return true; }
  void scaleProduct(TEveElementList *parent, FWViewType::EType, const FWViewContext *vc) override;
  void cleanLocal() override;

  REGISTER_PROXYBUILDER_METHODS();

  FWPFEcalRecHitRPProxyBuilder(const FWPFEcalRecHitRPProxyBuilder &) = delete;                   // Stop default
  const FWPFEcalRecHitRPProxyBuilder &operator=(const FWPFEcalRecHitRPProxyBuilder &) = delete;  // Stop default

private:
  // --------------------- Member Functions --------------------------
  TEveVector calculateCentre(const float *corners);

  // ----------------------- Data Members ----------------------------
  std::vector<FWPFRhoPhiRecHit *> m_towers;
};
#endif
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
