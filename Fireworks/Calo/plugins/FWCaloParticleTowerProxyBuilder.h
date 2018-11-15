#ifndef Fireworks_Calo_FWCaloParticleTowerProxyBuilder_h
#define Fireworks_Calo_FWCaloParticleTowerProxyBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloParticleTowerProxyBuilderBase
//
/**\class FWCaloParticleTowerProxyBuilderBase FWCaloParticleTowerProxyBuilderBase.h Fireworks/Calo/interface/FWCaloParticleTowerProxyBuilderBase.h

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
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

class FWHistSliceSelector;
//
// base
//

class FWCaloParticleTowerProxyBuilderBase : public FWCaloDataHistProxyBuilder
{
public:
   FWCaloParticleTowerProxyBuilderBase();
   ~FWCaloParticleTowerProxyBuilderBase() override;   

   REGISTER_PROXYBUILDER_METHODS();

private:
   void fillCaloData() override;
   FWHistSliceSelector* instantiateSliceSelector() override;
   void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) override;

   FWCaloParticleTowerProxyBuilderBase(const FWCaloParticleTowerProxyBuilderBase&) = delete; // stop default
   const FWCaloParticleTowerProxyBuilderBase& operator=(const FWCaloParticleTowerProxyBuilderBase&) = delete; // stop default

   const CaloParticleCollection* m_towers;
};

#endif
