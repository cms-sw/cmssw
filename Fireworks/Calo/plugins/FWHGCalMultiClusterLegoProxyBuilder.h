#ifndef Fireworks_Calo_FWCaloParticleTowerProxyBuilder_h
#define Fireworks_Calo_FWCaloParticleTowerProxyBuilder_h

#include "Rtypes.h"
#include <string>
#include <typeinfo>

#include "Fireworks/Calo/interface/FWCaloDataHistProxyBuilder.h"
#include "Fireworks/Core/interface/FWSimpleProxyHelper.h"

#include "DataFormats/ParticleFlowReco/interface/HGCalMultiCluster.h"

class FWHistSliceSelector;

class FWHGCalMultiClusterLegoProxyBuilder : public FWCaloDataHistProxyBuilder
{
public:
   FWHGCalMultiClusterLegoProxyBuilder();
   ~FWHGCalMultiClusterLegoProxyBuilder() override;   

   // ---------- static member functions --------------------
   ///Used by the plugin system to determine how the proxy uses the data from FWEventItem
   // static std::string typeOfBuilder();

   REGISTER_PROXYBUILDER_METHODS();
private:
   void fillCaloData() override {}; // dummy
   FWHistSliceSelector* instantiateSliceSelector() override;
   void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) override;
   void build(const reco::HGCalMultiCluster& iData, unsigned int iIndex, TEveElement &iItemHolder, const FWViewContext *);

   FWHGCalMultiClusterLegoProxyBuilder(const FWHGCalMultiClusterLegoProxyBuilder&) = delete; // stop default
   const FWHGCalMultiClusterLegoProxyBuilder& operator=(const FWHGCalMultiClusterLegoProxyBuilder&) = delete; // stop default

   FWSimpleProxyHelper m_helper;
};

#endif