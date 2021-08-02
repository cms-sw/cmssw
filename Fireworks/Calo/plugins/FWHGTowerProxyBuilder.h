#ifndef Fireworks_Calo_FWHGTowerProxyBuilder_h
#define Fireworks_Calo_FWHGTowerProxyBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWHGTowerProxyBuilder
//
/**\class FWHGTowerProxyBuilder FWHGTowerProxyBuilder.h Fireworks/Calo/interface/FWHGTowerProxyBuilder.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Mon May 31 16:41:23 CEST 2010
//

// system include files

// user include files
#include "Fireworks/Calo/interface/FWCaloDataProxyBuilderBase.h"
#include "Fireworks/Calo/src/FWFromTEveCaloDataSelector.h"
// #include "DataFormats/HGCRecHit/interface/HGRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

class TEveCaloDataVec;
//
// base
//
class FWHGTowerProxyBuilderBase : public FWCaloDataProxyBuilderBase {
public:
  FWHGTowerProxyBuilderBase();
  ~FWHGTowerProxyBuilderBase() override;

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  REGISTER_PROXYBUILDER_METHODS();

protected:
  void setCaloData(const fireworks::Context&) override;
  void fillCaloData() override;
  bool assertCaloDataSlice() override;

  void itemBeingDestroyed(const FWEventItem*) override;

public:
  FWHGTowerProxyBuilderBase(const FWHGTowerProxyBuilderBase&) = delete;  // stop default

  const FWHGTowerProxyBuilderBase& operator=(const FWHGTowerProxyBuilderBase&) = delete;  // stop default
private:
  void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) override;

  int fillTowerForDetId(unsigned int rawid, float);
  // ---------- member data --------------------------------

  const HGCRecHitCollection* m_hits;
  //   int   m_depth;
  TEveCaloDataVec* m_vecData;
};

#endif
