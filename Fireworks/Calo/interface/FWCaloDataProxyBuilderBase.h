#ifndef Fireworks_Calo_FWCaloDataProxyBuilderBase_h
#define Fireworks_Calo_FWCaloDataProxyBuilderBase_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloDataProxyBuilderBase
//
/**\class FWCaloDataProxyBuilderBase FWCaloDataProxyBuilderBase.h Fireworks/Calo/interface/FWCaloDataProxyBuilderBase.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Mon May 31 15:09:19 CEST 2010
//

// system include files
#include <string>

// user include files

#include "Rtypes.h"
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDefs.h"

// forward declarations
class TEveCaloData;

class FWCaloDataProxyBuilderBase : public FWProxyBuilderBase {
public:
  FWCaloDataProxyBuilderBase();
  ~FWCaloDataProxyBuilderBase() override;

  // ---------- const member functions ---------------------

  bool willHandleInteraction() const override { return true; }

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

protected:
  void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) override;

  virtual void setCaloData(const fireworks::Context&) = 0;
  virtual void fillCaloData() = 0;
  virtual bool assertCaloDataSlice() = 0;

  // ---------- member data --------------------------------
  TEveCaloData* m_caloData;
  Int_t m_sliceIndex;
  void itemBeingDestroyed(const FWEventItem*) override;

public:
  FWCaloDataProxyBuilderBase(const FWCaloDataProxyBuilderBase&) = delete;  // stop default

  const FWCaloDataProxyBuilderBase& operator=(const FWCaloDataProxyBuilderBase&) = delete;  // stop default

private:
  // ---------- member data --------------------------------

  void modelChanges(const FWModelIds&, Product*) override;

  void clearCaloDataSelection();
};

#endif
