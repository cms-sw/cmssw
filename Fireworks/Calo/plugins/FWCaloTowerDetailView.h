// -*- C++ -*-
#ifndef Fireworks_Calo_FWHCalTowerDetailView_h
#define Fireworks_Calo_FWHCalTowerDetailView_h

//
// Package:     Electrons
// Class  :     FWHCalTowerDetailView
//

// user include files
#include "Fireworks/Core/interface/FWDetailViewGL.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"


class FWECALDetailViewBuilder;
class TEveCaloData;

class FWCaloTowerDetailView : public FWDetailViewGL<CaloTower> {

public:
   FWCaloTowerDetailView();
   ~FWCaloTowerDetailView() override; 

   using FWDetailViewGL<CaloTower>::build;
   void build (const FWModelId &id, const CaloTower*) override;
private:
   void setTextInfo(const FWModelId&, const CaloTower*) override;
   TEveCaloData* m_data;
   FWECALDetailViewBuilder* m_builder;
};

#endif
