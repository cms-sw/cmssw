// -*- C++ -*-
#ifndef Fireworks_Calo_FWHCalTowerDetailView_h
#define Fireworks_Calo_FWHCalTowerDetailView_h

//
// Package:     Electrons
// Class  :     FWHCalTowerDetailView
// $Id: FWCaloTowerDetailView.h,v 1.1 2010/03/09 21:49:34 amraktad Exp $
//

// user include files
#include "Fireworks/Core/interface/FWDetailViewGL.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"


class FWECALDetailViewBuilder;
class TEveCaloData;

class FWCaloTowerDetailView : public FWDetailViewGL<CaloTower> {

public:
   FWCaloTowerDetailView();
   virtual ~FWCaloTowerDetailView(); 


private:
   virtual void build (const FWModelId &id, const CaloTower*);
   virtual void setTextInfo(const FWModelId&, const CaloTower*);
   TEveCaloData* m_data;
   FWECALDetailViewBuilder* m_builder;
};

#endif
