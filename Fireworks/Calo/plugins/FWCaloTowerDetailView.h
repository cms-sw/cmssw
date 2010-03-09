// -*- C++ -*-
#ifndef Fireworks_Calo_FWHCalTowerDetailView_h
#define Fireworks_Calo_FWHCalTowerDetailView_h

//
// Package:     Electrons
// Class  :     FWHCalTowerDetailView
// $Id: FWHCalTowerDetailView.h,v 1.5 2010/01/14 15:55:14 amraktad Exp $
//

// user include files
#include "Fireworks/Core/interface/FWDetailViewGL.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"


class FWECALDetailViewBuilder;
namespace reco {
  class HCalTower;
}

class FWCaloTowerDetailView : public FWDetailViewGL<CaloTower> {

public:
   FWCaloTowerDetailView();
   virtual ~FWCaloTowerDetailView(); 


private:
   virtual void build (const FWModelId &id, const CaloTower*);
   virtual void setTextInfo(const FWModelId&, const CaloTower*);
};

#endif
