#ifndef Fireworks_Calo_FWCaloTowerSliceSelector_h
#define Fireworks_Calo_FWCaloTowerSliceSelector_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloTowerSliceSelector
// 
/**\class FWCaloTowerSliceSelector FWCaloTowerSliceSelector.h Fireworks/Calo/interface/FWCaloTowerSliceSelector.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Wed Jun  2 19:21:19 CEST 2010
// $Id: FWCaloTowerSliceSelector.h,v 1.3 2010/12/01 21:40:31 amraktad Exp $
//

// system include files

// user include files

#include "Fireworks/Calo/src/FWFromSliceSelector.h"
class CaloTower;
// forward declarations
class TH2F;

class FWCaloTowerSliceSelector : public FWFromSliceSelector
{
public:
  FWCaloTowerSliceSelector(TH2F* h, const FWEventItem* i);
  virtual ~FWCaloTowerSliceSelector();
  
  virtual void doSelect(const TEveCaloData::CellId_t&);
  virtual void doUnselect(const TEveCaloData::CellId_t&);
  
private:
  TH2F* m_hist;
    bool matchCell(const TEveCaloData::CellId_t& iCell, const CaloTower& tower) const;
};

#endif
