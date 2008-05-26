#ifndef DATAFORMATS_CALOTOWERS_CALOTOWERFWD_H
#define DATAFORMATS_CALOTOWERS_CALOTOWERFWD_H 1

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

class CaloTower;
typedef edm::Ptr<CaloTower> CaloTowerPtr;
typedef edm::SortedCollection<CaloTower> CaloTowerCollection;
typedef edm::Ref<CaloTowerCollection> CaloTowerRef;
typedef edm::RefVector<CaloTowerCollection> CaloTowerRefs;
typedef edm::RefProd<CaloTowerCollection> CaloTowersRef;

#endif
