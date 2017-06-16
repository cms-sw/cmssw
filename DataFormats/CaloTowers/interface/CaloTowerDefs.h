#ifndef DATAFORMATS_CALOTOWERS_CALOTOWERFWD_H
#define DATAFORMATS_CALOTOWERS_CALOTOWERFWD_H 1

#include "DataFormats/CaloTowers/interface/CaloTower.h"

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/FwdPtr.h"

typedef edm::Ptr<CaloTower> CaloTowerPtr;
typedef edm::FwdPtr<CaloTower> CaloTowerFwdPtr;
typedef edm::SortedCollection<CaloTower> CaloTowerCollection;
typedef edm::Ref<CaloTowerCollection> CaloTowerRef;
typedef edm::RefVector<CaloTowerCollection> CaloTowerRefs;
typedef edm::RefProd<CaloTowerCollection> CaloTowersRef;
typedef edm::FwdRef<CaloTowerCollection> CaloTowerFwdRef;
typedef edm::PtrVector<CaloTower> CaloTowerPtrVector;
typedef std::vector<CaloTowerFwdRef> CaloTowerFwdRefVector;
typedef std::vector<CaloTowerFwdPtr> CaloTowerFwdPtrVector;

#endif
