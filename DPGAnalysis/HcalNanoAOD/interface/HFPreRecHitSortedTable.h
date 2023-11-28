#ifndef HFPreRecHitSortedTable_h
#define HFPreRecHitSortedTable_h

#include <vector>
#include <map>

#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

class HFPreRecHitSortedTable {
public:
  std::vector<HcalDetId> dids_;
  std::map<HcalDetId, unsigned int> did_indexmap_;  // Use std::map for efficient lookup, rather than std::find

  std::vector<int> charges_;
  std::vector<int> chargeAsymmetries_;
  std::vector<bool> valids_;

  HFPreRecHitSortedTable(const std::vector<HcalDetId>& dids);
  void add(const HFPreRecHitCollection::const_iterator itPreRecHit);
  void reset();
};
#endif
