#ifndef HODigiSortedTable_h
#define HODigiSortedTable_h

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
#include "DQM/HcalCommon/interface/Utilities.h"

class HODigiSortedTable {
public:
  std::vector<HcalDetId> dids_;
  std::map<HcalDetId, unsigned int> did_indexmap_;  // Use std::map for efficient lookup, rather than std::find

  std::vector<int> ietas_;
  std::vector<int> iphis_;
  std::vector<int> subdets_;
  std::vector<int> depths_;
  std::vector<int> rawIds_;
  std::vector<int> fiberIdleOffsets_;
  std::vector<int> sois_;
  std::vector<bool> valids_;

  unsigned int nTS_;
  std::vector<std::vector<int>> adcs_;
  std::vector<std::vector<float>> fcs_;
  std::vector<std::vector<float>> pedestalfcs_;
  std::vector<std::vector<int>> capids_;
  std::vector<std::vector<int>> fibers_;
  std::vector<std::vector<int>> fiberChans_;
  std::vector<std::vector<int>> dvs_;
  std::vector<std::vector<int>> ers_;

  HODigiSortedTable(const std::vector<HcalDetId>& dids, const unsigned int nTS);
  void add(const HODataFrame* digi, const edm::ESHandle<HcalDbService>& dbService);
  void reset();
};

#endif
