#include "CondFormats/GEMObjects/interface/GEMeMap.h"
#include "CondFormats/GEMObjects/interface/GEMROMapping.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

GEMeMap::GEMeMap() : theVersion("") {}

GEMeMap::GEMeMap(const std::string& version) : theVersion(version) {}

GEMeMap::~GEMeMap() {}

const std::string& GEMeMap::version() const { return theVersion; }

void GEMeMap::convert(GEMROMapping& romap) {
  // fed->amc->geb mapping to GEMDetId
  for (auto imap : theChamberMap_) {
    for (unsigned int ix = 0; ix < imap.fedId.size(); ix++) {
      GEMROMapping::chamEC ec;
      ec.fedId = imap.fedId[ix];
      ec.amcNum = imap.amcNum[ix];
      ec.gebId = imap.gebId[ix];

      GEMROMapping::chamDC dc;
      dc.detId = GEMDetId((imap.gemNum[ix] > 0) ? 1 : -1,
                          1,
                          abs(imap.gemNum[ix] / 1000),
                          abs(imap.gemNum[ix] / 100 % 10),
                          abs(imap.gemNum[ix] % 100),
                          0);
      dc.vfatVer = imap.vfatVer[ix];

      romap.add(ec, dc);
    }
  }
  // chamberType to vfatType
  for (auto imap : theVFatMap_) {
    for (unsigned int ix = 0; ix < imap.vfatAdd.size(); ix++) {
      GEMDetId gemId((imap.gemNum[ix] > 0) ? 1 : -1,
                     1,
                     abs(imap.gemNum[ix] / 1000),
                     abs(imap.gemNum[ix] / 100 % 10),
                     abs(imap.gemNum[ix] % 100),
                     imap.iEta[ix]);

      GEMROMapping::vfatEC ec;
      ec.detId = gemId.chamberId();
      ec.vfatAdd = imap.vfatAdd[ix] & chipIdMask_;

      GEMROMapping::vfatDC dc;
      dc.vfatType = imap.vfatType[ix];
      dc.detId = gemId;
      dc.localPhi = imap.localPhi[ix];

      romap.add(ec, dc);
      romap.add(gemId.chamberId(), ec);
    }
  }
  // channel mapping
  for (auto imap : theStripMap_) {
    for (unsigned int ix = 0; ix < imap.vfatType.size(); ix++) {
      GEMROMapping::channelNum cMap;
      cMap.vfatType = imap.vfatType[ix];
      cMap.chNum = imap.vfatCh[ix];

      GEMROMapping::stripNum sMap;
      sMap.vfatType = imap.vfatType[ix];
      sMap.stNum = imap.vfatStrip[ix];

      romap.add(cMap, sMap);
      romap.add(sMap, cMap);
    }
  }
}

void GEMeMap::convertDummy(GEMROMapping& romap) {
  // 12 bits for vfat, 5 bits for geb, 8 bit long GLIB serial number
  unsigned int fedId = FEDNumbering::MINGEMFEDID;
  uint8_t amcNum = 0;  //amc
  uint8_t gebId = 0;

  for (int re = -1; re <= 1; re = re + 2) {
    for (int st = GEMDetId::minStationId; st <= GEMDetId::maxStationId; ++st) {
      int maxVFat = maxVFatGE11_;
      if (st == 2)
        maxVFat = maxVFatGE21_;
      if (st == 0)
        maxVFat = maxVFatGE0_;

      for (int ch = 1; ch <= GEMDetId::maxChamberId; ++ch) {
        for (int ly = 1; ly <= GEMDetId::maxLayerId; ++ly) {
          GEMDetId gemId(re, 1, st, ly, ch, 0);

          GEMROMapping::chamEC ec;
          ec.fedId = fedId;
          ec.gebId = gebId;
          ec.amcNum = amcNum;

          GEMROMapping::chamDC dc;
          dc.detId = gemId;
          dc.vfatVer = vfatVerV3_;

          romap.add(ec, dc);

          uint16_t chipPos = 0;
          for (int lphi = 0; lphi < maxVFat; ++lphi) {
            for (int roll = 1; roll <= maxEtaPartition_; ++roll) {
              GEMROMapping::vfatEC vec;
              vec.vfatAdd = chipPos;
              vec.detId = gemId;

              GEMROMapping::vfatDC vdc;
              vdc.vfatType = vfatTypeV3_;  // > 10 is vfat v3
              vdc.detId = GEMDetId(re, 1, st, ly, ch, roll);
              vdc.localPhi = lphi;

              romap.add(vec, vdc);
              romap.add(gemId.chamberId(), vec);

              chipPos++;
            }
          }

          // 1 geb per chamber
          gebId++;
          // 5 bits for gebId
          if (gebId == maxGEBs_) {
            // 24 gebs per amc
            gebId = 0;
            amcNum++;
          }
          if (amcNum == maxAMCs_) {
            gebId = 0;
            amcNum = 0;
            fedId++;
          }
        }
      }
    }
  }

  for (int i = 0; i < maxChan_; ++i) {
    // only 1 vfat type for dummy map
    GEMROMapping::channelNum cMap;
    cMap.vfatType = vfatTypeV3_;
    cMap.chNum = i;

    GEMROMapping::stripNum sMap;
    sMap.vfatType = vfatTypeV3_;
    sMap.stNum = i;

    romap.add(cMap, sMap);
    romap.add(sMap, cMap);
  }
}
