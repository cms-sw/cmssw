#include "CondFormats/GEMObjects/interface/GEMChMap.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

GEMChMap::GEMChMap() : theVersion("") {}

GEMChMap::GEMChMap(const std::string& version) : theVersion(version) {}

GEMChMap::~GEMChMap() {}

const std::string& GEMChMap::version() const { return theVersion; }

void GEMChMap::setDummy() {
  // 12 bits for vfat, 5 bits for geb, 8 bit long GLIB serial number
  amcVec_.clear();

  chamberMap_.clear();

  chamVfats_.clear();
  chamIEtas_.clear();

  chStMap_.clear();
  stChMap_.clear();

  unsigned int fedId = 0;

  for (int st = GEMDetId::minStationId0; st <= GEMDetId::maxStationId; ++st) {
    int maxVFat = 0;
    int maxLayerId = GEMDetId::maxLayerId;
    int maxiEtaId = 0;
    if (st == 0) {
      maxVFat = maxVFatGE0_;
      maxLayerId = GEMDetId::maxLayerId0;
      maxiEtaId = maxiEtaIdGE0_;
    } else if (st == 1) {
      maxVFat = maxVFatGE11_;
      maxiEtaId = maxiEtaIdGE11_;
    } else if (st == 2) {
      maxVFat = maxVFatGE21_;
      maxiEtaId = maxiEtaIdGE21_;
    }

    uint16_t chipPos = 0;
    for (int lphi = 0; lphi < maxVFat; ++lphi) {
      for (int ieta = 1; ieta <= maxiEtaId; ++ieta) {
        if (st == 2 and ieta % 2 == 0)
          continue;
        for (int i = 0; i < maxChan_; ++i) {
          // only 1 vfat type for dummy map
          GEMChMap::channelNum cMap;
          cMap.chamberType = st;
          cMap.vfatAdd = chipPos;
          cMap.chNum = i;

          GEMChMap::stripNum sMap;
          sMap.chamberType = st;
          if (st != 2) {
            sMap.iEta = ieta;
            sMap.stNum = i + lphi * maxChan_;
          } else {
            sMap.iEta = ieta + i % 2;
            sMap.stNum = i / 2 + lphi * maxChan_ / 2;
          }

          add(cMap, sMap);
          add(sMap, cMap);

          GEMChMap::vfatEC ec;
          ec.vfatAdd = cMap.vfatAdd;
          ec.chamberType = st;

          add(cMap.chamberType, cMap.vfatAdd);
          add(ec, sMap.iEta);
        }
        chipPos++;
      }
    }

    for (int re = -1; re <= 1; re = re + 2) {
      uint8_t amcNum = 1;  //amc
      uint8_t gebId = 0;
      if (st == 0)
        fedId = (re == 1 ? FEDNumbering::MINGE0FEDID + 1 : FEDNumbering::MINGE0FEDID);
      else if (st == 1)
        fedId = (re == 1 ? FEDNumbering::MINGEMFEDID + 1 : FEDNumbering::MINGEMFEDID);
      else if (st == 2)
        fedId = (re == 1 ? FEDNumbering::MINGE21FEDID + 1 : FEDNumbering::MINGE21FEDID);

      for (int ch = 1; ch <= GEMDetId::maxChamberId; ++ch) {
        for (int ly = 1; ly <= maxLayerId; ++ly) {
          GEMDetId gemId(re, 1, st, ly, ch, 0);

          GEMChMap::chamEC ec;
          ec.fedId = fedId;
          ec.gebId = gebId;
          ec.amcNum = amcNum;

          GEMChMap::chamDC dc;
          dc.detId = gemId;
          dc.chamberType = st;
          add(ec, dc);

          GEMChMap::sectorEC amcEC = {fedId, amcNum};
          if (!isValidAMC(fedId, amcNum))
            add(amcEC);

          // 5 bits for gebId
          if (st > 0 && gebId == maxGEB1_) {
            gebId = 0;
            amcNum += 2;  // only odd amc No. is used for GE11
          } else if (st == 0 && gebId == maxGEBs_) {
            gebId = 0;
            amcNum++;
          } else {
            // 1 geb per chamber
            gebId++;
          }
        }
      }
    }
  }
}
