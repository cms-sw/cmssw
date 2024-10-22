#include "DQM/HcalCommon/interface/Utilities.h"
#include <utility>

namespace hcaldqm {
  using namespace constants;
  namespace utilities {
    /*
 *	Useful Detector Functions. For Fast Detector Validity Check
 */
    std::pair<uint16_t, uint16_t> fed2crate(int fed) {
      //  uTCA Crate is split in half
      uint16_t slot = 0;
      if (fed <= FED_VME_MAX) {
        slot = fed % 2 == 0 ? SLOT_uTCA_MIN : SLOT_uTCA_MIN + 6;
      } else if ((fed >= 1100 && fed <= 1117) || (fed >= 1140 && fed <= 1148)) {
        slot = fed >= 1140 ? SLOT_uTCA_MIN + 8 : fed % 2 == 0 ? SLOT_uTCA_MIN : SLOT_uTCA_MIN + 4;
      } else {
        slot = fed % 2 == 0 ? SLOT_uTCA_MIN : SLOT_uTCA_MIN + 6;
      }
      std::pair<uint16_t, uint16_t> crate_slot = std::make_pair<uint16_t, uint16_t>(0, 0);
      auto it_fed2crate = constants::fed2crate_map.find(fed);
      if (it_fed2crate != constants::fed2crate_map.end()) {
        crate_slot =
            std::make_pair<uint16_t const, uint16_t const>((uint16_t const)it_fed2crate->second, (uint16_t const)slot);
      }
      return crate_slot;
    }

    uint16_t crate2fed(int crate, int slot) {
      //	 for the details see Constants.h
      int fed = 0;
      auto it_crate2fed = constants::crate2fed_map.find(crate);
      if (it_crate2fed != constants::crate2fed_map.end()) {
        fed = it_crate2fed->second;
        if (fed <= FED_VME_MAX && fed > 0) {
          if (slot > 10 && (std::find(constants::crateListVME.begin(), constants::crateListVME.end(), crate) !=
                            constants::crateListVME.end())) {
            ++fed;
          }
        } else {
          if (crate == 22 || crate == 29 || crate == 32 || crate == 23 || crate == 27 || crate == 26 ||
              crate == 38) {  // needed to handle dual fed readout for HF and HO
            if (slot > 6 && (std::find(constants::crateListuTCA.begin(), constants::crateListuTCA.end(), crate) !=
                             constants::crateListuTCA.end())) {
              ++fed;  // hard coded mid slot FED numbering
            }
          } else {  // needed to handle  3-FED readout for HBHE
            if (slot > 8 && (std::find(constants::crateListuTCA.begin(), constants::crateListuTCA.end(), crate) !=
                             constants::crateListuTCA.end())) {
              fed = (fed + 1100) / 2 + 40;  // hard coded right slot FED numbering, no better way
            } else if (slot > 4 &&
                       (std::find(constants::crateListuTCA.begin(), constants::crateListuTCA.end(), crate) !=
                        constants::crateListuTCA.end())) {
              ++fed;  // hard coded mid slot FED numbering
            }
          }
        }
      }
      return fed;
    }

    uint32_t hash(HcalDetId const &did) { return did.rawId(); }
    uint32_t hash(HcalElectronicsId const &eid) { return eid.rawId(); }
    uint32_t hash(HcalTrigTowerDetId const &tid) { return tid.rawId(); }

    std::vector<int> getCrateList(HcalElectronicsMap const *emap) {
      std::vector<int> vCrates;
      std::vector<HcalElectronicsId> vids = emap->allElectronicsIdPrecision();
      for (std::vector<HcalElectronicsId>::const_iterator it = vids.begin(); it != vids.end(); ++it) {
        HcalElectronicsId eid = HcalElectronicsId(it->rawId());
        int crate = eid.crateId();
        if (std::find(vCrates.begin(), vCrates.end(), crate) == vCrates.end()) {
          vCrates.push_back(crate);
        }
      }
      std::sort(vCrates.begin(), vCrates.end());
      return vCrates;
    }

    std::map<int, uint32_t> getCrateHashMap(HcalElectronicsMap const *emap) {
      std::map<int, uint32_t> crateHashMap;
      std::vector<HcalElectronicsId> vids = emap->allElectronicsIdPrecision();
      for (std::vector<HcalElectronicsId>::const_iterator it = vids.begin(); it != vids.end(); ++it) {
        HcalElectronicsId eid = HcalElectronicsId(it->rawId());
        int this_crate = eid.crateId();
        uint32_t this_hash =
            (eid.isVMEid()
                 ? utilities::hash(HcalElectronicsId(FIBERCH_MIN, FIBER_VME_MIN, eid.spigot(), eid.dccid()))
                 : utilities::hash(HcalElectronicsId(eid.crateId(), eid.slot(), FIBER_uTCA_MIN1, FIBERCH_MIN, false)));
        if (crateHashMap.find(this_crate) == crateHashMap.end()) {
          crateHashMap[this_crate] = this_hash;
        }
      }
      return crateHashMap;
    }

    std::vector<int> getFEDList(HcalElectronicsMap const *emap) {
      std::vector<int> vfeds;
      std::vector<HcalElectronicsId> vids = emap->allElectronicsIdPrecision();
      for (std::vector<HcalElectronicsId>::const_iterator it = vids.begin(); it != vids.end(); ++it) {
        int fed = it->isVMEid() ? it->dccid() + FED_VME_MIN : crate2fed(it->crateId(), it->slot());
        uint32_t n = 0;
        for (std::vector<int>::const_iterator jt = vfeds.begin(); jt != vfeds.end(); ++jt)
          if (fed == *jt)
            break;
          else
            n++;
        if (n == vfeds.size())
          vfeds.push_back(fed);
      }

      std::sort(vfeds.begin(), vfeds.end());
      return vfeds;
    }
    std::vector<int> getFEDVMEList(HcalElectronicsMap const *emap) {
      std::vector<int> vfeds;
      std::vector<HcalElectronicsId> vids = emap->allElectronicsIdPrecision();
      for (std::vector<HcalElectronicsId>::const_iterator it = vids.begin(); it != vids.end(); ++it) {
        if (!it->isVMEid())
          continue;
        int fed = it->isVMEid() ? it->dccid() + FED_VME_MIN : crate2fed(it->crateId(), it->slot());
        uint32_t n = 0;
        for (std::vector<int>::const_iterator jt = vfeds.begin(); jt != vfeds.end(); ++jt)
          if (fed == *jt)
            break;
          else
            n++;
        if (n == vfeds.size())
          vfeds.push_back(fed);
      }

      std::sort(vfeds.begin(), vfeds.end());
      return vfeds;
    }
    std::vector<int> getFEDuTCAList(HcalElectronicsMap const *emap) {
      std::vector<int> vfeds;
      std::vector<HcalElectronicsId> vids = emap->allElectronicsIdPrecision();
      for (std::vector<HcalElectronicsId>::const_iterator it = vids.begin(); it != vids.end(); ++it) {
        if (it->isVMEid())
          continue;
        int fed = it->isVMEid() ? it->dccid() + FED_VME_MIN : crate2fed(it->crateId(), it->slot());
        uint32_t n = 0;
        for (std::vector<int>::const_iterator jt = vfeds.begin(); jt != vfeds.end(); ++jt)
          if (fed == *jt)
            break;
          else
            n++;
        if (n == vfeds.size())
          vfeds.push_back(fed);
      }

      std::sort(vfeds.begin(), vfeds.end());
      return vfeds;
    }

    bool isFEDHBHE(HcalElectronicsId const &eid) {
      if (eid.isVMEid()) {
        return false;
      } else {
        int fed = crate2fed(eid.crateId(), eid.slot());
        if ((fed >= 1100 && fed < 1118) || (fed >= 1140 && fed <= 1148))
          return true;
        else
          return false;
      }

      return false;
    }

    bool isFEDHF(HcalElectronicsId const &eid) {
      if (eid.isVMEid())
        return false;
      int fed = crate2fed(eid.crateId(), eid.slot());
      if (fed >= 1118 && fed <= 1123)
        return true;
      else
        return false;

      return false;
    }

    bool isFEDHO(HcalElectronicsId const &eid) {
      if (eid.isVMEid())
        return false;
      int fed = crate2fed(eid.crateId(), eid.slot());
      if (fed >= 1124 && fed <= 1135)
        return true;
      else
        return false;

      return false;
    }

    /*
 *	Orbit Gap Related
 */
    std::string ogtype2string(OrbitGapType type) {
      switch (type) {
        case tNull:
          return "Null";
        case tPhysics:
          return "Physics";
        case tPedestal:
          return "Pedestal";
        case tLED:
          return "LED";
        case tHFRaddam:
          return "HFRaddam";
        case tHBHEHPD:
          return "HBHEHPD";
        case tHO:
          return "HO";
        case tHF:
          return "HF";
        case tZDC:
          return "ZDC";
        case tHEPMega:
          return "HEPMegatile";
        case tHEMMega:
          return "HEMMegatile";
        case tHBPMega:
          return "HBPMegatile";
        case tHBMMega:
          return "HBMMegatile";
        case tCRF:
          return "CRF";
        case tCalib:
          return "Calib";
        case tSafe:
          return "Safe";
        case tSiPMPMT:
          return "SiPM-PMT";
        case tMegatile:
          return "Megatile";
        case tUnknown:
          return "Unknown";
        default:
          return "Null";
      }
    }

    int getRBX(uint32_t iphi) { return (((iphi + 2) % 72) + 4 - 1) / 4; }

  }  // namespace utilities
}  // namespace hcaldqm
