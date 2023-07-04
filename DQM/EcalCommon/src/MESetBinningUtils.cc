#include "DQM/EcalCommon/interface/MESetBinningUtils.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "TPRegexp.h"
#include "TObjArray.h"

namespace ecaldqm {
  namespace binning {
    AxisSpecs getBinning(EcalElectronicsMapping const *electronicsMap,
                         ObjectType _otype,
                         BinningType _btype,
                         bool _isMap,
                         int _axis,
                         unsigned _iME) {
      if (_otype >= nObjType || _btype >= unsigned(nPresetBinnings))
        return AxisSpecs();  // you are on your own

      switch (_otype) {
        case kEB:
          return getBinningEB_(_btype, _isMap, _axis);
        case kEE:
          return getBinningEE_(_btype, _isMap, 0, _axis);
        case kEEm:
          return getBinningEE_(_btype, _isMap, -1, _axis);
        case kEEp:
          return getBinningEE_(_btype, _isMap, 1, _axis);
        case kSM:
          return getBinningSM_(_btype, _isMap, _iME, _axis, electronicsMap);
        case kEBSM:
          return getBinningSM_(_btype, _isMap, _iME + 9, _axis, electronicsMap);
        case kEESM:
          if (_iME <= kEEmHigh)
            return getBinningSM_(_btype, _isMap, _iME, _axis, electronicsMap);
          else
            return getBinningSM_(_btype, _isMap, _iME + nEBDCC, _axis, electronicsMap);
        case kSMMEM:
          return getBinningSMMEM_(_btype, _isMap, _iME, _axis);
        case kEBSMMEM:
          return getBinningSMMEM_(_btype, _isMap, _iME + nEEDCCMEM / 2, _axis);
        case kEESMMEM:
          if (_iME <= kEEmHigh)
            return getBinningSMMEM_(_btype, _isMap, _iME, _axis);
          else
            return getBinningSMMEM_(_btype, _isMap, _iME + nEBDCC, _axis);
        case kEcal:
          return getBinningEcal_(_btype, _isMap, _axis);
        case kMEM:
          return getBinningMEM_(_btype, _isMap, -1, _axis);
        case kEBMEM:
          return getBinningMEM_(_btype, _isMap, EcalBarrel, _axis);
        case kEEMEM:
          return getBinningMEM_(_btype, _isMap, EcalEndcap, _axis);
        default:
          return AxisSpecs();
      }
    }

    int findBin1D(EcalElectronicsMapping const *electronicsMap,
                  ObjectType _otype,
                  BinningType _btype,
                  const DetId &_id) {
      switch (_otype) {
        case kSM:
        case kEBSM:
        case kEESM:
          if (_btype == kSuperCrystal)
            return towerId(_id, electronicsMap);
          else if (_btype == kTriggerTower) {
            unsigned tccid(tccId(_id, electronicsMap));
            if (tccid <= 36 || tccid >= 73) {  // EE
              unsigned bin(ttId(_id, electronicsMap));
              bool outer((tccid >= 19 && tccid <= 36) || (tccid >= 73 && tccid <= 90));
              // For the following, the constants nTTInner and nTTOuter are defined in
              // EcalDQMCommonUtils.h.
              if (outer)
                bin += 2 * nTTInner;  // For outer TCCs, sets bin number to increment
              // by twice the number of TTs in inner TCCs, because numbering of bins
              // is in the order (inner1, inner2, outer1, outer2).
              // ("inner"" := closer to the beam)
              bin += (tccid % 2) * (outer ? nTTOuter : nTTInner);  // Yields x-axis bin number
              // in the format above; TTs in even-numbered TCCs are filled in inner1
              // or outer1, and those in odd-numbered TCC are filled in inner2 and
              // outer2.
              return bin;
            } else
              return ttId(_id, electronicsMap);
          } else
            break;
        case kEcal:
          if (_btype == kDCC)
            return dccId(_id, electronicsMap);
          else if (_btype == kTCC)
            return tccId(_id, electronicsMap);
          else
            break;
        case kEB:
          if (_btype == kDCC)
            return dccId(_id, electronicsMap) - 9;
          else if (_btype == kTCC)
            return tccId(_id, electronicsMap) - 36;
          else
            break;
        case kEEm:
          if (_btype == kDCC)
            return dccId(_id, electronicsMap);
          else if (_btype == kTCC)
            return tccId(_id, electronicsMap);
          else
            break;
        case kEEp:
          if (_btype == kDCC)
            return dccId(_id, electronicsMap) - 45;
          else if (_btype == kTCC)
            return tccId(_id, electronicsMap) - 72;
          else
            break;
        case kEE:
          if (_btype == kDCC) {
            int bin(dccId(_id, electronicsMap));
            if (bin >= 46)
              bin -= 36;
            return bin;
          } else if (_btype == kTCC) {
            int bin(tccId(_id, electronicsMap));
            if (bin >= 72)
              bin -= 36;
            return bin;
          } else
            break;
        case kSMMEM:
        case kEBSMMEM:
        case kEESMMEM:
          if (_btype == kCrystal)
            return EcalPnDiodeDetId(_id).iPnId();
          else
            break;
        default:
          break;
      }

      return 0;
    }

    int findBin1D(EcalElectronicsMapping const *electronicsMap,
                  ObjectType _otype,
                  BinningType _btype,
                  const EcalElectronicsId &_id) {
      switch (_otype) {
        case kSM:
        case kEBSM:
        case kEESM:
          if (_btype == kSuperCrystal)
            return towerId(_id);
          else if (_btype == kTriggerTower) {
            unsigned tccid(tccId(_id, electronicsMap));
            if (tccid <= 36 || tccid >= 73) {  // EE
              unsigned bin(ttId(_id, electronicsMap));
              bool outer((tccid >= 19 && tccid <= 36) || (tccid >= 73 && tccid <= 90));
              // For the following, the constants nTTInner and nTTOuter are defined in
              // EcalDQMCommonUtils.h.
              if (outer)
                bin += 2 * nTTInner;  // For outer TCCs, sets bin number to increment
              // by twice the number of TTs in inner TCCs, because numbering of bins
              // is in the order (inner1, inner2, outer1, outer2).
              // ("inner"" := closer to the beam)
              bin += (tccid % 2) * (outer ? nTTOuter : nTTInner);  // Yields x-axis bin number
              // in the format above; TTs in even-numbered TCCs are filled in inner1
              // or outer1, and those in odd-numbered TCC are filled in inner2 and
              // outer2.
              return bin;
            } else
              return ttId(_id, electronicsMap);
          } else
            break;
        case kEcal:
          if (_btype == kDCC)
            return dccId(_id);
          else if (_btype == kTCC)
            return tccId(_id, electronicsMap);
          else
            break;
        case kEB:
          if (_btype == kDCC)
            return dccId(_id) - 9;
          else if (_btype == kTCC)
            return tccId(_id, electronicsMap) - 36;
          else
            break;
        case kEEm:
          if (_btype == kDCC)
            return dccId(_id);
          else if (_btype == kTCC)
            return tccId(_id, electronicsMap);
          else
            break;
        case kEEp:
          if (_btype == kDCC)
            return dccId(_id) - 45;
          else if (_btype == kTCC)
            return tccId(_id, electronicsMap) - 72;
          else
            break;
        case kEE:
          if (_btype == kDCC) {
            int bin(dccId(_id));
            if (bin >= 46)
              bin -= 36;
            return bin;
          } else if (_btype == kTCC) {
            int bin(tccId(_id, electronicsMap));
            if (bin >= 72)
              bin -= 36;
            return bin;
          } else
            break;
        default:
          break;
      }

      return 0;
    }

    int findBin1D(EcalElectronicsMapping const *electronicsMap, ObjectType _otype, BinningType _btype, int _dcctccid) {
      if (_otype == kEcal && _btype == kDCC)
        return _dcctccid;
      else if (_otype == kEcal && _btype == kTCC)
        return _dcctccid;
      if (_otype == kEB && _btype == kDCC)
        return _dcctccid - 9;
      else if (_otype == kEB && _btype == kTCC)
        return _dcctccid - 36;
      else if (_otype == kEEm && _btype == kDCC)
        return _dcctccid;
      else if (_otype == kEEm && _btype == kTCC)
        return _dcctccid;
      else if (_otype == kEEp && _btype == kDCC)
        return _dcctccid - 45;
      else if (_otype == kEEp && _btype == kTCC)
        return _dcctccid - 72;
      else if (_otype == kEE && _btype == kDCC)
        return _dcctccid <= 9 ? _dcctccid : _dcctccid - 36;
      else if (_otype == kEE && _btype == kTCC)
        return _dcctccid <= 36 ? _dcctccid : _dcctccid - 36;

      return 0;
    }

    int findBin2D(EcalElectronicsMapping const *electronicsMap,
                  ObjectType _otype,
                  BinningType _btype,
                  const DetId &_id) {
      if (_otype >= nObjType || _btype >= unsigned(nPresetBinnings))
        return 0;

      switch (_btype) {
        case kCrystal:
          return findBinCrystal_(electronicsMap, _otype, _id);
          break;
        case kTriggerTower:
          return findBinTriggerTower_(electronicsMap, _otype, _id);
          break;
        case kSuperCrystal:
          return findBinSuperCrystal_(electronicsMap, _otype, _id);
          break;
        case kPseudoStrip:
          return findBinPseudoStrip_(electronicsMap, _otype, _id);
          break;
        case kRCT:
          return findBinRCT_(_otype, _id);
          break;
        default:
          return 0;
      }
    }

    int findBin2D(EcalElectronicsMapping const *electronicsMap,
                  ObjectType _otype,
                  BinningType _btype,
                  const EcalElectronicsId &_id) {
      if (_otype >= nObjType || _btype >= unsigned(nPresetBinnings))
        return 0;

      switch (_btype) {
        case kCrystal:
          return findBinCrystal_(electronicsMap, _otype, _id);
          break;
        case kSuperCrystal:
          return findBinSuperCrystal_(electronicsMap, _otype, _id);
          break;
        default:
          return 0;
      }
    }

    int findBin2D(EcalElectronicsMapping const *electronicsMap, ObjectType _otype, BinningType _btype, int _dccid) {
      if (_otype != kEcal || _btype != kDCC)
        return 0;

      int nbinsX(9);
      unsigned iDCC(_dccid - 1);
      int xbin(0);
      if (iDCC <= kEEmHigh || iDCC >= kEEpLow)
        xbin = (iDCC + 6) % nbinsX + 1;
      else
        xbin = iDCC % nbinsX + 1;
      int ybin(6 - iDCC / nbinsX);

      return (nbinsX + 2) * ybin + xbin;
    }

    unsigned findPlotIndex(EcalElectronicsMapping const *electronicsMap, ObjectType _otype, const DetId &_id) {
      if (getNObjects(_otype) == 1)
        return 0;

      switch (_otype) {
        case kEcal3P:
          if (_id.subdetId() == EcalBarrel)
            return 1;
          else if (_id.subdetId() == EcalEndcap && zside(_id) > 0)
            return 2;
          else if (_id.subdetId() == EcalTriggerTower) {
            if (!isEndcapTTId(_id))
              return 1;
            else {
              if (zside(_id) > 0)
                return 2;
              else
                return 0;
            }
          } else
            return 0;

        case kEcal2P:
          if (_id.subdetId() == EcalBarrel)
            return 1;
          else if (_id.subdetId() == EcalTriggerTower && !isEndcapTTId(_id))
            return 1;
          else
            return 0;

        case kEE2P:
          if (zside(_id) > 0)
            return 1;
          else
            return 0;

        case kMEM2P:
          if (_id.subdetId() == EcalLaserPnDiode) {
            unsigned iDCC(dccId(_id, electronicsMap) - 1);
            if (iDCC >= kEBmLow && iDCC <= kEBpHigh)
              return 1;
            else
              return 0;
          } else
            return -1;

        default:
          return findPlotIndex(electronicsMap, _otype, dccId(_id, electronicsMap));
      }
    }

    unsigned findPlotIndex(EcalElectronicsMapping const *electronicsMap,
                           ObjectType _otype,
                           const EcalElectronicsId &_id) {
      if (getNObjects(_otype) == 1)
        return 0;

      return findPlotIndex(electronicsMap, _otype, _id.dccId());
    }

    unsigned findPlotIndex(EcalElectronicsMapping const *electronicsMap,
                           ObjectType _otype,
                           int _dcctccid,
                           BinningType _btype /* = kDCC*/) {
      if (getNObjects(_otype) == 1)
        return 0;

      int iSM(_dcctccid - 1);

      switch (_otype) {
        case kSM:
          if (_btype == kPseudoStrip) {
            iSM = iSM <= kEEmTCCHigh  ? (iSM + 1) % 18 / 2
                  : iSM >= kEEpTCCLow ? (iSM + 1 - 72) % 18 / 2 + 45
                                      : (iSM + 1) - kEEmTCCHigh;
            return iSM;
          } else
            return iSM;

        case kEBSM:
          return iSM - 9;

        case kEESM:
          if (iSM <= kEEmHigh)
            return iSM;
          else
            return iSM - nEBDCC;

        case kSMMEM:
          return memDCCIndex(_dcctccid);

        case kEBSMMEM:
          return memDCCIndex(_dcctccid) - nEEDCCMEM / 2;

        case kEESMMEM:
          if (iSM <= kEEmHigh)
            return memDCCIndex(_dcctccid);
          else
            return memDCCIndex(_dcctccid) - nEBDCC;

        case kEcal2P:
          if (_btype == kDCC) {
            if (iSM <= kEEmHigh || iSM >= kEEpLow)
              return 0;
            else
              return 1;
          } else if (_btype == kTCC) {
            if (iSM <= kEEmTCCHigh || iSM >= kEEpTCCLow)
              return 0;
            else
              return 1;
          } else {
            if (iSM == EcalBarrel - 1)
              return 1;
            else
              return 0;
          }

        case kEcal3P:
          if (_btype == kDCC) {
            if (iSM <= kEEmHigh)
              return 0;
            else if (iSM <= kEBpHigh)
              return 1;
            else
              return 2;
          } else if (_btype == kTCC) {
            if (iSM <= kEEmTCCHigh)
              return 0;
            else if (iSM <= kEBTCCHigh)
              return 1;
            else
              return 2;
          } else {
            if (iSM == -EcalEndcap - 1)
              return 0;
            else if (iSM == EcalBarrel - 1)
              return 1;
            else
              return 2;
          }

        case kEE2P:
          if (_btype == kDCC) {
            if (iSM >= kEEpLow)
              return 1;
            else
              return 0;
          } else {
            if (iSM >= kEEpTCCLow)
              return 1;
            else
              return 0;
          }

        case kMEM2P:
          if (_btype == kDCC) {
            if (iSM <= kEEmHigh || iSM >= kEEpLow)
              return 0;
            else
              return 1;
          } else if (_btype == kTCC)
            return -1;
          else {
            if (iSM == kEB)
              return 1;
            else
              return 0;
          }
        default:
          return -1;
      }
    }

    ObjectType getObject(ObjectType _otype, unsigned _iObj) {
      switch (_otype) {
        case kEcal3P:
          switch (_iObj) {
            case 0:
              return kEEm;
            case 1:
              return kEB;
            case 2:
              return kEEp;
            default:
              return nObjType;
          }
        case kEcal2P:
          switch (_iObj) {
            case 0:
              return kEE;
            case 1:
              return kEB;
            default:
              return nObjType;
          }
        case kEE2P:
          switch (_iObj) {
            case 0:
              return kEEm;
            case 1:
              return kEEp;
            default:
              return nObjType;
          }
        case kMEM2P:
          switch (_iObj) {
            case 0:
              return kEEMEM;
            case 1:
              return kEBMEM;
            default:
              return nObjType;
          }
        default:
          return _otype;
      }
    }

    unsigned getNObjects(ObjectType _otype) {
      switch (_otype) {
        case kSM:
          return nDCC;
        case kEBSM:
          return nEBDCC;
        case kEESM:
          return nEEDCC;
        case kSMMEM:
          return nDCCMEM;
        case kEBSMMEM:
          return nEBDCC;
        case kEESMMEM:
          return nEEDCCMEM;
        case kEcal2P:
          return 2;
        case kEcal3P:
          return 3;
        case kEE2P:
          return 2;
        case kMEM2P:
          return 2;
        default:
          return 1;
      }
    }

    bool isValidIdBin(
        EcalElectronicsMapping const *electronicsMap, ObjectType _otype, BinningType _btype, unsigned _iME, int _bin) {
      if (_otype == kEEm || _otype == kEEp) {
        if (_btype == kCrystal || _btype == kTriggerTower)
          return EEDetId::validDetId(_bin % 102, _bin / 102, 1);
        else if (_btype == kSuperCrystal)
          return EcalScDetId::validDetId(_bin % 22, _bin / 22, 1);
      } else if (_otype == kEE) {
        if (_btype == kCrystal || _btype == kTriggerTower) {
          int ix(_bin % 202);
          if (ix > 100)
            ix = (ix - 100) % 101;
          return EEDetId::validDetId(ix, _bin / 202, 1);
        } else if (_btype == kSuperCrystal) {
          int ix(_bin % 42);
          if (ix > 20)
            ix = (ix - 20) % 21;
          return EcalScDetId::validDetId(ix, _bin / 42, 1);
        }
      } else if (_otype == kSM || _otype == kEESM) {
        unsigned iSM(_iME);
        if (_otype == kEESM && iSM > kEEmHigh)
          iSM += nEBDCC;

        if (iSM >= kEBmLow && iSM <= kEBpHigh)
          return true;

        if (_btype == kCrystal || _btype == kTriggerTower) {
          int nX(nEESMX);
          if (iSM == kEEm02 || iSM == kEEm08 || iSM == kEEp02 || iSM == kEEp08)
            nX = nEESMXExt;
          if (iSM == kEEm01 || iSM == kEEm05 || iSM == kEEm09 || iSM == kEEp01 || iSM == kEEp05 || iSM == kEEp09)
            nX = nEESMXRed;
          int ix(_bin % (nX + 2) + xlow_(iSM));
          int iy(_bin / (nX + 2) + ylow_(iSM));
          int z(iSM <= kEEmHigh ? -1 : 1);
          return EEDetId::validDetId(ix, iy, 1) && iSM == dccId(EEDetId(ix, iy, z), electronicsMap) - 1;
        } else if (_btype == kSuperCrystal) {
          int nX(nEESMX / 5);
          if (iSM == kEEm02 || iSM == kEEm08 || iSM == kEEp02 || iSM == kEEp08)
            nX = nEESMXExt / 5;
          if (iSM == kEEm01 || iSM == kEEm05 || iSM == kEEm09 || iSM == kEEp01 || iSM == kEEp05 || iSM == kEEp09)
            nX = nEESMXRed / 5;
          int ix(_bin % (nX + 2) + xlow_(iSM) / 5);
          int iy(_bin / (nX + 2) + ylow_(iSM) / 5);
          int z(iSM <= kEEmHigh ? -1 : 1);
          return EcalScDetId::validDetId(ix, iy, z) && iSM == dccId(EcalScDetId(ix, iy, z), electronicsMap) - 1;
        }
      }

      return true;
    }

    std::string channelName(const EcalElectronicsMapping *electronicsMap,
                            uint32_t _rawId,
                            BinningType _btype /* = kDCC*/) {
      // assume the following IDs for respective binning types:
      // Crystal: EcalElectronicsId
      // TriggerTower: EcalTriggerElectronicsId (pstrip and channel ignored)
      // SuperCrystal: EcalElectronicsId (strip and crystal ignored)
      // TCC: TCC ID
      // DCC: DCC ID

      std::stringstream ss;

      switch (_btype) {
        case kCrystal: {
          // EB-03 DCC 12 CCU 12 strip 3 xtal 1 (EB ieta -13 iphi 60) (TCC 39 TT 12
          // pstrip 3 chan 1)
          EcalElectronicsId eid(_rawId);
          if (eid.towerId() >= 69)
            ss << smName(eid.dccId()) << " DCC " << eid.dccId() << " CCU " << eid.towerId() << " PN " << eid.xtalId();
          else {
            ss << smName(eid.dccId()) << " DCC " << eid.dccId() << " CCU " << eid.towerId() << " strip "
               << eid.stripId() << " xtal " << eid.xtalId();

            if (eid.dccId() >= kEBmLow + 1 && eid.dccId() <= kEBpHigh + 1) {
              EBDetId ebid(electronicsMap->getDetId(eid));
              ss << " (EB ieta " << std::showpos << ebid.ieta() << std::noshowpos << " iphi " << ebid.iphi() << ")";
            } else {
              EEDetId eeid(electronicsMap->getDetId(eid));
              ss << " (EE ix " << eeid.ix() << " iy " << eeid.iy() << ")";
            }
            EcalTriggerElectronicsId teid(electronicsMap->getTriggerElectronicsId(eid));
            ss << " (TCC " << teid.tccId() << " TT " << teid.ttId() << " pstrip " << teid.pseudoStripId() << " chan "
               << teid.channelId() << ")";
            break;
          }
        }
          [[fallthrough]];
        case kTriggerTower: {
          // EB-03 DCC 12 TCC 18 TT 3
          EcalTriggerElectronicsId teid(_rawId);
          EcalElectronicsId eid(electronicsMap->getElectronicsId(teid));
          ss << smName(eid.dccId()) << " DCC " << eid.dccId() << " TCC " << teid.tccId() << " TT " << teid.ttId();
          break;
        }
        case kSuperCrystal: {
          // EB-03 DCC 12 CCU 18 (EBTT ieta -13 iphi 60)
          EcalElectronicsId eid(_rawId);
          ss << smName(eid.dccId()) << " DCC " << eid.dccId() << " CCU " << eid.towerId();
          if (eid.dccId() >= kEBmLow + 1 && eid.dccId() <= kEBpHigh + 1) {
            EcalTrigTowerDetId ttid(EBDetId(electronicsMap->getDetId(eid)).tower());
            ss << " (EBTT ieta " << std::showpos << ttid.ieta() << std::noshowpos << " iphi " << ttid.iphi() << ")";
          } else {
            EcalScDetId scid(EEDetId(electronicsMap->getDetId(eid)).sc());
            ss << " (EESC ix " << scid.ix() << " iy " << scid.iy() << ")";
          }
          break;
        }
        case kTCC: {
          // EB-03 TCC 12
          int tccid(_rawId - nDCC);
          int dccid(electronicsMap->DCCid(electronicsMap->getTrigTowerDetId(tccid, 1)));
          ss << smName(dccid) << " TCC " << (_rawId - nDCC);
          break;
        }
        case kDCC:
          ss << smName(_rawId);
          break;
        default:
          break;
      }

      return ss.str();
    }

    uint32_t idFromName(std::string const &_name) {
      TString name(_name);
      TPRegexp re(
          "(EB|EE)([+-][0-9][0-9])(?: TCC ([0-9]+)| DCC ([0-9]+) (CCU|TCC) "
          "([0-9]+)(?: (TT|strip|PN) ([0-9]+)(?: xtal ([0-9]+)|)|)|)");
      //            1      2                       3             4        5 6 7 8 9
      uint32_t rawId(0);

      TObjArray *matches(re.MatchS(name));
      matches->SetOwner(true);
      if (matches->GetEntries() == 0)
        return 0;
      else if (matches->GetEntries() == 3) {
        TString subdet(static_cast<TObjString *>(matches->At(1))->GetString());
        if (subdet == "EB") {
          int dccid(static_cast<TObjString *>(matches->At(2))->GetString().Atoi());
          unsigned offset(0);
          if (dccid < 0) {
            dccid *= -1;
            offset = kEEmLow;
          } else
            offset = kEEpLow;
          rawId = (dccid + 2) % 9 + 1 + offset;
        } else {
          int dccid(static_cast<TObjString *>(matches->At(2))->GetString().Atoi());
          if (dccid < 0)
            dccid *= -1;
          else
            dccid += 18;
          rawId = kEBmLow + dccid;
        }
      } else if (matches->GetEntries() == 4)
        rawId = static_cast<TObjString *>(matches->At(3))->GetString().Atoi() + nDCC;
      else {
        TString subtype(static_cast<TObjString *>(matches->At(5))->GetString());
        if (subtype == "TCC") {
          int tccid(static_cast<TObjString *>(matches->At(6))->GetString().Atoi());
          int ttid(static_cast<TObjString *>(matches->At(8))->GetString().Atoi());
          rawId = EcalTriggerElectronicsId(tccid, ttid, 1, 1).rawId();
        } else {
          int dccid(static_cast<TObjString *>(matches->At(4))->GetString().Atoi());
          int towerid(static_cast<TObjString *>(matches->At(6))->GetString().Atoi());
          if (matches->GetEntries() == 7)
            rawId = EcalElectronicsId(dccid, towerid, 1, 1).rawId();
          else {
            TString chType(static_cast<TObjString *>(matches->At(7))->GetString());
            int stripOrPNid(static_cast<TObjString *>(matches->At(8))->GetString().Atoi());
            if (chType == "PN")
              rawId = EcalElectronicsId(dccid, towerid, 1, stripOrPNid).rawId();
            else if (chType == "strip") {
              int xtalid(static_cast<TObjString *>(matches->At(9))->GetString().Atoi());
              rawId = EcalElectronicsId(dccid, towerid, stripOrPNid, xtalid).rawId();
            }
            // case "TT" is already taken care of
          }
        }
      }

      delete matches;

      return rawId;
    }

    uint32_t idFromBin(ObjectType _otype, BinningType _btype, unsigned _iME, int _bin) {
      if (_otype == kEB) {
        if (_btype == kCrystal) {
          int ieta(_bin / 362 - 86);
          if (ieta >= 0)
            ++ieta;
          return EBDetId(ieta, _bin % 362);
        } else if (_btype == kTriggerTower || _btype == kSuperCrystal) {
          int ieta(_bin / 74 - 17);
          int z(1);
          if (ieta <= 0) {
            z = -1;
            ieta = -ieta + 1;
          }
          return EcalTrigTowerDetId(z, EcalBarrel, ieta, (_bin % 74 + 69) % 72 + 1);
        }
      } else if (_otype == kEEm || _otype == kEEp) {
        int z(_otype == kEEm ? -1 : 1);
        if (_btype == kCrystal || _btype == kTriggerTower)
          return EEDetId(_bin % 102, _bin / 102, z).rawId();
        else if (_btype == kSuperCrystal)
          return EcalScDetId(_bin % 22, _bin / 22, z).rawId();
      } else if (_otype == kEE) {
        if (_btype == kCrystal || _btype == kTriggerTower) {
          int ix(_bin % 202);
          int z(ix > 100 ? 1 : -1);
          if (z > 0)
            ix = (ix - 100) % 101;
          return EEDetId(ix, _bin / 202, z).rawId();
        } else if (_btype == kSuperCrystal) {
          int ix(_bin % 42);
          int z(ix > 20 ? 1 : -1);
          if (z > 0)
            ix = (ix - 20) % 21;
          return EcalScDetId(ix, _bin / 42, z).rawId();
        }
      } else if (_otype == kSM || _otype == kEBSM || _otype == kEESM) {
        unsigned iSM(_iME);
        if (_otype == kEBSM)
          iSM += 9;
        else if (_otype == kEESM && iSM > kEEmHigh)
          iSM += nEBDCC;

        int z(iSM <= kEBmHigh ? -1 : 1);

        if ((iSM >= kEBmLow && iSM <= kEBmHigh) || (iSM >= kEBpLow && iSM <= kEBpHigh)) {
          if (_btype == kCrystal) {
            int iphi(((iSM - 9) % 18) * 20 + (z < 0 ? _bin / 87 : 21 - _bin / 87));
            int ieta((_bin % 87) * z);
            return EBDetId(ieta, iphi).rawId();
          } else if (_btype == kTriggerTower || _btype == kSuperCrystal) {
            int iphi((((iSM - 9) % 18) * 4 + (z < 0 ? _bin / 19 : 5 - _bin / 19) + 69) % 72 + 1);
            int ieta(_bin % 19);
            return EcalTrigTowerDetId(z, EcalBarrel, ieta, iphi).rawId();
          }
        } else {
          if (_btype == kCrystal || _btype == kTriggerTower) {
            int nX(nEESMX);
            if (iSM == kEEm02 || iSM == kEEm08 || iSM == kEEp02 || iSM == kEEp08)
              nX = nEESMXExt;
            if (iSM == kEEm01 || iSM == kEEm05 || iSM == kEEm09 || iSM == kEEp01 || iSM == kEEp05 || iSM == kEEp09)
              nX = nEESMXRed;
            return EEDetId(_bin % (nX + 2) + xlow_(iSM), _bin / (nX + 2) + ylow_(iSM), z).rawId();
          } else if (_btype == kSuperCrystal) {
            int nX(nEESMX / 5);
            if (iSM == kEEm02 || iSM == kEEm08 || iSM == kEEp02 || iSM == kEEp08)
              nX = nEESMXExt / 5;
            if (iSM == kEEm01 || iSM == kEEm05 || iSM == kEEm09 || iSM == kEEp01 || iSM == kEEp05 || iSM == kEEp09)
              nX = nEESMXRed / 5;
            return EcalScDetId(_bin % (nX + 2) + xlow_(iSM) / 5, _bin / (nX + 2) + ylow_(iSM) / 5, z).rawId();
          }
        }
      }

      return 0;
    }

    AxisSpecs formAxis(edm::ParameterSet const &_axisParams) {
      AxisSpecs axis;

      if (_axisParams.existsAs<std::vector<double>>("edges", false)) {
        std::vector<double> const &vEdges(_axisParams.getUntrackedParameter<std::vector<double>>("edges"));
        axis.nbins = vEdges.size() - 1;
        axis.edges.assign(vEdges.begin(), vEdges.end());
      } else {
        axis.nbins = _axisParams.getUntrackedParameter<int>("nbins");
        axis.low = _axisParams.getUntrackedParameter<double>("low");
        bool highSet(_axisParams.existsAs<double>("high", false));
        bool perBinSet(_axisParams.existsAs<double>("unitsPerBin", false));
        if (highSet) {
          if (perBinSet)
            edm::LogWarning("EcalDQM") << "Maximum and bin width both set in an axis; using the former";
          axis.high = _axisParams.getUntrackedParameter<double>("high");
        } else if (perBinSet)
          axis.high = axis.low + _axisParams.getUntrackedParameter<double>("unitsPerBin") * axis.nbins;
        else
          axis.high = 0.;
      }

      if (_axisParams.existsAs<std::vector<std::string>>("labels", false)) {
        std::vector<std::string> const &labels(_axisParams.getUntrackedParameter<std::vector<std::string>>("labels"));
        if (int(labels.size()) == axis.nbins) {
          axis.labels = labels;
        }
      }

      axis.title = _axisParams.getUntrackedParameter<std::string>("title");

      return axis;
    }

    void fillAxisDescriptions(edm::ParameterSetDescription &_desc) {
      _desc.addUntracked<std::string>("title", "");
      _desc.addUntracked<int>("nbins", 0);
      _desc.addUntracked<double>("low", 0.);
      _desc.addOptionalNode(edm::ParameterDescription<double>("high", 0., false) ^
                                edm::ParameterDescription<double>("unitsPerBin", 0., false),
                            false);
      _desc.addOptionalUntracked<std::vector<double>>("edges");
      _desc.addOptionalUntracked<std::vector<std::string>>("labels");
    }

    ObjectType translateObjectType(std::string const &_otypeName) {
      if (_otypeName == "EB")
        return kEB;
      else if (_otypeName == "EE")
        return kEE;
      else if (_otypeName == "EEm")
        return kEEm;
      else if (_otypeName == "EEp")
        return kEEp;
      else if (_otypeName == "SM")
        return kSM;
      else if (_otypeName == "EBSM")
        return kEBSM;
      else if (_otypeName == "EESM")
        return kEESM;
      else if (_otypeName == "SMMEM")
        return kSMMEM;
      else if (_otypeName == "EBSMMEM")
        return kEBSMMEM;
      else if (_otypeName == "EESMMEM")
        return kEESMMEM;
      else if (_otypeName == "Ecal")
        return kEcal;
      else if (_otypeName == "MEM")
        return kMEM;
      else if (_otypeName == "EBMEM")
        return kEBMEM;
      else if (_otypeName == "EEMEM")
        return kEEMEM;
      else if (_otypeName == "Ecal2P")
        return kEcal2P;
      else if (_otypeName == "Ecal3P")
        return kEcal3P;
      else if (_otypeName == "EE2P")
        return kEE2P;
      else if (_otypeName == "MEM2P")
        return kMEM2P;
      else if (_otypeName == "Channel")
        return kChannel;
      else if (_otypeName == "None")
        return nObjType;

      throw cms::Exception("InvalidConfiguration") << "No object type " << _otypeName << " defined";
    }

    BinningType translateBinningType(std::string const &_btypeName) {
      if (_btypeName == "Crystal")
        return kCrystal;
      else if (_btypeName == "TriggerTower")
        return kTriggerTower;
      else if (_btypeName == "SuperCrystal")
        return kSuperCrystal;
      else if (_btypeName == "PseudoStrip")
        return kPseudoStrip;
      else if (_btypeName == "TCC")
        return kTCC;
      else if (_btypeName == "DCC")
        return kDCC;
      else if (_btypeName == "ProjEta")
        return kProjEta;
      else if (_btypeName == "ProjPhi")
        return kProjPhi;
      else if (_btypeName == "RCT")
        return kRCT;
      else if (_btypeName == "User")
        return kUser;
      else if (_btypeName == "Report")
        return kReport;
      else if (_btypeName == "Trend")
        return kTrend;

      throw cms::Exception("InvalidConfiguration") << "No binning type " << _btypeName << " defined";
    }

    dqm::reco::MonitorElement::Kind translateKind(std::string const &_kindName) {
      if (_kindName == "REAL")
        return dqm::reco::MonitorElement::Kind::REAL;
      else if (_kindName == "TH1F")
        return dqm::reco::MonitorElement::Kind::TH1F;
      else if (_kindName == "TProfile")
        return dqm::reco::MonitorElement::Kind::TPROFILE;
      else if (_kindName == "TH2F")
        return dqm::reco::MonitorElement::Kind::TH2F;
      else if (_kindName == "TProfile2D")
        return dqm::reco::MonitorElement::Kind::TPROFILE2D;
      else
        return dqm::reco::MonitorElement::Kind::INVALID;
    }
  }  // namespace binning
}  // namespace ecaldqm
