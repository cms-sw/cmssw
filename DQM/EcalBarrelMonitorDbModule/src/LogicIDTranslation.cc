#include "../interface/LogicIDTranslation.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace ecaldqm
{
  EcalLogicID
  subdetID(EcalSubdetector _subdet)
  {
    switch(_subdet){
    case EcalBarrel:
      return EcalLogicID("EB", 1000000000UL);
    case EcalEndcap:
      return EcalLogicID("EE", 2000000000UL);
    default:
      throw cms::Exception("UndefinedLogicID");
    }
  }

  EcalLogicID
  crystalID(DetId const& _id)
  {
    unsigned iDCC(dccId(_id) - 1);
    if(iDCC <= kEEmHigh || iDCC >= kEEpLow){
      EEDetId eeid(_id);
      return EcalLogicID("EE_crystal_number",
                         2010000000UL + 1000000 * (eeid.positiveZ() ? 2 : 0) + 1000 * eeid.ix() + eeid.iy(),
                         eeid.zside(), eeid.ix(), eeid.iy());
    }
    else{
      int ism(iDCC <= kEBmHigh ? 19 + iDCC - kEBmLow : 1 + iDCC - kEBpLow);
      EBDetId ebid(_id);
      return EcalLogicID("EB_crystal_number",
                         1011000000UL + 10000 * ism + ebid.ic(),
                         ism, ebid.ic());
    }
  }

  EcalLogicID
  towerID(EcalElectronicsId const& _id)
  {
    unsigned iDCC(_id.dccId() - 1);
    if(iDCC <= kEEmHigh || iDCC >= kEEpLow){
      return EcalLogicID("EE_readout_tower",
                         2110000000UL + 100 * (601 + iDCC) + _id.towerId(),
                         601 + iDCC, _id.towerId());
    }
    else{
      int ism(iDCC <= kEBmHigh ? 19 + iDCC - kEBmLow : 1 + iDCC - kEBpLow);
      return EcalLogicID("EB_trigger_tower",
                         1021000000UL + 10000 * ism + _id.towerId(),
                         ism, _id.towerId());
    }
  }

  EcalLogicID
  memChannelID(EcalPnDiodeDetId const& _id)
  {
    // using the PN ID degenerates the logic ID - 50 time samples are actually split into 5 channels each
    unsigned iDCC(_id.iDCCId() - 1);
    int memId((_id.iPnId() - 1) % 5 + ((_id.iPnId() - 1) / 5) * 25 + 1);
    if(iDCC <= kEEmHigh || iDCC >= kEEpLow){
      return EcalLogicID("EE_mem_channel",
                         100 * (601 + iDCC) + memId,
                         601 + iDCC, memId);
    }
    else{
      int ism(iDCC <= kEBmHigh ? 19 + iDCC - kEBmLow : 1 + iDCC - kEBpLow);
      return EcalLogicID("EB_mem_channel",
                         1191000000UL + 10000 * ism + memId,
                         ism, memId);
    }
  }

  EcalLogicID
  memTowerID(EcalElectronicsId const& _id)
  {
    unsigned iDCC(_id.dccId() - 1);
    if(iDCC <= kEEmHigh || iDCC >= kEEpLow){
      return EcalLogicID("EE_mem_TT",
                         100 * (601 + iDCC) + _id.towerId(),
                         601 + iDCC, _id.towerId());
    }
    else{
      int ism(iDCC <= kEBmHigh ? 19 + iDCC - kEBmLow : 1 + iDCC - kEBpLow);
      return EcalLogicID("EB_mem_TT",
                         1181000000UL + 10000 * ism + _id.towerId(),
                         ism, _id.towerId());
    }
  }

  EcalLogicID
  lmPNID(EcalPnDiodeDetId const& _id)
  {
    unsigned iDCC(_id.iDCCId() - 1);
    int pnid(_id.iPnId());
    if(iDCC <= kEEmHigh || iDCC >= kEEpLow){
      return EcalLogicID("EE_LM_PN",
                         100 * (601 + iDCC) + pnid,
                         601 + iDCC, pnid);
    }
    else{
      int ism(iDCC <= kEBmHigh ? 19 + iDCC - kEBmLow : 1 + iDCC - kEBpLow);
      return EcalLogicID("EB_LM_PN",
                         1131000000UL + 10000 * ism + pnid,
                         ism, pnid);
    }
  }

  DetId
  toDetId(EcalLogicID const& _id)
  {
    return DetId();
  }
}

