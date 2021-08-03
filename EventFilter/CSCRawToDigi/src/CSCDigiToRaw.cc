/** \file
 *
 *  \author A. Tumanov - Rice
 *  But long, long ago...
 */

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCConstants.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDigiToRaw.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "boost/dynamic_bitset.hpp"
#include "EventFilter/CSCRawToDigi/interface/bitset_append.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "FWCore/Utilities/interface/CRC16.h"
#include "CondFormats/CSCObjects/interface/CSCChamberMap.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <algorithm>

using namespace edm;
using namespace std;

namespace cscd2r {
  /// takes layer ID, converts to chamber ID, switching ME1A to ME11
  CSCDetId chamberID(const CSCDetId& cscDetId) {
    CSCDetId chamberId = cscDetId.chamberId();
    if (chamberId.ring() == 4) {
      chamberId = CSCDetId(chamberId.endcap(), chamberId.station(), 1, chamberId.chamber(), 0);
    }
    return chamberId;
  }

  template <typename LCTCollection>
  bool accept(
      const CSCDetId& cscId, const LCTCollection& lcts, int bxMin, int bxMax, int nominalBX, bool me1abCheck = false) {
    if (bxMin == -999)
      return true;
    CSCDetId chamberId = chamberID(cscId);
    typename LCTCollection::Range lctRange = lcts.get(chamberId);
    bool result = false;
    for (typename LCTCollection::const_iterator lctItr = lctRange.first; lctItr != lctRange.second; ++lctItr) {
      int bx = lctItr->getBX() - nominalBX;
      if (bx >= bxMin && bx <= bxMax) {
        result = true;
        break;
      }
    }

    bool me1 = cscId.station() == 1 && cscId.ring() == 1;
    //this is another "creative" recovery of smart ME1A-ME1B TMB logic cases:
    //wire selective readout requires at least one (A)LCT in the full chamber
    if (me1 && result == false && me1abCheck) {
      CSCDetId me1aId = CSCDetId(chamberId.endcap(), chamberId.station(), 4, chamberId.chamber(), 0);
      lctRange = lcts.get(me1aId);
      for (typename LCTCollection::const_iterator lctItr = lctRange.first; lctItr != lctRange.second; ++lctItr) {
        int bx = lctItr->getBX() - nominalBX;
        if (bx >= bxMin && bx <= bxMax) {
          result = true;
          break;
        }
      }
    }

    return result;
  }

  // need to specialize for pretriggers, since they don't have a getBX()
  template <>
  bool accept(const CSCDetId& cscId,
              const CSCCLCTPreTriggerCollection& lcts,
              int bxMin,
              int bxMax,
              int nominalBX,
              bool me1abCheck) {
    if (bxMin == -999)
      return true;
    CSCDetId chamberId = chamberID(cscId);
    CSCCLCTPreTriggerCollection::Range lctRange = lcts.get(chamberId);
    bool result = false;
    for (CSCCLCTPreTriggerCollection::const_iterator lctItr = lctRange.first; lctItr != lctRange.second; ++lctItr) {
      int bx = *lctItr - nominalBX;
      if (bx >= bxMin && bx <= bxMax) {
        result = true;
        break;
      }
    }
    bool me1a = cscId.station() == 1 && cscId.ring() == 4;
    if (me1a && result == false && me1abCheck) {
      //check pretriggers in me1a as well; relevant for TMB emulator writing to separate detIds
      lctRange = lcts.get(cscId);
      for (CSCCLCTPreTriggerCollection::const_iterator lctItr = lctRange.first; lctItr != lctRange.second; ++lctItr) {
        int bx = *lctItr - nominalBX;
        if (bx >= bxMin && bx <= bxMax) {
          result = true;
          break;
        }
      }
    }
    return result;
  }

}  // namespace cscd2r

CSCDigiToRaw::CSCDigiToRaw(const edm::ParameterSet& pset)
    : alctWindowMin_(pset.getParameter<int>("alctWindowMin")),
      alctWindowMax_(pset.getParameter<int>("alctWindowMax")),
      clctWindowMin_(pset.getParameter<int>("clctWindowMin")),
      clctWindowMax_(pset.getParameter<int>("clctWindowMax")),
      preTriggerWindowMin_(pset.getParameter<int>("preTriggerWindowMin")),
      preTriggerWindowMax_(pset.getParameter<int>("preTriggerWindowMax")),
      // pre-LS1 - '2005'. post-LS1 - '2013'
      formatVersion_(pset.getParameter<unsigned>("formatVersion")),
      // don't check for consistency with trig primitives
      // overrides usePreTriggers
      packEverything_(pset.getParameter<bool>("packEverything")),
      usePreTriggers_(pset.getParameter<bool>("usePreTriggers")) {}

CSCEventData& CSCDigiToRaw::findEventData(const CSCDetId& cscDetId, FindEventDataInfo& info) const {
  CSCDetId chamberId = cscd2r::chamberID(cscDetId);
  // find the entry into the map
  map<CSCDetId, CSCEventData>::iterator chamberMapItr = info.theChamberDataMap.find(chamberId);
  if (chamberMapItr == info.theChamberDataMap.end()) {
    // make an entry, telling it the correct chamberType
    int chamberType = chamberId.iChamberType();
    chamberMapItr = info.theChamberDataMap
                        .insert(pair<CSCDetId, CSCEventData>(chamberId, CSCEventData(chamberType, info.formatVersion_)))
                        .first;
  }
  CSCEventData& cscData = chamberMapItr->second;
  cscData.dmbHeader()->setCrateAddress(info.theElectronicsMap->crate(cscDetId), info.theElectronicsMap->dmb(cscDetId));

  if (info.formatVersion_ == 2013) {
    // Set DMB version field to distinguish between ME11s and other chambers (ME11 - 2, others - 1)
    bool me11 = ((chamberId.station() == 1) && (chamberId.ring() == 4)) ||
                ((chamberId.station() == 1) && (chamberId.ring() == 1));
    if (me11) {
      cscData.dmbHeader()->setdmbVersion(2);
    } else {
      cscData.dmbHeader()->setdmbVersion(1);
    }
  }
  return cscData;
}

void CSCDigiToRaw::add(const CSCStripDigiCollection& stripDigis,
                       const CSCCLCTPreTriggerCollection* preTriggers,
                       const CSCCLCTPreTriggerDigiCollection* preTriggerDigis,
                       FindEventDataInfo& fedInfo) const {
  //iterate over chambers with strip digis in them
  for (CSCStripDigiCollection::DigiRangeIterator j = stripDigis.begin(); j != stripDigis.end(); ++j) {
    CSCDetId cscDetId = (*j).first;
    // only digitize if there are pre-triggers

    bool me1abCheck = fedInfo.formatVersion_ == 2013;
    // pretrigger flag must be set and the pretrigger collection must be nonzero!
    const bool usePreTriggers = usePreTriggers_ and preTriggers != nullptr;
    if (!usePreTriggers || packEverything_ ||
        (usePreTriggers && cscd2r::accept(cscDetId,
                                          *preTriggerDigis,
                                          preTriggerWindowMin_,
                                          preTriggerWindowMax_,
                                          CSCConstants::CLCT_CENTRAL_BX,
                                          me1abCheck))) {
      bool me1a = (cscDetId.station() == 1) && (cscDetId.ring() == 4);
      bool zplus = (cscDetId.endcap() == 1);
      bool me1b = (cscDetId.station() == 1) && (cscDetId.ring() == 1);

      CSCEventData& cscData = findEventData(cscDetId, fedInfo);

      std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
      std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
      for (; digiItr != last; ++digiItr) {
        CSCStripDigi digi = *digiItr;
        int strip = digi.getStrip();
        // From LS1 on ME1a strips are unganged
        if (fedInfo.formatVersion_ == 2013) {
          if (me1a && zplus) {
            digi.setStrip(CSCConstants::NUM_STRIPS_ME1A_UNGANGED + 1 - strip);  // 1-48 -> 48-1
          }
          if (me1b && !zplus) {
            digi.setStrip(CSCConstants::NUM_STRIPS_ME1B + 1 - strip);  // 1-64 -> 64-1
          }
          if (me1a) {
            strip = digi.getStrip();  // reset back 1-16 to 65-80 digi
            digi.setStrip(strip + CSCConstants::NUM_STRIPS_ME1B);
          }
        }
        // During Run-1 ME1a strips are triple-ganged
        else {
          if (me1a && zplus) {
            digi.setStrip(CSCConstants::NUM_STRIPS_ME1A_GANGED + 1 - strip);  // 1-16 -> 16-1
          }
          if (me1b && !zplus) {
            digi.setStrip(CSCConstants::NUM_STRIPS_ME1B + 1 - strip);  // 1-64 -> 64-1
          }
          if (me1a) {
            strip = digi.getStrip();  // reset back 1-16 to 65-80 digi
            digi.setStrip(strip + CSCConstants::NUM_STRIPS_ME1B);
          }
        }
        cscData.add(digi, cscDetId.layer());
      }
    }
  }
}

void CSCDigiToRaw::add(const CSCWireDigiCollection& wireDigis,
                       const CSCALCTDigiCollection& alctDigis,
                       FindEventDataInfo& fedInfo) const {
  add(alctDigis, fedInfo);
  for (CSCWireDigiCollection::DigiRangeIterator j = wireDigis.begin(); j != wireDigis.end(); ++j) {
    CSCDetId cscDetId = (*j).first;
    bool me1abCheck = fedInfo.formatVersion_ == 2013;
    if (packEverything_ ||
        cscd2r::accept(
            cscDetId, alctDigis, alctWindowMin_, alctWindowMax_, CSCConstants::ALCT_CENTRAL_BX, me1abCheck)) {
      CSCEventData& cscData = findEventData(cscDetId, fedInfo);
      std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
      std::vector<CSCWireDigi>::const_iterator last = (*j).second.second;
      for (; digiItr != last; ++digiItr) {
        cscData.add(*digiItr, cscDetId.layer());
      }
    }
  }
}

void CSCDigiToRaw::add(const CSCComparatorDigiCollection& comparatorDigis,
                       const CSCCLCTDigiCollection& clctDigis,
                       FindEventDataInfo& fedInfo) const {
  add(clctDigis, fedInfo);
  for (auto const& j : comparatorDigis) {
    CSCDetId cscDetId = j.first;
    CSCEventData& cscData = findEventData(cscDetId, fedInfo);
    bool me1abCheck = fedInfo.formatVersion_ == 2013;
    if (packEverything_ ||
        cscd2r::accept(
            cscDetId, clctDigis, clctWindowMin_, clctWindowMax_, CSCConstants::CLCT_CENTRAL_BX, me1abCheck)) {
      bool me1a = (cscDetId.station() == 1) && (cscDetId.ring() == 4);

      /*
        Add the comparator digi to the fedInfo.
        Move ME1/A comparators from CFEB=0 to CFEB=4 if this has not
        been done already.
        Consider the case for triple-ganged and unganged ME1A strips
       */
      for (auto digi = j.second.first; digi != j.second.second; ++digi) {
        if (fedInfo.formatVersion_ == 2013) {
          // unganged case
          if (me1a && digi->getStrip() <= CSCConstants::NUM_STRIPS_ME1A_UNGANGED) {
            CSCComparatorDigi digi_corr(
                CSCConstants::NUM_STRIPS_ME1B + digi->getStrip(), digi->getComparator(), digi->getTimeBinWord());
            cscData.add(digi_corr, cscDetId);
          } else {
            cscData.add(*digi, cscDetId);
          }
        }
        // triple-ganged case
        else {
          if (me1a && digi->getStrip() <= CSCConstants::NUM_STRIPS_ME1A_GANGED) {
            CSCComparatorDigi digi_corr(
                CSCConstants::NUM_STRIPS_ME1B + digi->getStrip(), digi->getComparator(), digi->getTimeBinWord());
            cscData.add(digi_corr, cscDetId.layer());
          } else {
            cscData.add(*digi, cscDetId.layer());
          }
        }
      }
    }
  }
}

void CSCDigiToRaw::add(const CSCALCTDigiCollection& alctDigis, FindEventDataInfo& fedInfo) const {
  for (CSCALCTDigiCollection::DigiRangeIterator j = alctDigis.begin(); j != alctDigis.end(); ++j) {
    CSCDetId cscDetId = (*j).first;
    CSCEventData& cscData = findEventData(cscDetId, fedInfo);

    cscData.add(std::vector<CSCALCTDigi>((*j).second.first, (*j).second.second));
  }
}

void CSCDigiToRaw::add(const CSCCLCTDigiCollection& clctDigis, FindEventDataInfo& fedInfo) const {
  for (CSCCLCTDigiCollection::DigiRangeIterator j = clctDigis.begin(); j != clctDigis.end(); ++j) {
    CSCDetId cscDetId = (*j).first;
    CSCEventData& cscData = findEventData(cscDetId, fedInfo);

    bool me11a = cscDetId.station() == 1 && cscDetId.ring() == 4;
    //CLCTs are packed by chamber not by A/B parts in ME11
    //me11a appears only in simulation with SLHC algorithm settings
    //without the shift, it's impossible to distinguish A and B parts
    if (me11a && fedInfo.formatVersion_ == 2013) {
      std::vector<CSCCLCTDigi> shiftedDigis((*j).second.first, (*j).second.second);
      for (std::vector<CSCCLCTDigi>::iterator iC = shiftedDigis.begin(); iC != shiftedDigis.end(); ++iC) {
        if (iC->getCFEB() < CSCConstants::NUM_CFEBS_ME1A_UNGANGED) {  //sanity check, mostly
          (*iC) = CSCCLCTDigi(iC->isValid(),
                              iC->getQuality(),
                              iC->getPattern(),
                              iC->getStripType(),
                              iC->getBend(),
                              iC->getStrip(),
                              iC->getCFEB() + CSCConstants::NUM_CFEBS_ME1B,
                              iC->getBX(),
                              iC->getTrknmb(),
                              iC->getFullBX());
        }
      }
      cscData.add(shiftedDigis);
    } else {
      cscData.add(std::vector<CSCCLCTDigi>((*j).second.first, (*j).second.second));
    }
  }
}

void CSCDigiToRaw::add(const CSCCorrelatedLCTDigiCollection& corrLCTDigis, FindEventDataInfo& fedInfo) const {
  for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator j = corrLCTDigis.begin(); j != corrLCTDigis.end(); ++j) {
    CSCDetId cscDetId = (*j).first;
    CSCEventData& cscData = findEventData(cscDetId, fedInfo);

    bool me11a = cscDetId.station() == 1 && cscDetId.ring() == 4;
    //LCTs are packed by chamber not by A/B parts in ME11
    //me11a appears only in simulation with SLHC algorithm settings
    //without the shift, it's impossible to distinguish A and B parts
    if (me11a && fedInfo.formatVersion_ == 2013) {
      std::vector<CSCCorrelatedLCTDigi> shiftedDigis((*j).second.first, (*j).second.second);
      for (std::vector<CSCCorrelatedLCTDigi>::iterator iC = shiftedDigis.begin(); iC != shiftedDigis.end(); ++iC) {
        if (iC->getStrip() < CSCConstants::NUM_HALF_STRIPS_ME1A_UNGANGED) {  //sanity check, mostly
          (*iC) = CSCCorrelatedLCTDigi(iC->getTrknmb(),
                                       iC->isValid(),
                                       iC->getQuality(),
                                       iC->getKeyWG(),
                                       iC->getStrip() + CSCConstants::NUM_HALF_STRIPS_ME1B,
                                       iC->getPattern(),
                                       iC->getBend(),
                                       iC->getBX(),
                                       iC->getMPCLink(),
                                       iC->getBX0(),
                                       iC->getSyncErr(),
                                       iC->getCSCID());
        }
      }
      cscData.add(shiftedDigis);
    } else {
      cscData.add(std::vector<CSCCorrelatedLCTDigi>((*j).second.first, (*j).second.second));
    }
  }
}

void CSCDigiToRaw::add(const CSCShowerDigiCollection& cscShowerDigis, FindEventDataInfo& fedInfo) const {
  for (const auto& shower : cscShowerDigis) {
    const CSCDetId& cscDetId = shower.first;
    CSCEventData& cscData = findEventData(cscDetId, fedInfo);

    cscData.add(std::vector<CSCShowerDigi>(shower.second.first, shower.second.second));
  }
}

void CSCDigiToRaw::add(const GEMPadDigiClusterCollection& gemPadClusters, FindEventDataInfo& fedInfo) const {
  for (const auto& jclus : gemPadClusters) {
    const GEMDetId& gemDetId = jclus.first;

    const int zendcap = gemDetId.region() == 1 ? 1 : 2;
    CSCDetId cscDetId(zendcap, gemDetId.station(), 1, gemDetId.chamber(), 0);
    CSCEventData& cscData = findEventData(cscDetId, fedInfo);

    cscData.add(std::vector<GEMPadDigiCluster>(jclus.second.first, jclus.second.second), gemDetId);
  }
}

void CSCDigiToRaw::createFedBuffers(const CSCStripDigiCollection& stripDigis,
                                    const CSCWireDigiCollection& wireDigis,
                                    const CSCComparatorDigiCollection& comparatorDigis,
                                    const CSCALCTDigiCollection& alctDigis,
                                    const CSCCLCTDigiCollection& clctDigis,
                                    const CSCCLCTPreTriggerCollection* preTriggers,
                                    const CSCCLCTPreTriggerDigiCollection* preTriggerDigis,
                                    const CSCCorrelatedLCTDigiCollection& correlatedLCTDigis,
                                    const CSCShowerDigiCollection* cscShowerDigis,
                                    const GEMPadDigiClusterCollection* gemPadDigiClusters,
                                    FEDRawDataCollection& fed_buffers,
                                    const CSCChamberMap* mapping,
                                    const EventID& eid) const {
  //get fed object from fed_buffers
  // make a map from the index of a chamber to the event data from it
  FindEventDataInfo fedInfo{mapping, formatVersion_};
  add(stripDigis, preTriggers, preTriggerDigis, fedInfo);
  add(wireDigis, alctDigis, fedInfo);
  add(comparatorDigis, clctDigis, fedInfo);
  add(correlatedLCTDigis, fedInfo);

  // Starting Run-3, the CSC DAQ will pack/unpack CSC showers
  if (cscShowerDigis) {
    add(*cscShowerDigis, fedInfo);
  }

  // Starting Run-3, the CSC DAQ will pack/unpack GEM clusters
  if (gemPadDigiClusters) {
    add(*gemPadDigiClusters, fedInfo);
  }
  int l1a = eid.event();  //need to add increments or get it from lct digis
  int bx = l1a;           //same as above

  if (fedInfo.formatVersion_ == 2005)  /// Handle pre-LS1 format data
  {
    std::map<int, CSCDCCEventData> dccMap;
    //idcc goes from startingFed to startingFED+7
    for (int idcc = FEDNumbering::MINCSCFEDID; idcc <= FEDNumbering::MAXCSCFEDID; ++idcc) {
      dccMap.insert(std::pair<int, CSCDCCEventData>(idcc, CSCDCCEventData(idcc, CSCConstants::NUM_DDUS, bx, l1a)));
    }

    for (int idcc = FEDNumbering::MINCSCFEDID; idcc <= FEDNumbering::MAXCSCFEDID; ++idcc) {
      // for every chamber with data, add to a DDU in this DCC Event
      for (map<CSCDetId, CSCEventData>::iterator chamberItr = fedInfo.theChamberDataMap.begin();
           chamberItr != fedInfo.theChamberDataMap.end();
           ++chamberItr) {
        int indexDCC = mapping->slink(chamberItr->first);
        if (indexDCC == idcc) {
          //FIXME (What does this mean? Is something wrong?)
          std::map<int, CSCDCCEventData>::iterator dccMapItr = dccMap.find(indexDCC);
          if (dccMapItr == dccMap.end()) {
            throw cms::Exception("CSCDigiToRaw") << "Bad DCC number:" << indexDCC;
          }
          // get id's based on ChamberId from mapping

          int dduId = mapping->ddu(chamberItr->first);
          int dduSlot = mapping->dduSlot(chamberItr->first);
          int dduInput = mapping->dduInput(chamberItr->first);
          int dmbId = mapping->dmb(chamberItr->first);
          dccMapItr->second.addChamber(chamberItr->second, dduId, dduSlot, dduInput, dmbId, formatVersion_);
        }
      }
    }

    // FIXME: FEDRawData size set to 2*64 to add FED header and trailer
    for (std::map<int, CSCDCCEventData>::iterator dccMapItr = dccMap.begin(); dccMapItr != dccMap.end(); ++dccMapItr) {
      boost::dynamic_bitset<> dccBits = dccMapItr->second.pack();
      FEDRawData& fedRawData = fed_buffers.FEDData(dccMapItr->first);
      fedRawData.resize(dccBits.size());
      //fill data with dccEvent
      bitset_utilities::bitsetToChar(dccBits, fedRawData.data());
      FEDTrailer cscFEDTrailer(fedRawData.data() + (fedRawData.size() - 8));
      cscFEDTrailer.set(fedRawData.data() + (fedRawData.size() - 8),
                        fedRawData.size() / 8,
                        evf::compute_crc(fedRawData.data(), fedRawData.size()),
                        0,
                        0);
    }
  }
  /// Handle post-LS1 format data
  else if (formatVersion_ == 2013) {
    std::map<int, CSCDDUEventData> dduMap;
    unsigned int ddu_fmt_version = 0x7;  /// 2013 Format
    const unsigned postLS1_map[] = {841, 842, 843, 844, 845, 846, 847, 848, 849, 831, 832, 833,
                                    834, 835, 836, 837, 838, 839, 861, 862, 863, 864, 865, 866,
                                    867, 868, 869, 851, 852, 853, 854, 855, 856, 857, 858, 859};

    /// Create dummy DDU buffers
    for (unsigned int i = 0; i < 36; i++) {
      unsigned int iddu = postLS1_map[i];
      // make a new one
      CSCDDUHeader newDDUHeader(bx, l1a, iddu, ddu_fmt_version);

      dduMap.insert(std::pair<int, CSCDDUEventData>(iddu, CSCDDUEventData(newDDUHeader)));
    }

    /// Loop over post-LS1 DDU FEDs
    for (unsigned int i = 0; i < 36; i++) {
      unsigned int iddu = postLS1_map[i];
      // for every chamber with data, add to a DDU in this DCC Event
      for (map<CSCDetId, CSCEventData>::iterator chamberItr = fedInfo.theChamberDataMap.begin();
           chamberItr != fedInfo.theChamberDataMap.end();
           ++chamberItr) {
        unsigned int indexDDU = mapping->slink(chamberItr->first);

        /// Lets handle possible mapping issues
        // Still preLS1 DCC FEDs mapping
        if ((indexDDU >= FEDNumbering::MINCSCFEDID) && (indexDDU <= FEDNumbering::MAXCSCFEDID)) {
          int dduID = mapping->ddu(chamberItr->first);  // try to switch to DDU ID mapping
          // DDU ID is in expectedi post-LS1 FED ID range
          if ((dduID >= FEDNumbering::MINCSCDDUFEDID) && (dduID <= FEDNumbering::MAXCSCDDUFEDID)) {
            indexDDU = dduID;
          } else  // Messy
          {
            // Lets try to change pre-LS1 1-36 ID range to post-LS1 MINCSCDDUFEDID - MAXCSCDDUFEDID range
            dduID &= 0xFF;
            // indexDDU = FEDNumbering::MINCSCDDUFEDID + dduID-1;
            if ((dduID <= 36) && (dduID > 0))
              indexDDU = postLS1_map[dduID - 1];
          }
        }

        if (indexDDU == iddu) {
          std::map<int, CSCDDUEventData>::iterator dduMapItr = dduMap.find(indexDDU);
          if (dduMapItr == dduMap.end()) {
            throw cms::Exception("CSCDigiToRaw") << "Bad DDU number:" << indexDDU;
          }
          // get id's based on ChamberId from mapping

          int dduInput = mapping->dduInput(chamberItr->first);
          int dmbId = mapping->dmb(chamberItr->first);
          // int crateId  = mapping->crate(chamberItr->first);
          dduMapItr->second.add(chamberItr->second, dmbId, dduInput, formatVersion_);
        }
      }
    }

    // FIXME: FEDRawData size set to 2*64 to add FED header and trailer
    for (std::map<int, CSCDDUEventData>::iterator dduMapItr = dduMap.begin(); dduMapItr != dduMap.end(); ++dduMapItr) {
      boost::dynamic_bitset<> dduBits = dduMapItr->second.pack();
      FEDRawData& fedRawData = fed_buffers.FEDData(dduMapItr->first);
      fedRawData.resize(dduBits.size() / 8);
      //fill data with dduEvent
      bitset_utilities::bitsetToChar(dduBits, fedRawData.data());
      FEDTrailer cscFEDTrailer(fedRawData.data() + (fedRawData.size() - 8));
      cscFEDTrailer.set(fedRawData.data() + (fedRawData.size() - 8),
                        fedRawData.size() / 8,
                        evf::compute_crc(fedRawData.data(), fedRawData.size()),
                        0,
                        0);
    }
  }
}
