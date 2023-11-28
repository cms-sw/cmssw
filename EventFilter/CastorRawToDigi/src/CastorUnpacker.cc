#include "EventFilter/CastorRawToDigi/interface/CastorUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "EventFilter/HcalRawToDigi/interface/HcalTTPUnpacker.h"
#include "DataFormats/HcalDetId/interface/HcalOtherDetId.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include "DataFormats/HcalDigi/interface/CastorDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveSample.h"
#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace CastorUnpacker_impl {
  template <class DigiClass>
  const HcalQIESample* unpack(const HcalQIESample* startPoint,
                              const HcalQIESample* limit,
                              DigiClass& digi,
                              int presamples,
                              const CastorElectronicsId& eid,
                              int startSample,
                              int endSample,
                              int expectedTime,
                              const HcalHTRData& hhd) {
    // set parameters
    digi.setPresamples(presamples);
    int fiber = startPoint->fiber();
    int fiberchan = startPoint->fiberChan();
    uint32_t zsmask = hhd.zsBunchMask() >> startSample;
    digi.setZSInfo(hhd.isUnsuppressed(), hhd.wasMarkAndPassZS(fiber, fiberchan), zsmask);

    //   digi.setReadoutIds(eid);
    //   setReadoutIds is missing in  CastorDataFrame class  digi.setReadoutIds(eid);
    if (expectedTime >= 0 && !hhd.isUnsuppressed()) {
      // std::cout << hhd.getFibOrbMsgBCN(fiber) << " " << expectedTime << " fiber="<<fiber<< std::endl;
      digi.setFiberIdleOffset(hhd.getFibOrbMsgBCN(fiber) - expectedTime);
    }
    // what is my sample number?
    int myFiberChan = startPoint->fiberAndChan();
    int ncurr = 0, ntaken = 0;
    const HcalQIESample* qie_work = startPoint;
    while (qie_work != limit && qie_work->fiberAndChan() == myFiberChan) {
      if (ncurr >= startSample && ncurr <= endSample) {
        digi.setSample(ntaken, *qie_work);
        ++ntaken;
      }
      ncurr++;
      qie_work++;
    }
    digi.setSize(ntaken);
    return qie_work;
  }
}  // namespace CastorUnpacker_impl

namespace {
  inline bool isTPGSOI(const HcalTriggerPrimitiveSample& s) { return (s.raw() & 0x200) != 0; }
}  // namespace

CastorUnpacker::CastorUnpacker(int sourceIdOffset, int beg, int end)
    : sourceIdOffset_(sourceIdOffset), expectedOrbitMessageTime_(-1) {
  if (beg >= 0 && beg <= CastorDataFrame::MAXSAMPLES - 1) {
    startSample_ = beg;
  } else {
    startSample_ = 0;
  }
  if (end >= 0 && end <= CastorDataFrame::MAXSAMPLES - 1 && end >= beg) {
    endSample_ = end;
  } else {
    endSample_ = CastorDataFrame::MAXSAMPLES - 1;
  }
}

static int slb(const HcalTriggerPrimitiveSample& theSample) { return ((theSample.raw() >> 13) & 0x7); }
static int slbChan(const HcalTriggerPrimitiveSample& theSample) { return (theSample.raw() >> 11) & 0x3; }
static int slbAndChan(const HcalTriggerPrimitiveSample& theSample) { return (theSample.raw() >> 11) & 0x1F; }

void CastorUnpacker::unpack(const FEDRawData& raw,
                            const CastorElectronicsMap& emap,
                            CastorRawCollections& colls,
                            HcalUnpackerReport& report,
                            bool silent) {
  if (raw.size() < 16) {
    if (!silent)
      edm::LogWarning("Invalid Data") << "Empty/invalid DCC data, size = " << raw.size();
    return;
  }

  // get the DCC header
  const HcalDCCHeader* dccHeader = (const HcalDCCHeader*)(raw.data());
  int dccid = dccHeader->getSourceId() - sourceIdOffset_;

  // check the summary status

  // walk through the HTR data...
  HcalHTRData htr;
  const unsigned short *daq_first, *daq_last, *tp_first, *tp_last;
  const HcalQIESample *qie_begin, *qie_end, *qie_work;
  const HcalTriggerPrimitiveSample *tp_begin, *tp_end, *tp_work;
  for (int spigot = 0; spigot < HcalDCCHeader::SPIGOT_COUNT; spigot++) {
    if (!dccHeader->getSpigotPresent(spigot))
      continue;

    int retval = dccHeader->getSpigotData(spigot, htr, raw.size());
    if (retval != 0) {
      if (retval == -1) {
        if (!silent)
          edm::LogWarning("Invalid Data") << "Invalid HTR data (data beyond payload size) observed on spigot " << spigot
                                          << " of DCC with source id " << dccHeader->getSourceId();
        report.countSpigotFormatError();
      }
      continue;
    }
    // check
    if (dccHeader->getSpigotCRCError(spigot)) {
      if (!silent)
        edm::LogWarning("Invalid Data") << "CRC Error on HTR data observed on spigot " << spigot
                                        << " of DCC with source id " << dccHeader->getSourceId();
      report.countSpigotFormatError();
      continue;
    }
    if (!htr.check()) {
      if (!silent)
        edm::LogWarning("Invalid Data") << "Invalid HTR data observed on spigot " << spigot << " of DCC with source id "
                                        << dccHeader->getSourceId();
      report.countSpigotFormatError();
      continue;
    }
    if (htr.isHistogramEvent()) {
      if (!silent)
        edm::LogWarning("Invalid Data") << "Histogram data passed to non-histogram unpacker on spigot " << spigot
                                        << " of DCC with source id " << dccHeader->getSourceId();
      continue;
    }
    if ((htr.getFirmwareFlavor() & 0xE0) == 0x80) {  // some kind of TTP data
      if (colls.ttp != nullptr) {
        HcalTTPUnpacker ttpUnpack;
        colls.ttp->push_back(HcalTTPDigi());
        ttpUnpack.unpack(htr, colls.ttp->back());
      } else {
        LogDebug("CastorUnpackerHcalTechTrigProcessor")
            << "Skipping data on spigot " << spigot << " of DCC with source id " << dccHeader->getSourceId()
            << " which is from the TechTrigProcessor (use separate unpacker!)";
      }
      continue;
    }
    if (htr.getFirmwareFlavor() >= 0x80) {
      if (!silent)
        edm::LogWarning("CastorUnpacker")
            << "Skipping data on spigot " << spigot << " of DCC with source id " << dccHeader->getSourceId()
            << " which is of unknown flavor " << htr.getFirmwareFlavor();
      continue;
    }
    // calculate "real" number of presamples
    int nps = htr.getNPS() - startSample_;

    // get pointers
    htr.dataPointers(&daq_first, &daq_last, &tp_first, &tp_last);
    unsigned int smid = htr.getSubmodule();
    int htr_tb = smid & 0x1;
    int htr_slot = (smid >> 1) & 0x1F;
    int htr_cr = (smid >> 6) & 0x1F;

    tp_begin = (const HcalTriggerPrimitiveSample*)tp_first;
    tp_end = (const HcalTriggerPrimitiveSample*)(tp_last + 1);  // one beyond last..

    /// work through the samples
    int currFiberChan = 0x3F;  // invalid fiber+channel...
    int ncurr = 0;
    bool valid = false;
    //////////////////////////////////////////////
    bool tpgSOIbitInUse = htr.getFormatVersion() >= 3;  // version 3 and later
    // bool isHOtpg=htr.getFormatVersion()>=3 && htr.getFirmwareFlavor()==0; // HO is flavor zero
    /*
      Unpack the trigger primitives
    */
    // lookup the right channel
    bool dotp = true;
    CastorElectronicsId eid(0, 1, spigot, dccid);
    eid.setHTR(htr_cr, htr_slot, htr_tb);
    DetId did = emap.lookup(eid);
    if (did.null())
      dotp = false;
    HcalCastorDetId id1(did);
    HcalCastorDetId id((id1.zside() == 0), id1.sector(), 1);
    if (id1.module() > 12)
      dotp = false;
    if (dotp) {
      // std::cout << " tp_first="<< tp_first << " tp_last="<< tp_last<< " tb="<<htr_tb<<" slot="<<htr_slot<<" crate="<<htr_cr<<" dccid="<< dccid<< std::endl;
      // regular TPs (not HO)
      for (tp_work = tp_begin; tp_work != tp_end; tp_work++) {
        //	  std::cout << "raw=0x"<<std::hex<< tp_work->raw()<<std::dec <<std::endl;
        if (tp_work->raw() == 0xFFFF)
          continue;                                   // filler word
        if (slbAndChan(*tp_work) != currFiberChan) {  // start new set
          currFiberChan = slbAndChan(*tp_work);

          // std::cout<< " NEW SET "<<std::endl;
          //HcalElectronicsId eid(tp_work->slbChan(),tp_work->slb(),spigot,dccid,htr_cr,htr_slot,htr_tb);
          //DetId did=emap.lookupTrigger(eid);
          //if (did.null()) {
          //report.countUnmappedTPDigi(eid);
          //if (unknownIdsTrig_.find(eid)==unknownIdsTrig_.end()) {
          //if (!silent) edm::LogWarning("HCAL") << "HcalUnpacker: No trigger primitive match found for electronics id :" << eid;
          //unknownIdsTrig_.insert(eid);
          //}
          //valid=false;
          //continue;
          //} else if (did==HcalTrigTowerDetId::Undefined ||
          //(did.det()==DetId::Hcal && did.subdetId()==0)) {
          //// known to be unmapped
          //valid=false;
          //continue;
          //}

          colls.tpCont->push_back(CastorTriggerPrimitiveDigi(id));
          // set the various bits
          if (!tpgSOIbitInUse)
            colls.tpCont->back().setPresamples(nps);
          colls.tpCont->back().setZSInfo(htr.isUnsuppressed(),
                                         htr.wasMarkAndPassZSTP(slb(*tp_work), slbChan(*tp_work)));

          // no hits recorded for current
          ncurr = 0;
          valid = true;
        }
        // add the word (if within settings or recent firmware [recent firmware ignores startSample/endSample])
        if (valid && ((tpgSOIbitInUse && ncurr < 10) || (ncurr >= startSample_ && ncurr <= endSample_))) {
          colls.tpCont->back().setSample(colls.tpCont->back().size(), *tp_work);
          colls.tpCont->back().setSize(colls.tpCont->back().size() + 1);
        }
        // set presamples,if SOI
        if (valid && tpgSOIbitInUse && isTPGSOI(*tp_work)) {
          colls.tpCont->back().setPresamples(ncurr);
        }
        ncurr++;
      }
    }

    //////////////////////////////////////////////
    qie_begin = (const HcalQIESample*)daq_first;
    qie_end = (const HcalQIESample*)(daq_last + 1);  // one beyond last..

    /// work through the samples

    for (qie_work = qie_begin; qie_work != qie_end;) {
      if (qie_work->raw() == 0xFFFF) {
        qie_work++;
        continue;  // filler word
      }

      // lookup the right channel
      CastorElectronicsId eid(qie_work->fiberChan(), qie_work->fiber(), spigot, dccid);
      eid.setHTR(htr_cr, htr_slot, htr_tb);
      DetId did = emap.lookup(eid);

      if (!did.null()) {
        colls.castorCont->push_back(CastorDataFrame(HcalCastorDetId(did)));
        qie_work = CastorUnpacker_impl::unpack<CastorDataFrame>(qie_work,
                                                                qie_end,
                                                                colls.castorCont->back(),
                                                                nps,
                                                                eid,
                                                                startSample_,
                                                                endSample_,
                                                                expectedOrbitMessageTime_,
                                                                htr);
      } else {
        report.countUnmappedDigi();
        if (unknownIds_.find(eid) == unknownIds_.end()) {
          if (!silent)
            edm::LogWarning("CASTOR") << "CastorUnpacker: No match found for electronics id :" << eid;
          unknownIds_.insert(eid);
        }
        for (int fiberC = qie_work->fiberAndChan(); qie_work != qie_end && qie_work->fiberAndChan() == fiberC;
             qie_work++)
          ;
      }
    }
  }
}
