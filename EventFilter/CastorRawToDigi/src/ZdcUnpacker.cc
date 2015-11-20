#include "EventFilter/CastorRawToDigi/interface/ZdcUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "EventFilter/HcalRawToDigi/interface/HcalTTPUnpacker.h"
#include "DataFormats/HcalDetId/interface/HcalOtherDetId.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include "DataFormats/HcalDigi/interface/ZDCDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveSample.h"
#include "EventFilter/HcalRawToDigi/interface/AMC13Header.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDTCHeader.h"
#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <map>

namespace ZdcUnpacker_impl {
  template <class DigiClass>
  const HcalQIESample* unpack(const HcalQIESample* startPoint, const HcalQIESample* limit, DigiClass& digi, int presamples, const CastorElectronicsId& eid, int startSample, int endSample, int expectedTime, const HcalHTRData& hhd) {
    // set parameters
    digi.setPresamples(presamples);
    int fiber=startPoint->fiber();
    int fiberchan=startPoint->fiberChan();
    uint32_t zsmask=hhd.zsBunchMask()>>startSample;
    digi.setZSInfo(hhd.isUnsuppressed(),hhd.wasMarkAndPassZS(fiber,fiberchan),zsmask);

    if (expectedTime>=0 && !hhd.isUnsuppressed()) {
      digi.setFiberIdleOffset(hhd.getFibOrbMsgBCN(fiber)-expectedTime);
    }
    int myFiberChan=startPoint->fiberAndChan();
    int ncurr=0,ntaken=0;
    const HcalQIESample* qie_work=startPoint;
    while (qie_work!=limit && qie_work->fiberAndChan()==myFiberChan) {
      if (ncurr>=startSample && ncurr<=endSample) {
        digi.setSample(ntaken,*qie_work);
        ++ntaken;
      }
      ncurr++;
      qie_work++;
    }
    digi.setSize(ntaken);
    return qie_work;
  }
}

namespace ZdcUnpacker_implOldData {
  template <class DigiClass>
  const HcalQIESample* unpackOld(const HcalQIESample* startPoint, const HcalQIESample* limit, DigiClass& digi, int presamples, const HcalElectronicsId& eid, int startSample, int endSample, int expectedTime, const HcalHTRData& hhd) {
    // set parameters
    digi.setPresamples(presamples);
    int fiber=startPoint->fiber();
    int fiberchan=startPoint->fiberChan();
    uint32_t zsmask=hhd.zsBunchMask()>>startSample;
    digi.setZSInfo(hhd.isUnsuppressed(),hhd.wasMarkAndPassZS(fiber,fiberchan),zsmask);

    if (expectedTime>=0 && !hhd.isUnsuppressed()) {
      digi.setFiberIdleOffset(hhd.getFibOrbMsgBCN(fiber)-expectedTime);
    }
    int myFiberChan=startPoint->fiberAndChan();
    int ncurr=0,ntaken=0;
    const HcalQIESample* qie_work=startPoint;
    while (qie_work!=limit && qie_work->fiberAndChan()==myFiberChan) {
      if (ncurr>=startSample && ncurr<=endSample) {
        digi.setSample(ntaken,*qie_work);
        ++ntaken;
      }
      ncurr++;
      qie_work++;
    }
    digi.setSize(ntaken);
    return qie_work;
  }
}

namespace { inline bool isTPGSOI(const HcalTriggerPrimitiveSample& s) {
  return (s.raw()&0x200)!=0;
}
}


ZdcUnpacker::ZdcUnpacker(int sourceIdOffset, int beg, int end) : sourceIdOffset_(sourceIdOffset) , expectedOrbitMessageTime_(-1)
{
  if ( beg >= 0 && beg <= ZDCDataFrame::MAXSAMPLES -1 ) {
    startSample_ = beg;
  } else {
    startSample_ = 0;
  }
  if ( end >= 0 && end <= ZDCDataFrame::MAXSAMPLES -1 && end >= beg ) {
    endSample_ = end;
  } else {
    endSample_ = ZDCDataFrame::MAXSAMPLES -1;
  }
}

void ZdcUnpacker::unpack(const FEDRawData& raw, const CastorElectronicsMap& emap,
                         CastorRawCollections& colls, HcalUnpackerReport& report, bool silent) {

  if (raw.size()<16) {
    if (!silent) edm::LogWarning("Invalid Data") << "Empty/invalid DCC data, size = " << raw.size();
    return;
  }

  // get the DCC header
  const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
  int dccid=dccHeader->getSourceId();//-sourceIdOffset_;

  // walk through the HTR data...
  HcalHTRData htr;
  const unsigned short* daq_first, *daq_last, *tp_first, *tp_last;
  const HcalQIESample* qie_begin, *qie_end, *qie_work;
  std::map<CastorElectronicsId,uint32_t> myEMap;
  std::map<HcalElectronicsId,uint32_t> myEMapOldData;


//////ZDC MAP for NEW data (2015 PbPb are newer)
  //PZDC
  myEMap[CastorElectronicsId(0,1,0,3,18,8,1)]=0x54000051;//PZDC EM1
  myEMap[CastorElectronicsId(1,1,0,3,18,8,1)]=0x54000052;//PZDC EM2
  myEMap[CastorElectronicsId(2,1,0,3,18,8,1)]=0x54000053;//PZDC EM3
  myEMap[CastorElectronicsId(0,2,0,3,18,8,1)]=0x54000054;//PZDC EM4
  myEMap[CastorElectronicsId(1,2,0,3,18,8,1)]=0x54000055;//PZDC EM5
  myEMap[CastorElectronicsId(2,2,0,3,18,8,1)]=0x54000061;//PZDC HAD1
  myEMap[CastorElectronicsId(0,3,0,3,18,8,1)]=0x54000062;//PZDC HAD2
  myEMap[CastorElectronicsId(1,3,0,3,18,8,1)]=0x54000063;//PZDC HAD3
  myEMap[CastorElectronicsId(2,3,0,3,18,8,1)]=0x54000064;//PZDC HAD4

  //NZDC
  myEMap[CastorElectronicsId(0,1,1,3,18,8,0)]=0x54000011;//NZDC EM1
  myEMap[CastorElectronicsId(1,1,1,3,18,8,0)]=0x54000012;//NZDC EM2
  myEMap[CastorElectronicsId(2,1,1,3,18,8,0)]=0x54000013;//NZDC EM3
  myEMap[CastorElectronicsId(0,2,1,3,18,8,0)]=0x54000014;//NZDC EM4
  myEMap[CastorElectronicsId(1,2,1,3,18,8,0)]=0x54000015;//NZDC EM5
  myEMap[CastorElectronicsId(2,2,1,3,18,8,0)]=0x54000021;//NZDC HAD1
  myEMap[CastorElectronicsId(0,3,1,3,18,8,0)]=0x54000022;//NZDC HAD2
  myEMap[CastorElectronicsId(1,3,1,3,18,8,0)]=0x54000023;//NZDC HAD3
  myEMap[CastorElectronicsId(2,3,1,3,18,8,0)]=0x54000024;//NZDC HAD4

/////ZDC MAP for OLD data (2013 pPb run and earlier)
  //PZDC
  myEMapOldData[HcalElectronicsId(0,1,12,22,12,8,1)]=0x54000051;//PZDC EM1
  myEMapOldData[HcalElectronicsId(1,1,12,22,12,8,1)]=0x54000052;//PZDC EM2
  myEMapOldData[HcalElectronicsId(2,1,12,22,12,8,1)]=0x54000053;//PZDC EM3
  myEMapOldData[HcalElectronicsId(0,2,12,22,12,8,1)]=0x54000054;//PZDC EM4
  myEMapOldData[HcalElectronicsId(1,2,12,22,12,8,1)]=0x54000055;//PZDC EM5
  myEMapOldData[HcalElectronicsId(2,2,12,22,12,8,1)]=0x54000061;//PZDC HAD1
  myEMapOldData[HcalElectronicsId(0,3,12,22,12,8,1)]=0x54000062;//PZDC HAD2
  myEMapOldData[HcalElectronicsId(1,3,12,22,12,8,1)]=0x54000063;//PZDC HAD3
  myEMapOldData[HcalElectronicsId(2,3,12,22,12,8,1)]=0x54000064;//PZDC HAD4

  //NZDC
  myEMapOldData[HcalElectronicsId(0,1,13,22,12,8,0)]=0x54000011;//NZDC EM1
  myEMapOldData[HcalElectronicsId(1,1,13,22,12,8,0)]=0x54000012;//NZDC EM2
  myEMapOldData[HcalElectronicsId(2,1,13,22,12,8,0)]=0x54000013;//NZDC EM3
  myEMapOldData[HcalElectronicsId(0,2,13,22,12,8,0)]=0x54000014;//NZDC EM4
  myEMapOldData[HcalElectronicsId(1,2,13,22,12,8,0)]=0x54000015;//NZDC EM5
  myEMapOldData[HcalElectronicsId(2,2,13,22,12,8,0)]=0x54000021;//NZDC HAD1
  myEMapOldData[HcalElectronicsId(0,3,13,22,12,8,0)]=0x54000022;//NZDC HAD2
  myEMapOldData[HcalElectronicsId(1,3,13,22,12,8,0)]=0x54000023;//NZDC HAD3
  myEMapOldData[HcalElectronicsId(2,3,13,22,12,8,0)]=0x54000024;//NZDC HAD4
  //slot is 17

  for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {
    if (!dccHeader->getSpigotPresent(spigot)) continue;

    int retval=dccHeader->getSpigotData(spigot,htr,raw.size());
    if (retval!=0) {
      if (retval==-1) {
        if (!silent) edm::LogWarning("Invalid Data") << "Invalid HTR data (data beyond payload size) observed on spigot " << spigot << " of DCC with source id " << dccHeader->getSourceId();
        report.countSpigotFormatError();
      }
      continue;
    }
    // check
    if (dccHeader->getSpigotCRCError(spigot)) {
      if (!silent)
        edm::LogWarning("Invalid Data") << "CRC Error on HTR data observed on spigot " << spigot << " of DCC with source id " << dccHeader->getSourceId();
      report.countSpigotFormatError();
      continue;
    }
    if (!htr.check()) {
      if (!silent)
        edm::LogWarning("Invalid Data") << "Invalid HTR data observed on spigot " << spigot << " of DCC with source id " << dccHeader->getSourceId();
      report.countSpigotFormatError();
      continue;
    }
    if (htr.isHistogramEvent()) {
      if (!silent)
        edm::LogWarning("Invalid Data") << "Histogram data passed to non-histogram unpacker on spigot " << spigot << " of DCC with source id " << dccHeader->getSourceId();
      continue;

    }
    if ((htr.getFirmwareFlavor()&0xE0)==0x80) { // some kind of TTP data
      if (colls.ttp!=0) {
        HcalTTPUnpacker ttpUnpack;
        colls.ttp->push_back(HcalTTPDigi());
        ttpUnpack.unpack(htr,colls.ttp->back());
      } else {
        LogDebug("ZdcUnpackerHcalTechTrigProcessor") << "Skipping data on spigot " << spigot << " of DCC with source id " << dccHeader->getSourceId() << " which is from the TechTrigProcessor (use separate unpacker!)";
      }
      continue;
    }
    if (htr.getFirmwareFlavor()>=0x80) {
      if (!silent) edm::LogWarning("ZdcUnpacker") << "Skipping data on spigot " << spigot << " of DCC with source id " << dccHeader->getSourceId() << " which is of unknown flavor " << htr.getFirmwareFlavor();
      continue;
    }
    // calculate "real" number of presamples
    //int nps=htr.getNPS()-startSample_;

    // get pointers
    htr.dataPointers(&daq_first,&daq_last,&tp_first,&tp_last);
    unsigned int smid=htr.getSubmodule();
    int htr_tb=smid&0x1;
    int htr_slot=(smid>>1)&0x1F;
    int htr_cr=(smid>>6)&0x1F;

    //////////////////////////////////////////////
    qie_begin=(const HcalQIESample*)daq_first;
    qie_end=(const HcalQIESample*)(daq_last+1); // one beyond last..

    for (qie_work=qie_begin; qie_work!=qie_end;) {    
      if (qie_work->raw()==0xFFFF) {
        qie_work++;
        continue; // filler word
      }
      // lookup the right channel
      HcalElectronicsId eidOld(qie_work->fiberChan(),qie_work->fiber(),spigot,dccid);
      CastorElectronicsId eid(qie_work->fiberChan(),qie_work->fiber(),spigot,dccid);
      eid.setHTR(htr_cr,htr_slot,htr_tb);
	  if(myEMap.find(CastorElectronicsId(qie_work->fiberChan(),qie_work->fiber(),spigot,3,htr_cr,htr_slot,htr_tb))->second!=0x0)
	  {
	  colls.zdcCont->push_back(ZDCDataFrame(HcalZDCDetId(myEMap.find(CastorElectronicsId(qie_work->fiberChan(),qie_work->fiber(),spigot,3,htr_cr,htr_slot,htr_tb))->second)));
	  qie_work=ZdcUnpacker_impl::unpack<ZDCDataFrame>(qie_work,qie_end,colls.zdcCont->back(),0,CastorElectronicsId(qie_work->fiberChan(),qie_work->fiber(),spigot,3,htr_cr,htr_slot,htr_tb),startSample_,endSample_,expectedOrbitMessageTime_,htr);
	  }
	  else if (myEMapOldData.find(HcalElectronicsId(qie_work->fiberChan(),qie_work->fiber(),spigot,22,htr_cr,htr_slot,htr_tb))->second!=0x0)
	  {
	  colls.zdcCont->push_back(ZDCDataFrame(HcalZDCDetId(myEMapOldData.find(HcalElectronicsId(qie_work->fiberChan(),qie_work->fiber(),spigot,22,htr_cr,htr_slot,htr_tb))->second)));
	  qie_work=ZdcUnpacker_implOldData::unpackOld<ZDCDataFrame>(qie_work,qie_end,colls.zdcCont->back(),0,HcalElectronicsId(qie_work->fiberChan(),qie_work->fiber(),spigot,22,htr_cr,htr_slot,htr_tb),startSample_,endSample_,expectedOrbitMessageTime_,htr);
	  }
	  else for (int fiberC=qie_work->fiberAndChan();
	  qie_work!=qie_end && qie_work->fiberAndChan()==fiberC;
	  qie_work++);


    }//end of loop over qies
  }//end of loop over spigots
}


//  LocalWords:  htr
