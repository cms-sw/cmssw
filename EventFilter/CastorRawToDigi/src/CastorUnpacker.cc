#include "EventFilter/CastorRawToDigi/interface/CastorUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "DataFormats/HcalDetId/interface/HcalOtherDetId.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include "DataFormats/HcalDigi/interface/CastorDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveSample.h"
#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace CastorUnpacker_impl {
  template <class DigiClass>
  const HcalQIESample* unpack(const HcalQIESample* startPoint, const HcalQIESample* limit, DigiClass& digi, int presamples, const CastorElectronicsId& eid, int startSample, int endSample) {
    // set parameters
    digi.setPresamples(presamples);
 //   digi.setReadoutIds(eid);
 //   setReadoutIds is missing in  CastorDataFrame class  digi.setReadoutIds(eid);

    // what is my sample number?
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


CastorUnpacker::CastorUnpacker(int sourceIdOffset, int beg, int end) : sourceIdOffset_(sourceIdOffset)
{
 if ( beg >= 0 && beg <= CastorDataFrame::MAXSAMPLES -1 ) {
 	startSample_ = beg;
 } else {
 	startSample_ = 0;
 }
 if ( end >= 0 && end <= CastorDataFrame::MAXSAMPLES -1 && end >= beg ) {
 	endSample_ = end;
 } else {
 	endSample_ = CastorDataFrame::MAXSAMPLES -1;
 }
}


void CastorUnpacker::unpack(const FEDRawData& raw, const CastorElectronicsMap& emap,
			  CastorRawCollections& colls, HcalUnpackerReport& report) {

  if (raw.size()<16) {
    edm::LogWarning("Invalid Data") << "Empty/invalid DCC data, size = " << raw.size();
    return;
  }

  // get the DCC header
  const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
  int dccid=dccHeader->getSourceId()-sourceIdOffset_;

  // check the summary status
  
  // walk through the HTR data...
  HcalHTRData htr;
  const unsigned short* daq_first, *daq_last, *tp_first, *tp_last;
  const HcalQIESample* qie_begin, *qie_end, *qie_work;
//   const HcalTriggerPrimitiveSample *tp_begin, *tp_end, *tp_work; 
  for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {
    if (!dccHeader->getSpigotPresent(spigot)) continue;

    int retval=dccHeader->getSpigotData(spigot,htr,raw.size());
    if (retval!=0) {
      if (retval==-1) {
	edm::LogWarning("Invalid Data") << "Invalid HTR data (data beyond payload size) observed on spigot " << spigot << " of DCC with source id " << dccHeader->getSourceId();
	report.countSpigotFormatError();
      }
      continue;
    }
    // check
    if (!htr.check()) {
      edm::LogWarning("Invalid Data") << "Invalid HTR data observed on spigot " << spigot << " of DCC with source id " << dccHeader->getSourceId();
      report.countSpigotFormatError();
      continue;
    }
    if (htr.isHistogramEvent()) {
      edm::LogWarning("Invalid Data") << "Histogram data passed to non-histogram unpacker on spigot " << spigot << " of DCC with source id " << dccHeader->getSourceId();
      continue;

    }
    // calculate "real" number of presamples
    int nps=htr.getNPS()-startSample_;
    
    // get pointers
    htr.dataPointers(&daq_first,&daq_last,&tp_first,&tp_last);
    unsigned int smid=htr.getSubmodule();
    int htr_tb=smid&0x1;
    int htr_slot=(smid>>1)&0x1F;
    int htr_cr=(smid>>6)&0x1F;
    
    
    /// work through the samples
    int currFiberChan=0x3F; // invalid fiber+channel...
    int ncurr=0;
    bool valid=false;

    qie_begin=(HcalQIESample*)daq_first;
    qie_end=(HcalQIESample*)(daq_last+1); // one beyond last..

    /// work through the samples
    currFiberChan=0x3F; // invalid fiber+channel...
    ncurr=0;
    valid=false;
    
    for (qie_work=qie_begin; qie_work!=qie_end; ) {
      if (qie_work->raw()==0xFFFF) {
	qie_work++;
	continue; // filler word
      }
      // always at the beginning ...
      currFiberChan=qie_work->fiberAndChan();

      // lookup the right channel
      CastorElectronicsId eid(qie_work->fiberChan(),qie_work->fiber(),spigot,dccid);
      eid.setHTR(htr_cr,htr_slot,htr_tb);
      DetId did=emap.lookup(eid);

      if (!did.null()) {
	    colls.castorCont->push_back(CastorDataFrame(HcalCastorDetId(did)));
	    qie_work=CastorUnpacker_impl::unpack<CastorDataFrame>(qie_work, qie_end, colls.castorCont->back(), nps, eid, startSample_, endSample_);
      } else {
	report.countUnmappedDigi();
   	if (unknownIds_.find(eid)==unknownIds_.end()) {
   	  edm::LogWarning("CASTOR") << "CastorUnpacker: No match found for electronics id :" << eid;
   	  unknownIds_.insert(eid);
   	}
	for (int fiberC=qie_work->fiberAndChan();
	     qie_work!=qie_end && qie_work->fiberAndChan()==fiberC;
	     qie_work++);
      }
    }
  }
}

