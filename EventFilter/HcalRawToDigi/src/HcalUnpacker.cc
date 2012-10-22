#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "DataFormats/HcalDetId/interface/HcalOtherDetId.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/HcalRawToDigi/interface/HcalTTPUnpacker.h"

namespace HcalUnpacker_impl {
  template <class DigiClass>
  const HcalQIESample* unpack(const HcalQIESample* startPoint, const HcalQIESample* limit, DigiClass& digi, int presamples, const HcalElectronicsId& eid, int startSample, int endSample, int expectedTime, const HcalHTRData& hhd) {
    // set parameters
    digi.setPresamples(presamples);
    digi.setReadoutIds(eid);

    int fiber=startPoint->fiber();
    int fiberchan=startPoint->fiberChan();
    uint32_t zsmask=hhd.zsBunchMask()>>startSample;
    digi.setZSInfo(hhd.isUnsuppressed(),hhd.wasMarkAndPassZS(fiber,fiberchan),zsmask);

    if (expectedTime>=0 && !hhd.isUnsuppressed()) {
      //      std::cout << hhd.getFibOrbMsgBCN(fiber) << " " << expectedTime << std::endl;
      digi.setFiberIdleOffset(hhd.getFibOrbMsgBCN(fiber)-expectedTime);
    }

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


  template <class DigiClass>
  const unsigned short* unpack_compact(const unsigned short* startPoint, const unsigned short* limit, DigiClass& digi, 
				       int presamples, const HcalElectronicsId& eid, int startSample, int endSample, 
				       int expectedTime, const HcalHTRData& hhd) {
    // set parameters
    digi.setPresamples(presamples);
    digi.setReadoutIds(eid);
    int flavor, error_flags, capid0, channelid;

    HcalHTRData::unpack_per_channel_header(*startPoint,flavor,error_flags,capid0,channelid);
    bool isCapRotating=!(error_flags&0x1);
    bool fiberErr=(error_flags&0x2);
    bool dataValid=!(error_flags&0x2);
    int fiberchan=channelid&0x3;
    int fiber=((channelid>>2)&0x7)+1;

    uint32_t zsmask=hhd.zsBunchMask()>>startSample;
    digi.setZSInfo(hhd.isUnsuppressed(),hhd.wasMarkAndPassZS(fiber,fiberchan),zsmask);

    if (expectedTime>=0 && !hhd.isUnsuppressed()) {
      //      std::cout << hhd.getFibOrbMsgBCN(fiber) << " " << expectedTime << std::endl;
      digi.setFiberIdleOffset(hhd.getFibOrbMsgBCN(fiber)-expectedTime);
    }

    // what is my sample number?
    int ncurr=0,ntaken=0;
    const unsigned short* qie_work=startPoint;
    // we branch here between normal (flavor=5) and error mode (flavor=6)
    if (flavor==5) {
      for (qie_work++; qie_work!=limit && !HcalHTRData::is_channel_header(*qie_work); qie_work++) {
	int capidn=(isCapRotating)?((capid0+ncurr)%4):(capid0);
	int capidn1=(isCapRotating)?((capid0+ncurr+1)%4):(capid0);
	// two samples in one...
	HcalQIESample s0((*qie_work)&0x7F,capidn,fiber,fiberchan,dataValid,fiberErr);
	HcalQIESample s1(((*qie_work)>>8)&0x7F,capidn1,fiber,fiberchan,dataValid,fiberErr);
	
	if (ncurr>=startSample && ncurr<=endSample) {
	  digi.setSample(ntaken,s0);
	  ++ntaken;
	}
	ncurr++;
	if (ncurr>=startSample && ncurr<=endSample) {
	  digi.setSample(ntaken,s1);
	  ++ntaken;
	}
	ncurr++;
      }
      digi.setSize(ntaken);
    } else if (flavor==6) {
      for (qie_work++; qie_work!=limit && !HcalHTRData::is_channel_header(*qie_work); qie_work++) {
	if (ncurr>=startSample && ncurr<=endSample) {
	  HcalQIESample sample((*qie_work)&0x7F,((*qie_work)>>8)&0x3,fiber,fiberchan,((*qie_work)>>10)&0x1,((*qie_work)>>11)&0x1);
	  digi.setSample(ntaken,sample);
	  ++ntaken;
	}
	ncurr++;
      }
      digi.setSize(ntaken);
    }
    return qie_work;
  }

}

static inline bool isTPGSOI(const HcalTriggerPrimitiveSample& s) {
  return (s.raw()&0x200)!=0;
}


struct HOUnrolledTP { // parts of an HO trigger primitive, unpacked
  bool valid, checked;
  int ieta, iphi, samples, soi;
  unsigned int databits;
  HOUnrolledTP() {
    valid=false;
    checked=false;
    ieta=0;
    iphi=0;
    samples=0;
    soi=0;
    databits=0;
  }
  void setbit(int i) { databits|=(1<<i); }    
};

void HcalUnpacker::unpack(const FEDRawData& raw, const HcalElectronicsMap& emap,
			  Collections& colls, HcalUnpackerReport& report, bool silent) {

  if (raw.size()<16) {
    if (!silent) edm::LogWarning("Invalid Data") << "Empty/invalid DCC data, size = " << raw.size();
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
  const HcalTriggerPrimitiveSample *tp_begin, *tp_end, *tp_work; 
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
    // check for EE
    if (htr.isEmptyEvent()) {
      report.countEmptyEventSpigot();
    }
    if (htr.isOverflowWarning()) {
      report.countOFWSpigot();
    }
    if (htr.isBusy()) {
      report.countBusySpigot();
    }
    if (!htr.check()) {
      if (!silent) 
	edm::LogWarning("Invalid Data") << "Invalid HTR data observed on spigot " << spigot << " of DCC with source id " << dccHeader->getSourceId();
      report.countSpigotFormatError();
      continue;
    }  
    if (htr.isHistogramEvent()) {
      if (!silent) edm::LogWarning("Invalid Data") << "Histogram data passed to non-histogram unpacker on spigot " << spigot << " of DCC with source id " << dccHeader->getSourceId();
      continue;
    }
    if ((htr.getFirmwareFlavor()&0xE0)==0x80) { // some kind of TTP data
      if (colls.ttp!=0) {
	HcalTTPUnpacker ttpUnpack;
	colls.ttp->push_back(HcalTTPDigi());
	ttpUnpack.unpack(htr,colls.ttp->back());
      } else {
	LogDebug("HcalTechTrigProcessor") << "Skipping data on spigot " << spigot << " of DCC with source id " << dccHeader->getSourceId() << " which is from the TechTrigProcessor (use separate unpacker!)";
      }
      continue;
    }
    if (htr.getFirmwareFlavor()>=0x80) {
      if (!silent) edm::LogWarning("HcalUnpacker") << "Skipping data on spigot " << spigot << " of DCC with source id " << dccHeader->getSourceId() << " which is of unknown flavor " << htr.getFirmwareFlavor();
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
    
    tp_begin=(HcalTriggerPrimitiveSample*)tp_first;
    tp_end=(HcalTriggerPrimitiveSample*)(tp_last+1); // one beyond last..
    
    /// work through the samples
    int currFiberChan=0x3F; // invalid fiber+channel...
    int ncurr=0;
    bool valid=false;

    bool tpgSOIbitInUse=htr.getFormatVersion()>=3; // version 3 and later
    bool isHOtpg=htr.getFormatVersion()>=3 && htr.getFirmwareFlavor()==0; // HO is flavor zero
    int npre=0;
    /*
      Unpack the trigger primitives
    */
    if (isHOtpg) {
      HOUnrolledTP unrolled[24];
      for (tp_work=tp_begin; tp_work!=tp_end; tp_work++) {
	if (tp_work->raw()==0xFFFF) continue; // filler word
	int sector=tp_work->slbChan();
	if (sector>2) continue;

	for (int ibit=0; ibit<8; ibit++) {
	  int linear=sector*8+ibit; 
	  if (!unrolled[linear].checked) {
	    unrolled[linear].checked=true;
	    int fiber=(linear/3)+1;
	    int fc=(linear%3);
	    // electronics id (use precision match for HO TP)
	    HcalElectronicsId eid(fc,fiber,spigot,dccid);	
	    eid.setHTR(htr_cr,htr_slot,htr_tb);
	    DetId did=emap.lookup(eid);
	    if (!did.null()) {
	      if (did.det()==DetId::Hcal && ((HcalSubdetector)did.subdetId())==HcalOuter ) {
		HcalDetId hid(did);
		unrolled[linear].valid=true;
		unrolled[linear].ieta=hid.ieta();
		unrolled[linear].iphi=hid.iphi();
	      }
	    } else {
	      report.countUnmappedTPDigi(eid);
	    }
	  }
	  if (unrolled[linear].valid) {
	    if (isTPGSOI(*tp_work)) unrolled[linear].soi=unrolled[linear].samples;
	    if (tp_work->raw()&(1<<ibit)) unrolled[linear].setbit(unrolled[linear].samples);
	    unrolled[linear].samples++;
	  }
	}
      }
      for (int i=0; i<24; i++) {
	if (unrolled[i].valid) 
	  colls.tphoCont->push_back(HOTriggerPrimitiveDigi(
							   unrolled[i].ieta,
							   unrolled[i].iphi,
							   unrolled[i].samples,
							   unrolled[i].soi,
							   unrolled[i].databits));
      }
    } else { // regular TPs (not HO)
      for (tp_work=tp_begin; tp_work!=tp_end; tp_work++) {
	if (tp_work->raw()==0xFFFF) continue; // filler word
	if (tp_work->slbAndChan()!=currFiberChan) { // start new set
	  npre=0;
	  currFiberChan=tp_work->slbAndChan();
	  // lookup the right channel
	  HcalElectronicsId eid(tp_work->slbChan(),tp_work->slb(),spigot,dccid,htr_cr,htr_slot,htr_tb);
	  DetId did=emap.lookupTrigger(eid);
	  if (did.null()) {
	    report.countUnmappedTPDigi(eid);
	    if (unknownIdsTrig_.find(eid)==unknownIdsTrig_.end()) {
	      if (!silent) edm::LogWarning("HCAL") << "HcalUnpacker: No trigger primitive match found for electronics id :" << eid;
	      unknownIdsTrig_.insert(eid);
	    }
	    valid=false;
	    continue;
	  } else if (did==HcalTrigTowerDetId::Undefined || 
		     (did.det()==DetId::Hcal && did.subdetId()==0)) {
	    // known to be unmapped
	    valid=false;
	    continue;
	  }
	  HcalTrigTowerDetId id(did);
	  colls.tpCont->push_back(HcalTriggerPrimitiveDigi(id));
	  // set the various bits
	  if (!tpgSOIbitInUse) colls.tpCont->back().setPresamples(nps);
	  colls.tpCont->back().setZSInfo(htr.isUnsuppressed(),htr.wasMarkAndPassZSTP(tp_work->slb(),tp_work->slbChan()));

	  // no hits recorded for current
	  ncurr=0;
	  valid=true;
	}      
	// add the word (if within settings or recent firmware [recent firmware ignores startSample/endSample])
	if (valid && ((tpgSOIbitInUse && ncurr<10) || (ncurr>=startSample_ && ncurr<=endSample_))) {
	  colls.tpCont->back().setSample(colls.tpCont->back().size(),*tp_work);
	  colls.tpCont->back().setSize(colls.tpCont->back().size()+1);
	}
	// set presamples,if SOI
	if (valid && tpgSOIbitInUse && isTPGSOI(*tp_work)) {
	  colls.tpCont->back().setPresamples(ncurr);
	}
	ncurr++;
	npre++;
      }
    }

    /// branch point between 2006-2011 data format and 2012+ data format
    if (htr.getFormatVersion() < HcalHTRData::FORMAT_VERSION_COMPACT_DATA) {
 
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
	HcalElectronicsId eid(qie_work->fiberChan(),qie_work->fiber(),spigot,dccid);
	eid.setHTR(htr_cr,htr_slot,htr_tb);
	DetId did=emap.lookup(eid);
	
	if (!did.null()) {
	  if (did.det()==DetId::Calo && did.subdetId()==HcalZDCDetId::SubdetectorId) {
	    colls.zdcCont->push_back(ZDCDataFrame(HcalZDCDetId(did)));
	    qie_work=HcalUnpacker_impl::unpack<ZDCDataFrame>(qie_work, qie_end, colls.zdcCont->back(), nps, eid, startSample_, endSample_, expectedOrbitMessageTime_, htr); 
	  } else if (did.det()==DetId::Hcal) {
	    switch (((HcalSubdetector)did.subdetId())) {
	    case (HcalBarrel):
	    case (HcalEndcap): {
	      colls.hbheCont->push_back(HBHEDataFrame(HcalDetId(did)));
	      qie_work=HcalUnpacker_impl::unpack<HBHEDataFrame>(qie_work, qie_end, colls.hbheCont->back(), nps, eid, startSample_, endSample_, expectedOrbitMessageTime_, htr);
	    } break;
	    case (HcalOuter): {
	      colls.hoCont->push_back(HODataFrame(HcalDetId(did)));
	      qie_work=HcalUnpacker_impl::unpack<HODataFrame>(qie_work, qie_end, colls.hoCont->back(), nps, eid, startSample_, endSample_, expectedOrbitMessageTime_, htr);
	  } break;
	    case (HcalForward): {
	      colls.hfCont->push_back(HFDataFrame(HcalDetId(did)));
	      qie_work=HcalUnpacker_impl::unpack<HFDataFrame>(qie_work, qie_end, colls.hfCont->back(), nps, eid, startSample_, endSample_, expectedOrbitMessageTime_, htr);
	    } break;
	    case (HcalOther) : {
	      HcalOtherDetId odid(did);
	      if (odid.subdet()==HcalCalibration) {
		colls.calibCont->push_back(HcalCalibDataFrame(HcalCalibDetId(did)));
		qie_work=HcalUnpacker_impl::unpack<HcalCalibDataFrame>(qie_work, qie_end, colls.calibCont->back(), nps, eid, startSample_, endSample_, expectedOrbitMessageTime_, htr); 
	      }
	    } break;
	    case (HcalEmpty): 
	    default: {
	      for (int fiberC=qie_work->fiberAndChan();
		   qie_work!=qie_end && qie_work->fiberAndChan()==fiberC;
		   qie_work++);
	    }
	    break;
	    }
	  }
	} else {
	  report.countUnmappedDigi(eid);
	  if (unknownIds_.find(eid)==unknownIds_.end()) {
	    if (!silent) edm::LogWarning("HCAL") << "HcalUnpacker: No match found for electronics id :" << eid;
	    unknownIds_.insert(eid);
	}
	  for (int fiberC=qie_work->fiberAndChan();
	       qie_work!=qie_end && qie_work->fiberAndChan()==fiberC;
	       qie_work++);
	}
      }
    } else {
      // this is the branch for unpacking the compact data format with per-channel headers
      const unsigned short* ptr_header=daq_first;
      const unsigned short* ptr_end=daq_last+1;
      int flavor, error_flags, capid0, channelid;

      while (ptr_header!=ptr_end) {
	if (*ptr_header==0xFFFF) { // impossible filler word
	  ptr_header++;
	  continue;
	}
	// unpack the header word
	bool isheader=HcalHTRData::unpack_per_channel_header(*ptr_header,flavor,error_flags,capid0,channelid);
	if (!isheader) {
	  ptr_header++;
	  continue;
	}

	int fiberchan=channelid&0x3;
	int fiber=((channelid>>2)&0x7)+1;

	// lookup the right channel
	HcalElectronicsId eid(fiberchan,fiber,spigot,dccid);
	eid.setHTR(htr_cr,htr_slot,htr_tb);
	DetId did=emap.lookup(eid);
	
	if (!did.null()) {
	  if (did.det()==DetId::Calo && did.subdetId()==HcalZDCDetId::SubdetectorId) {
	    colls.zdcCont->push_back(ZDCDataFrame(HcalZDCDetId(did)));
	    ptr_header=HcalUnpacker_impl::unpack_compact<ZDCDataFrame>(ptr_header, ptr_end, colls.zdcCont->back(), nps, eid, startSample_, endSample_, expectedOrbitMessageTime_, htr); 
	  } else if (did.det()==DetId::Hcal) {
	    switch (((HcalSubdetector)did.subdetId())) {
	    case (HcalBarrel):
	    case (HcalEndcap): {
	      colls.hbheCont->push_back(HBHEDataFrame(HcalDetId(did)));
	      ptr_header=HcalUnpacker_impl::unpack_compact<HBHEDataFrame>(ptr_header, ptr_end, colls.hbheCont->back(), nps, eid, startSample_, endSample_, expectedOrbitMessageTime_, htr);
	    } break;
	    case (HcalOuter): {
	      colls.hoCont->push_back(HODataFrame(HcalDetId(did)));
	      ptr_header=HcalUnpacker_impl::unpack_compact<HODataFrame>(ptr_header, ptr_end, colls.hoCont->back(), nps, eid, startSample_, endSample_, expectedOrbitMessageTime_, htr);
	  } break;
	    case (HcalForward): {
	      colls.hfCont->push_back(HFDataFrame(HcalDetId(did)));
	      ptr_header=HcalUnpacker_impl::unpack_compact<HFDataFrame>(ptr_header, ptr_end, colls.hfCont->back(), nps, eid, startSample_, endSample_, expectedOrbitMessageTime_, htr);
	    } break;
	    case (HcalOther) : {
	      HcalOtherDetId odid(did);
	      if (odid.subdet()==HcalCalibration) {
		colls.calibCont->push_back(HcalCalibDataFrame(HcalCalibDetId(did)));
		ptr_header=HcalUnpacker_impl::unpack_compact<HcalCalibDataFrame>(ptr_header, ptr_end, colls.calibCont->back(), nps, eid, startSample_, endSample_, expectedOrbitMessageTime_, htr); 
	      }
	    } break;
	    case (HcalEmpty): 
	    default: {
	      for (ptr_header++;
		   ptr_header!=ptr_end && !HcalHTRData::is_channel_header(*ptr_header);
		   ptr_header++);
	    }
	    break;
	    }
	  }
	} else {
	  report.countUnmappedDigi(eid);
	  if (unknownIds_.find(eid)==unknownIds_.end()) {
	    if (!silent) edm::LogWarning("HCAL") << "HcalUnpacker: No match found for electronics id :" << eid;
	    unknownIds_.insert(eid);
	  }
	  for (ptr_header++;
	       ptr_header!=ptr_end && !HcalHTRData::is_channel_header(*ptr_header);
	       ptr_header++);
	}
      }

    }
  }
}

HcalUnpacker::Collections::Collections() {
  hbheCont=0;
  hoCont=0;
  hfCont=0;
  tpCont=0;
  zdcCont=0;
  calibCont=0;
  ttp=0;
}

void HcalUnpacker::unpack(const FEDRawData& raw, const HcalElectronicsMap& emap, std::vector<HBHEDataFrame>& container, std::vector<HcalTriggerPrimitiveDigi>& tp) {
  Collections c;
  c.hbheCont=&container;
  c.tpCont=&tp;
  HcalUnpackerReport r;
  unpack(raw,emap,c,r);
}

void HcalUnpacker::unpack(const FEDRawData& raw, const HcalElectronicsMap& emap, std::vector<HODataFrame>& container, std::vector<HcalTriggerPrimitiveDigi>& tp) {
  Collections c;
  c.hoCont=&container;
  c.tpCont=&tp;
  HcalUnpackerReport r;
  unpack(raw,emap,c,r);
}

void HcalUnpacker::unpack(const FEDRawData& raw, const HcalElectronicsMap& emap, std::vector<HFDataFrame>& container, std::vector<HcalTriggerPrimitiveDigi>& tp) {
  Collections c;
  c.hfCont=&container;
  c.tpCont=&tp;
  HcalUnpackerReport r;
  unpack(raw,emap,c,r);
}

void HcalUnpacker::unpack(const FEDRawData& raw, const HcalElectronicsMap& emap, std::vector<HcalHistogramDigi>& histoDigis) {

  // get the DCC header
  const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
  int dccid=dccHeader->getSourceId()-sourceIdOffset_;
  
  // check the summary status
  
  // walk through the HTR data...
  HcalHTRData htr;
  for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {
    if (!dccHeader->getSpigotPresent(spigot)) continue;
    
    int retval=dccHeader->getSpigotData(spigot,htr,raw.size());
    // check
    if (retval || !htr.check()) {
      edm::LogWarning("Invalid Data") << "Invalid HTR data observed on spigot " << spigot << " of DCC with source id " << dccHeader->getSourceId();
      continue;
    }
    if (!htr.isHistogramEvent()) {
      edm::LogWarning("Invalid Data") << "Non-histogram data passed to histogram unpacker on spigot " << spigot << " of DCC with source id " << dccHeader->getSourceId();
      continue;
    }

    unsigned int smid=htr.getSubmodule();
    int htr_tb=smid&0x1;
    int htr_slot=(smid>>1)&0x1F;
    int htr_cr=(smid>>6)&0x1F;
    
    // find out the fibers
    int f[2],fc;
    htr.getHistogramFibers(f[0],f[1]);

    for (int nf=0; nf<2; nf++) {
      if (f[nf]<0 || (nf==1 && f[0]==f[1])) continue; // skip if invalid or the same
      for (fc=0; fc<=2; fc++) {
	HcalElectronicsId eid(fc,f[nf],spigot,dccid);	  
	eid.setHTR(htr_cr,htr_slot,htr_tb);
	DetId did=emap.lookup(eid);

	if (did.null() || did.det()!=DetId::Hcal || did.subdetId()==0) {
	  if (unknownIds_.find(eid)==unknownIds_.end()) {
	    edm::LogWarning("HCAL") << "HcalHistogramUnpacker: No match found for electronics id :" << eid;
	    unknownIds_.insert(eid);
	  }	  
	  continue;
	}
	histoDigis.push_back(HcalHistogramDigi(HcalDetId(did))); // add it!
	HcalHistogramDigi& digi=histoDigis.back();
	
	// unpack the four capids
	for (int capid=0; capid<4; capid++) 
	  htr.unpackHistogram(f[nf],fc,capid,digi.getArray(capid));
	
      }
    }
  }
}      

