#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDTCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/AMC13Header.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUHTRData.h"
#include "DataFormats/HcalDetId/interface/HcalOtherDetId.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/HcalRawToDigi/interface/HcalTTPUnpacker.h"

//#define DebugLog

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
#ifdef DebugLog
      std::cout << hhd.getFibOrbMsgBCN(fiber) << " " << expectedTime << std::endl;
#endif
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
#ifdef DebugLog
      std::cout << hhd.getFibOrbMsgBCN(fiber) << " " << expectedTime << std::endl;
#endif
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
    } else {
      edm::LogWarning("Bad Data") << "Invalid flavor " << flavor;
      qie_work=limit;
    }
    return qie_work;
  }
  
  template <class DigiClass>
  void unpack_compact(HcalUHTRData::const_iterator& i, const HcalUHTRData::const_iterator& iend, DigiClass& digi, 
		      int presamples, const HcalElectronicsId& eid, int startSample, int endSample) {
    // set parameters
    digi.setPresamples(presamples-startSample);
    digi.setReadoutIds(eid);
    int error_flags=i.errFlags();
    int capid0=i.capid0();

    bool isCapRotating=!(error_flags&0x1);
    bool fiberErr=(error_flags&0x2);
    bool dataValid=!(error_flags&0x2);
    int fiberchan=i.channelid()&0x3;
    int fiber=((i.channelid()>>2)&0x7)+1;

    //    digi.setZSInfo(hhd.isUnsuppressed(),hhd.wasMarkAndPassZS(fiber,fiberchan),zsmask);

    // what is my sample number?
    int ncurr=0,ntaken=0;
    for (++i; i!=iend && !i.isHeader(); ++i) {
      int capidn=(isCapRotating)?((capid0+ncurr)%4):(capid0);
      
      HcalQIESample s(i.adc(),capidn,fiber,fiberchan,dataValid,fiberErr);
      
      if (ncurr>=startSample && ncurr<=endSample) {
	digi.setSample(ntaken,s);
	++ntaken;
      }
      ncurr++;
    }
    digi.setSize(ntaken);
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
    if (!silent) edm::LogWarning("Invalid Data") << "Empty/invalid data, size = " << raw.size();
    return;
  }

  // get the DCC header
  const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());

  if (dccHeader->BOEshouldBeZeroAlways()==0) // also includes uTCA before the common AMC13XG firmware
    unpackVME(raw,emap,colls,report,silent);
  else unpackUTCA(raw,emap,colls,report,silent);
}

static int slb(uint16_t theSample) { return ((theSample>>13)&0x7); }
static int slbChan(uint16_t theSample) { return (theSample>>11)&0x3; }
static int slbAndChan(uint16_t theSample) { return (theSample>>11)&0x1F; }


void HcalUnpacker::unpackVME(const FEDRawData& raw, const HcalElectronicsMap& emap,
			  Collections& colls, HcalUnpackerReport& report, bool silent) {

  // get the DCC header
  const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
  const HcalDTCHeader* dtcHeader=(const HcalDTCHeader*)(raw.data());
  bool is_VME_DCC=(dccHeader->getDCCDataFormatVersion()<0x10) || ((mode_&0x1)==0);
  
  int dccid=(is_VME_DCC)?(dccHeader->getSourceId()-sourceIdOffset_):(dtcHeader->getSourceId()-sourceIdOffset_);

  // check the summary status
  
  // walk through the HTR data.  For the uTCA, use spigot=slot+1
  HcalHTRData htr;
  const unsigned short* daq_first, *daq_last, *tp_first, *tp_last;
  const HcalQIESample* qie_begin, *qie_end, *qie_work;
  const HcalTriggerPrimitiveSample *tp_begin, *tp_end, *tp_work; 
  for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {
    
    if (is_VME_DCC) {
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
    } else { // is_uTCA (!is_VME_DCC)
      int slot=spigot+1;
      if (slot>HcalDTCHeader::MAXIMUM_SLOT) continue;

      if (!dtcHeader->getSlotPresent(slot)) continue;

      int retval=dtcHeader->getSlotData(slot,htr,raw.size());
      if (retval!=0) {
	if (retval==-1) {
	  if (!silent) edm::LogWarning("Invalid Data") << "Invalid uHTR data (data beyond payload size) observed on slot " << slot << " of DTC with source id " << dtcHeader->getSourceId();
	  report.countSpigotFormatError();
	}
	continue;
      }
      // check
      if (dtcHeader->getSlotCRCError(slot)) {
	if (!silent) 
	  edm::LogWarning("Invalid Data") << "CRC Error on uHTR data observed on slot " << slot << " of DTC with source id " << dtcHeader->getSourceId();
	report.countSpigotFormatError();
	continue;
      } 
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
	int sector=slbChan(tp_work->raw());
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
	if (slbAndChan(tp_work->raw())!=currFiberChan) { // start new set
	  npre=0;
	  currFiberChan=slbAndChan(tp_work->raw());
	  // lookup the right channel
	  HcalElectronicsId eid(slbChan(tp_work->raw()),slb(tp_work->raw()),spigot,dccid,htr_cr,htr_slot,htr_tb);
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
	  colls.tpCont->back().setZSInfo(htr.isUnsuppressed(),htr.wasMarkAndPassZSTP(slb(tp_work->raw()),slbChan(tp_work->raw())));

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

void HcalUnpacker::unpackUTCA(const FEDRawData& raw, const HcalElectronicsMap& emap,
			      Collections& colls, HcalUnpackerReport& report, bool silent) {

  const hcal::AMC13Header* amc13=(const hcal::AMC13Header*)(raw.data());

  // how many AMC in this packet
  int namc=amc13->NAMC();
  for (int iamc=0; iamc<namc; iamc++) {
    // if not enabled, ignore
    if (!amc13->AMCEnabled(iamc)) continue; 

    if (!amc13->AMCDataPresent(iamc)) {
      if (!silent) 
	edm::LogWarning("Invalid Data") << "Missing data observed on iamc " << iamc << " of AMC13 with source id " << amc13->sourceId();
      report.countSpigotFormatError();
      continue;     
    }
    if (!amc13->AMCCRCOk(iamc)) {
      if (!silent) 
	edm::LogWarning("Invalid Data") << "CRC Error on uHTR data observed on iamc " << iamc << " of AMC13 with source id " << amc13->sourceId();
      report.countSpigotFormatError();
      //      continue;
    }
    // this unpacker cannot handle segmented data!
    if (amc13->AMCSegmented(iamc)) {
      if (!silent) 
	edm::LogWarning("Invalid Data") << "Unpacker cannot handle segmented data observed on iamc " << iamc << " of AMC13 with source id " << amc13->sourceId();
      report.countSpigotFormatError();
      continue;
    }
    
    // ok, now we're work-able
    int slot=amc13->AMCSlot(iamc);
    int crate=amc13->AMCId(iamc)&0xFF;
    // this is used only for the 1.6 Gbps link data
    int nps=(amc13->AMCId(iamc)>>12)&0xF;
    
    HcalUHTRData uhtr(amc13->AMCPayload(iamc),amc13->AMCSize(iamc));
    //Check to make sure uMNio is not unpacked here
    if(uhtr.getFormatVersion() != 1) continue;
#ifdef DebugLog
    //debug printouts
    int nwords=uhtr.getRawLengthBytes()/2;
    for (int iw=0; iw<nwords; iw++) 
        printf("%04d %04x\n",iw,uhtr.getRawData16()[iw]);
#endif

    HcalUHTRData::const_iterator i=uhtr.begin(), iend=uhtr.end();
    while (i!=iend) {
#ifdef DebugLog
      std::cout << "This data is flavored:" << i.flavor() << std::endl;
#endif

      if (!i.isHeader()) {
	++i;
#ifdef DebugLog
	std::cout << "its not a header" << std::endl;
#endif
	continue;
      }
      ///////////////////////////////////////////////HE UNPACKER//////////////////////////////////////////////////////////////////////////////////////
      if (i.flavor() == 1 || i.flavor() == 0) {
          int ifiber=((i.channelid()>>3)&0x1F);
          int ichan=(i.channelid()&0x7);
          HcalElectronicsId eid(crate,slot,ifiber,ichan, false);
          DetId did=emap.lookup(eid);
          // Count from current position to next header, or equal to end
          const uint16_t* head_pos = i.raw();
          int ns = 0;
          for (++i; i != iend && !i.isHeader(); ++i) {
              ns++;
          }
          // Check QEI11 container exists
          if (colls.qie11 == 0) {
              colls.qie11 = new QIE11DigiCollection(ns);
          }
          else if (colls.qie11->samples() != ns) {
              // This is horrible
              edm::LogError("Invalid Data") << "Collection has " << colls.qie11->samples() << " samples per digi, raw data has " << ns << "!";
              return;
          }

          // Insert data
          /////////////////////////////////////////////CODE FROM OLD STYLE DIGIS///////////////////////////////////////////////////////////////
          if (!did.null()) { // unpack and store...
              colls.qie11->addDataFrame(did, head_pos);
          } else {
              report.countUnmappedDigi(eid);
              if (unknownIds_.find(eid)==unknownIds_.end()) {
                  if (!silent) edm::LogWarning("HCAL") << "HcalUnpacker: No match found for electronics id :" << eid;
                  unknownIds_.insert(eid);
#ifdef DebugLog
                  std::cout << "HcalUnpacker: No match found for electronics id :" << eid << std::endl;
#endif
              }
#ifdef DebugLog
              std::cout << "OH NO! detector id is null!" << std::endl;
#endif
          }
      }

      //////////////////////////////////////////////////HF UNPACKER/////////////////////////////////////////////////////////////////////
      if (i.flavor() == 2) {
	int ifiber=((i.channelid()>>3)&0x1F);
	int ichan=(i.channelid()&0x7);
	HcalElectronicsId eid(crate,slot,ifiber,ichan, false);
	DetId did=emap.lookup(eid);

	// Count from current position to next header, or equal to end
	const uint16_t* head_pos = i.raw();
	int ns = 0;
	for (++i; i != iend && !i.isHeader(); ++i) {
	  ns++; 
	}

	// Check QEI10 container exists
	if (colls.qie10 == 0) {
	  colls.qie10 = new QIE10DigiCollection(ns);
	}
	else if (colls.qie10->samples() != ns) {
	  // This is horrible
	  edm::LogError("Invalid Data") << "Collection has " << colls.qie10->samples() << " samples per digi, raw data has " << ns << "!";
	  return;
	}

	// Insert data
    /////////////////////////////////////////////CODE FROM OLD STYLE DIGIS///////////////////////////////////////////////////////////////
	if (!did.null()) { // unpack and store...
		colls.qie10->addDataFrame(did, head_pos);
	} else {
		report.countUnmappedDigi(eid);
		if (unknownIds_.find(eid)==unknownIds_.end()) {
			if (!silent) edm::LogWarning("HCAL") << "HcalUnpacker: No match found for electronics id :" << eid;
			unknownIds_.insert(eid);
#ifdef DebugLog
			std::cout << "HcalUnpacker: No match found for electronics id :" << eid << std::endl;
#endif
		}
#ifdef DebugLog
		std::cout << "OH NO! HcalUnpacker: No match found for electronics id :" << eid << std::endl;
#endif
	}
      }
      else if (i.flavor()==0x5) { // Old-style digis
	int ifiber=((i.channelid()>>2)&0x1F);
	int ichan=(i.channelid()&0x3);
	HcalElectronicsId eid(crate,slot,ifiber,ichan, false);
	DetId did=emap.lookup(eid);
	
	if (!did.null()) { // unpack and store...
	  if (did.det()==DetId::Calo && did.subdetId()==HcalZDCDetId::SubdetectorId) {
	    colls.zdcCont->push_back(ZDCDataFrame(HcalZDCDetId(did)));
	    HcalUnpacker_impl::unpack_compact<ZDCDataFrame>(i,iend, colls.zdcCont->back(), nps, eid, startSample_, endSample_); 
	  } else if (did.det()==DetId::Hcal) {
	    switch (((HcalSubdetector)did.subdetId())) {
	    case (HcalBarrel):
	    case (HcalEndcap): {
	      colls.hbheCont->push_back(HBHEDataFrame(HcalDetId(did)));
	      HcalUnpacker_impl::unpack_compact<HBHEDataFrame>(i, iend, colls.hbheCont->back(), nps, eid, startSample_, endSample_);
	    } break;
	    case (HcalOuter): {
	      colls.hoCont->push_back(HODataFrame(HcalDetId(did)));
	      HcalUnpacker_impl::unpack_compact<HODataFrame>(i, iend, colls.hoCont->back(), nps, eid, startSample_, endSample_);
	    } break;
	    case (HcalForward): {
	      colls.hfCont->push_back(HFDataFrame(HcalDetId(did)));
	      HcalUnpacker_impl::unpack_compact<HFDataFrame>(i, iend, colls.hfCont->back(), nps, eid, startSample_, endSample_);
	    } break;
	    case (HcalOther) : {
	      HcalOtherDetId odid(did);
	      if (odid.subdet()==HcalCalibration) {
		colls.calibCont->push_back(HcalCalibDataFrame(HcalCalibDetId(did)));
		HcalUnpacker_impl::unpack_compact<HcalCalibDataFrame>(i, iend, colls.calibCont->back(), nps, eid, startSample_, endSample_); 
	      }
	    } break;
	    case (HcalEmpty): 
	    default: {
	      for (++i;
		   i!=iend && !i.isHeader();
		   ++i);
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
	  for (++i;
	       i!=iend && !i.isHeader();
	       ++i);
	}
      } else if (i.flavor()==0x4) { // TP digis
	int ilink=((i.channelid()>>4)&0xF);
	int itower=(i.channelid()&0xF);
	HcalElectronicsId eid(crate,slot,ilink,itower,true);
	DetId did=emap.lookupTrigger(eid);
#ifdef DebugLog
	std::cout << "Unpacking " << eid << " " << i.channelid() << std::endl;
#endif
	if (did.null()) {
	  report.countUnmappedTPDigi(eid);
	  if (unknownIdsTrig_.find(eid)==unknownIdsTrig_.end()) {
	    if (!silent) edm::LogWarning("HCAL") << "HcalUnpacker: No trigger primitive match found for electronics id :" << eid;
	    unknownIdsTrig_.insert(eid);
	  }
	  // Skip it
	  for (++i; i!=iend && !i.isHeader(); ++i);	
	} else if (did==HcalTrigTowerDetId::Undefined || 
		   (did.det()==DetId::Hcal && did.subdetId()==0)) {
	  for (++i; i!=iend && !i.isHeader(); ++i);	
	} else {
	  HcalTrigTowerDetId id(did);
#ifdef DebugLog
	  std::cout << "Unpacking " << id << std::endl;
#endif
	  colls.tpCont->push_back(HcalTriggerPrimitiveDigi(id));
	  int j=0;
	  for (++i; i!=iend && !i.isHeader(); ++i) {
	    colls.tpCont->back().setSample(j,i.value());
	    if (i.soi()) colls.tpCont->back().setPresamples(j);
	    j++;
	  }
	  colls.tpCont->back().setSize(j);
	}      
      } else {
	// consume any not-understood channel data
	  for (++i;
	       i!=iend && !i.isHeader();
	       ++i);	
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
  qie10=0;
  qie11=0;
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

