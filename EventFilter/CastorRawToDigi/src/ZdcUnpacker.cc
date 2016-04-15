#include "EventFilter/CastorRawToDigi/interface/ZdcUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
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
	int dccid=dccHeader->getSourceId()-sourceIdOffset_;

	// walk through the HTR data...
	HcalHTRData htr;
	const unsigned short* daq_first, *daq_last, *tp_first, *tp_last;
	std::map<HcalElectronicsId,DetId> myEMap;


	//////ZDC MAP for NEW data (2015 PbPb are newer)
	//PZDC
	HcalElectronicsId eid = HcalElectronicsId(0, 1, 0, 3);
	eid.setHTR(18, 8, 1);
	myEMap[eid]=DetId(0x54000051);//PZDC EM1

	eid = HcalElectronicsId(1, 1, 0, 3);
	eid.setHTR(18, 8, 1);
	myEMap[eid]=DetId(0x54000052);//PZDC EM2

	eid = HcalElectronicsId(2, 1, 0, 3);
	eid.setHTR(18, 8, 1);
	myEMap[eid]=DetId(0x54000053);//PZDC EM3

	eid = HcalElectronicsId(0, 2, 0, 3);
	eid.setHTR(18, 8, 1);
	myEMap[eid]=DetId(0x54000061);//PZDC HAD1

	eid = HcalElectronicsId(1, 2, 0, 3);
	eid.setHTR(18, 8, 1);
	myEMap[eid]=DetId(0x54000054);//PZDC EM4

	eid = HcalElectronicsId(2, 2, 0, 3);
	eid.setHTR(18, 8, 1);
	myEMap[eid]=DetId(0x54000055);//PZDC EM5

	eid = HcalElectronicsId(0, 3, 0, 3);
	eid.setHTR(18, 8, 1);
	myEMap[eid]=DetId(0x54000062);//PZDC HAD2

	eid = HcalElectronicsId(1, 3, 0, 3);
	eid.setHTR(18, 8, 1);
	myEMap[eid]=DetId(0x54000063);//PZDC HAD3

	eid = HcalElectronicsId(2, 3, 0, 3);
	eid.setHTR(18, 8, 1);
	myEMap[eid]=DetId(0x54000064);//PZDC HAD4

	//NZDC
	eid = HcalElectronicsId(0,1,1,3);
	eid.setHTR(18, 8, 0);
	myEMap[eid]=DetId(0x54000011);//NZDC EM1

	eid = HcalElectronicsId(1,1,1,3);
	eid.setHTR(18, 8, 0);
	myEMap[eid]=DetId(0x54000012);//NZDC EM2

	eid = HcalElectronicsId(2,1,1,3);
	eid.setHTR(18, 8, 0);
	myEMap[eid]=DetId(0x54000013);//NZDC EM3

	eid = HcalElectronicsId(0,2,1,3);
	eid.setHTR(18, 8, 0);
	myEMap[eid]=DetId(0x54000015);//NZDC EM5

	eid = HcalElectronicsId(1,2,1,3);
	eid.setHTR(18, 8, 0);
	myEMap[eid]=DetId(0x54000021);//NZDC HAD1

	eid = HcalElectronicsId(2,2,1,3);
	eid.setHTR(18, 8, 0);
	myEMap[eid]=DetId(0x54000014);//NZDC EM4

	eid = HcalElectronicsId(0,3,1,3);
	eid.setHTR(18, 8, 0);
	myEMap[eid]=DetId(0x54000022);//NZDC HAD2

	eid = HcalElectronicsId(1,3,1,3);
	eid.setHTR(18, 8, 0);
	myEMap[eid]=DetId(0x54000023);//NZDC HAD3

	eid = HcalElectronicsId(2,3,1,3);
	eid.setHTR(18, 8, 0);
	myEMap[eid]=DetId(0x54000024);//NZDC HAD4

	for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++)
	{
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
		int nps=htr.getNPS()-startSample_;

		// get pointers
		htr.dataPointers(&daq_first,&daq_last,&tp_first,&tp_last);
		unsigned int smid=htr.getSubmodule();
		int htr_tb=smid&0x1;
		int htr_slot=(smid>>1)&0x1F;
		int htr_cr=(smid>>6)&0x1F;


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
			auto it = myEMap.find(eid);
			DetId did;
			if (it != myEMap.end())
			{
				did = it->second;
			}

			if (!did.null()) {
				if (did.det()==DetId::Calo && did.subdetId()==HcalZDCDetId::SubdetectorId) {
					colls.zdcCont->push_back(ZDCDataFrame(HcalZDCDetId(did)));
					ptr_header=ZdcUnpacker_impl::unpack_compact<ZDCDataFrame>(ptr_header, ptr_end, colls.zdcCont->back(), nps, eid, startSample_, endSample_, expectedOrbitMessageTime_, htr);
				}
			} else {
				report.countUnmappedDigi(eid);
				for (ptr_header++;
					ptr_header!=ptr_end && !HcalHTRData::is_channel_header(*ptr_header);
					ptr_header++);
			}

		}
	}//end of loop over spigots
}


