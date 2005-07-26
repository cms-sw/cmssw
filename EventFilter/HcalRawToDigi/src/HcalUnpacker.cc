#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include <iostream>

namespace HcalUnpacker_impl {
  template <class HcalDataFrame> 
  void unpack(const raw::FEDRawData& raw, const cms::hcal::HcalMapping& emap, std::vector<HcalDataFrame>& cont, std::vector<cms::HcalTriggerPrimitiveDigi>& tpcont, const int startSample, const int endSample, const int sourceIdOffset) {

    // get the DCC header
    const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
    int dccid=dccHeader->getSourceId()-sourceIdOffset;

    // check the summary status

    // walk through the HTR data...
    HcalHTRData htr;
    const unsigned short* daq_first, *daq_last, *tp_first, *tp_last;
    const cms::HcalQIESample* qie_begin, *qie_end, *qie_work;
    const cms::HcalTriggerPrimitiveSample *tp_begin, *tp_end, *tp_work; 
    for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {
      if (!dccHeader->getSpigotPresent(spigot)) continue;

      dccHeader->getSpigotData(spigot,htr);
      // check
      if (!htr.check()) {
	// TODO: log error!
	continue;
      }
      // calculate "real" number of presamples
      int nps=htr.getNPS()-startSample;
      
      // get pointers
      htr.dataPointers(&daq_first,&daq_last,&tp_first,&tp_last);

      tp_begin=(cms::HcalTriggerPrimitiveSample*)tp_first;
      tp_end=(cms::HcalTriggerPrimitiveSample*)(tp_last+1); // one beyond last..

      /// work through the samples
      int currFiberChan=0x3F; // invalid fiber+channel...
      int ncurr=0;
      bool valid=false;

      for (tp_work=tp_begin; tp_work!=tp_end; tp_work++) {
	if (tp_work->raw()==0xFFFF) continue; // filler word
	if (tp_work->fiberAndChan()!=currFiberChan) { // start new set
	  currFiberChan=tp_work->fiberAndChan();
	  // lookup the right channel
	  cms::HcalElectronicsId eid(tp_work->fiberChan(),tp_work->fiber(),spigot,dccid);
	  cms::HcalTrigTowerDetId id=emap.lookupTrigger(eid);
	  if (id.null()) {
	    //	    std::cerr << "No match found for " << eid << std::endl;
	    valid=false;
	    continue;
	  } 
	  tpcont.push_back(cms::HcalTriggerPrimitiveDigi(id));
	  // set the various bits
	  tpcont.back().setPresamples(nps);
	  // no hits recorded for current
	  ncurr=0;
	  valid=true;
	}
	// add the word (if within settings) [ TODO: correct behavior when just one TP... ]
	if (valid && ncurr>=startSample && ncurr<=endSample) {
	  tpcont.back().setSample(tpcont.back().size(),*tp_work);
	  tpcont.back().setSize(tpcont.back().size()+1);
	}
	ncurr++;
      }


      qie_begin=(cms::HcalQIESample*)daq_first;
      qie_end=(cms::HcalQIESample*)(daq_last+1); // one beyond last..

      /// work through the samples
      currFiberChan=0x3F; // invalid fiber+channel...
      ncurr=0;
      valid=false;

      for (qie_work=qie_begin; qie_work!=qie_end; qie_work++) {
	if (qie_work->raw()==0xFFFF) continue; // filler word
	if (qie_work->fiberAndChan()!=currFiberChan) { // start new set
	  currFiberChan=qie_work->fiberAndChan();
	  // lookup the right channel
	  cms::HcalElectronicsId eid(qie_work->fiberChan(),qie_work->fiber(),spigot,dccid);
	  cms::HcalDetId id=emap.lookup(eid);
	  if (id.null()) {
	    //	    std::cerr << "No match found for " << eid << std::endl;
	    valid=false;
	    continue;
	  } 
	  cont.push_back(HcalDataFrame(id));
	  // set the various bits
	  cont.back().setPresamples(nps);
	  cont.back().setReadoutIds(eid);
	  // no hits recorded for current
	  ncurr=0;
	  valid=true;
	}
	// add the word (if within settings
	if (valid && ncurr>=startSample && ncurr<=endSample) {
	  cont.back().setSample(cont.back().size(),*qie_work);
	  cont.back().setSize(cont.back().size()+1);
	}
	ncurr++;
      }
    }
  }
}

void HcalUnpacker::unpack(const raw::FEDRawData& raw, const cms::hcal::HcalMapping& emap, std::vector<cms::HBHEDataFrame>& container, std::vector<cms::HcalTriggerPrimitiveDigi>& tp) {
  HcalUnpacker_impl::unpack<cms::HBHEDataFrame>(raw,emap,container,tp,startSample_,endSample_, sourceIdOffset_);
}

void HcalUnpacker::unpack(const raw::FEDRawData& raw, const cms::hcal::HcalMapping& emap, std::vector<cms::HODataFrame>& container, std::vector<cms::HcalTriggerPrimitiveDigi>& tp) {
  HcalUnpacker_impl::unpack<cms::HODataFrame>(raw,emap,container,tp,startSample_,endSample_, sourceIdOffset_);
}

void HcalUnpacker::unpack(const raw::FEDRawData& raw, const cms::hcal::HcalMapping& emap, std::vector<cms::HFDataFrame>& container, std::vector<cms::HcalTriggerPrimitiveDigi>& tp) {
  HcalUnpacker_impl::unpack<cms::HFDataFrame>(raw,emap,container,tp,startSample_,endSample_, sourceIdOffset_);
}
