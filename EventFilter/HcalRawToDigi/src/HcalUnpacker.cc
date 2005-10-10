#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include <iostream>

namespace HcalUnpacker_impl {
  template <class HcalDataFrame> 
  void unpack(const FEDRawData& raw, const HcalMapping& emap, std::vector<HcalDataFrame>& cont, std::vector<HcalTriggerPrimitiveDigi>& tpcont, const int startSample, const int endSample, const int sourceIdOffset, HcalSubdetector sd1, HcalSubdetector sd2) {

    // get the DCC header
    const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
    int dccid=dccHeader->getSourceId()-sourceIdOffset;

    // check the summary status

    // walk through the HTR data...
    HcalHTRData htr;
    const unsigned short* daq_first, *daq_last, *tp_first, *tp_last;
    const HcalQIESample* qie_begin, *qie_end, *qie_work;
    const HcalTriggerPrimitiveSample *tp_begin, *tp_end, *tp_work; 
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

      tp_begin=(HcalTriggerPrimitiveSample*)tp_first;
      tp_end=(HcalTriggerPrimitiveSample*)(tp_last+1); // one beyond last..

      /// work through the samples
      int currFiberChan=0x3F; // invalid fiber+channel...
      int ncurr=0;
      bool valid=false;

      /** NOTE : there is a minor bug here, if a FED contains both HF and HB/HE data or 
	  HO and HB/HE data and trigger primitives are being sent.  The trigger primitives can be duplicated.
      */
      for (tp_work=tp_begin; tp_work!=tp_end; tp_work++) {
	if (tp_work->raw()==0xFFFF) continue; // filler word
	if (tp_work->fiberAndChan()!=currFiberChan) { // start new set
	  currFiberChan=tp_work->fiberAndChan();
	  // lookup the right channel
	  HcalElectronicsId eid(tp_work->fiberChan(),tp_work->fiber(),spigot,dccid);
	  HcalTrigTowerDetId id=emap.lookupTrigger(eid);
	  if (id.null()) {
	    //	    std::cerr << "No match found for " << eid << std::endl;
	    valid=false;
	    continue;
	  } 
	  tpcont.push_back(HcalTriggerPrimitiveDigi(id));
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


      qie_begin=(HcalQIESample*)daq_first;
      qie_end=(HcalQIESample*)(daq_last+1); // one beyond last..

      /// work through the samples
      currFiberChan=0x3F; // invalid fiber+channel...
      ncurr=0;
      valid=false;

      for (qie_work=qie_begin; qie_work!=qie_end; qie_work++) {
	if (qie_work->raw()==0xFFFF) continue; // filler word
	if (qie_work->fiberAndChan()!=currFiberChan) { // start new set
	  currFiberChan=qie_work->fiberAndChan();
	  // lookup the right channel
	  HcalElectronicsId eid(qie_work->fiberChan(),qie_work->fiber(),spigot,dccid);
	  HcalDetId id=emap.lookup(eid);
	  
	  if (id.subdet()!=sd1 && id.subdet()!=sd2) continue; // filter on subdet id

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

void HcalUnpacker::unpack(const FEDRawData& raw, const HcalMapping& emap, std::vector<HBHEDataFrame>& container, std::vector<HcalTriggerPrimitiveDigi>& tp) {
  HcalUnpacker_impl::unpack<HBHEDataFrame>(raw,emap,container,tp,startSample_,endSample_, sourceIdOffset_, HcalBarrel, HcalEndcap);
}

void HcalUnpacker::unpack(const FEDRawData& raw, const HcalMapping& emap, std::vector<HODataFrame>& container, std::vector<HcalTriggerPrimitiveDigi>& tp) {
  HcalUnpacker_impl::unpack<HODataFrame>(raw,emap,container,tp,startSample_,endSample_, sourceIdOffset_, HcalOuter, HcalOuter);
}

void HcalUnpacker::unpack(const FEDRawData& raw, const HcalMapping& emap, std::vector<HFDataFrame>& container, std::vector<HcalTriggerPrimitiveDigi>& tp) {
  HcalUnpacker_impl::unpack<HFDataFrame>(raw,emap,container,tp,startSample_,endSample_, sourceIdOffset_, HcalForward, HcalForward);
}
