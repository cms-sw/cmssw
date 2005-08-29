#include "RecoCalorimetry/HcalTBObjectUnpacker/interface/HcalTBTriggerDataUnpacker.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include <iostream>

using namespace std;

// Structure for "old" TriggerData - contains header and trailer
//
struct oldTriggerDataFormat {
  uint32_t cdfHeader[4];
  uint32_t triggerWord;
  uint32_t triggerNumber;
  uint32_t triggerTime_usec;
  uint32_t triggerTime_base;
  uint32_t spillNumber;
  uint32_t runNumber;
  char     runNumberSequenceId[16];
  uint32_t orbitNumber;
  uint32_t bunchNumber;
  uint32_t eventStatus;
  uint32_t filler1;
  uint32_t cdfTrailer[2];
};

// structures for "new" trigger data format - does NOT
// contain header or trailer (header size is variable -
// either 64 or 128 bits)
//
typedef struct StandardTrgMsgBlkStruct {
  uint32_t orbitNumber;
  uint32_t eventNumber;
  uint32_t flags_daq_ttype;
  uint32_t algo_bits_3;
  uint32_t algo_bits_2;
  uint32_t algo_bits_1;
  uint32_t algo_bits_0;
  uint32_t tech_bits;
  uint32_t gps_1234;
  uint32_t gps_5678;
} StandardTrgMsgBlk;

typedef struct newExtendedTrgMsgBlkStruct {
  StandardTrgMsgBlk stdBlock;
  uint32_t triggerWord;
  uint32_t triggerTime_usec;
  uint32_t triggerTime_base;
  uint32_t spillNumber;
  uint32_t runNumber;
  char     runNumberSequenceId[16];
  uint32_t eventStatus;
} newExtendedTrgMsgBlk;


namespace hcaltb {

  void HcalTBTriggerDataUnpacker::unpack(const raw::FEDRawData& raw, hcaltb::HcalTBTriggerData& htbtd) {

    // Use the size to determine which format we have received:
    //
    if (raw.data_.size() == 80) { // "old" test beam trigger format

      const oldTriggerDataFormat *oldtrgblk = (const oldTriggerDataFormat *)(raw.data());
      htbtd.setStandardData(oldtrgblk->orbitNumber,
			    oldtrgblk->triggerNumber,
			    oldtrgblk->bunchNumber,
			    0,          // flags_daq_ttype
			    0, 0, 0, 0, // algo_bits_3->0
			    0,          // tech_bits
			    0, 0        // gps_1234, gps_5678
			    );

      htbtd.setExtendedData(oldtrgblk->triggerWord,
			    oldtrgblk->triggerTime_usec,
			    oldtrgblk->triggerTime_base,
			    oldtrgblk->spillNumber,
			    oldtrgblk->runNumber,
			    oldtrgblk->runNumberSequenceId);
    }
    else {

      const newExtendedTrgMsgBlk *newtrgblk;

      if (raw.data_.size() == 96)  // "new" test beam trigger format,
	                           // 64-bit header
	newtrgblk = (const newExtendedTrgMsgBlk *)(raw.data()+8);

      else if (raw.data_.size() == 104) // "new" test beam trigger format,
	                                // 128-bit header
	newtrgblk = (const newExtendedTrgMsgBlk *)(raw.data()+16);
      else {
	cerr << "HcalTBtdUnpacker.unpack: data of unknown size ";
	cerr << raw.data_.size() << endl;
	return;
      }

      // get bunch number from the header
      //
      const uint32_t *cdflow = (const uint32_t *)raw.data();
      int bunch_id = (*cdflow)>>20;

      htbtd.setStandardData(newtrgblk->stdBlock.orbitNumber,
			    newtrgblk->stdBlock.eventNumber,
			    bunch_id,
			    newtrgblk->stdBlock.flags_daq_ttype,
			    newtrgblk->stdBlock.algo_bits_3,
			    newtrgblk->stdBlock.algo_bits_2,
			    newtrgblk->stdBlock.algo_bits_1,
			    newtrgblk->stdBlock.algo_bits_0,
			    newtrgblk->stdBlock.tech_bits,
			    newtrgblk->stdBlock.gps_1234,
			    newtrgblk->stdBlock.gps_5678);

      htbtd.setExtendedData(newtrgblk->triggerWord,
			    newtrgblk->triggerTime_usec,
			    newtrgblk->triggerTime_base,
			    newtrgblk->spillNumber,
			    newtrgblk->runNumber,
			    newtrgblk->runNumberSequenceId);
    }

  }
}

