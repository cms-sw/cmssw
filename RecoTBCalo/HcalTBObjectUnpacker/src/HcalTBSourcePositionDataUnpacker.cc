#include "RecoTBCalo/HcalTBObjectUnpacker/interface/HcalTBSourcePositionDataUnpacker.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <string>
#include <map>

using namespace std;

/// Structure for Source Position Data
struct xdaqSourcePositionDriverData {
  int message_counter;
  int timestamp1_sec;
  int timestamp1_usec;
  int status;
  int index_counter;
  int reel_counter;
  int motor_current;
  int speed;
  int timestamp2_sec;
  int timestamp2_usec;
};

struct xdaqSourcePositionDataFormat {
  int cdfHeader[4];
  int maxDrivers;
  unsigned int globalStatus;
  xdaqSourcePositionDriverData driverInfo[4];
  unsigned int word1_low;
  unsigned int word1_high;
};

namespace hcaltb {
  
  void HcalTBSourcePositionDataUnpacker::unpack(const FEDRawData&  raw,
						HcalSourcePositionData&    hspd) const {
    
    const struct xdaqSourcePositionDataFormat* sp =
      (const struct xdaqSourcePositionDataFormat*)(raw.data());
    
    if (raw.size()<sizeof(xdaqSourcePositionDataFormat)) {
      throw cms::Exception("DataFormatError","Fragment too small");
    }
    
    
    hspd.set(sp->driverInfo[0].message_counter,//int message_counter,
	     sp->driverInfo[0].timestamp1_sec,//int timestamp1_sec,
	     sp->driverInfo[0].timestamp1_usec,//int timestamp1_usec,
	     sp->driverInfo[0].timestamp2_sec,//int timestamp2_sec,
	     sp->driverInfo[0].timestamp2_usec,//int timestamp2_usec,
	     sp->driverInfo[0].status,//int status,
	     sp->driverInfo[0].index_counter, //  int index_counter,
	     sp->driverInfo[0].reel_counter,//int reel_counter,
	     sp->driverInfo[0].motor_current,//int motor_current,
	     sp->driverInfo[0].speed,//int speed,
	     -1,//int tube_number,
	     -1,// int driver_id 
	     -1); //int source_id);
    
    return;
  }
}

