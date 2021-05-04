#ifndef CondFormats_EcalObjects_EcalChannelStatusCode_H
#define CondFormats_EcalObjects_EcalChannelStatusCode_H
/**
 * Author: Paolo Meridiani
 * Created: 14 Nov 2006
 **/

#include "CondFormats/Serialization/interface/Serializable.h"

#include <iostream>
#include <cstdint>

/**
   

 */

class EcalChannelStatusCode {
public:
  enum Code {
    kOk = 0,
    kDAC,
    kNoLaser,
    kNoisy,
    kNNoisy,
    kNNNoisy,
    kNNNNoisy,
    kNNNNNoisy,
    kFixedG6,
    kFixedG1,
    kFixedG0,
    kNonRespondingIsolated,
    kDeadVFE,
    kDeadFE,
    kNoDataNoTP
  };

  enum Bits { kHV = 0, kLV, kDAQ, kTP, kTrigger, kTemperature, kNextToDead };

public:
  EcalChannelStatusCode() : status_(0) {}
  EcalChannelStatusCode(const uint16_t& encodedStatus) : status_(encodedStatus){};

  void print(std::ostream& s) const { s << "status is: " << status_; }

  /// return decoded status
  Code getStatusCode() const { return Code(status_ & chStatusMask); }

  /// Return the encoded raw status
  uint16_t getEncodedStatusCode() const { return status_; }

  /// Check status of desired bit
  bool checkBit(Bits bit) { return status_ & (0x1 << (bit + kBitsOffset)); }

  static const int chStatusMask = 0x1F;

private:
  static const int kBitsOffset = 5;
  /* bits 1-5 store a status code:
       	0 	channel ok 
  	1 	DAC settings problem, pedestal not in the design range 	
  	2 	channel with no laser, ok elsewhere    
  	3 	noisy 	
  	4 	very noisy 	
  	5-7 	reserved for more categories of noisy channels 	
  	8 	channel at fixed gain 6 (or 6 and 1)
  	9 	channel at fixed gain 1 	
  	10 	channel at fixed gain 0 (dead of type this) 	
  	11 	non responding isolated channel (dead of type other) 
  	12 	channel and one or more neigbors not responding 
                (e.g.: in a dead VFE 5x1 channel) 	
  	13 	channel in TT with no data link, TP data ok    
  	14 	channel in TT with no data link and no TP data  

	bit 6 : HV on/off
        bit 7 : LV on/off
        bit 8 : DAQ in/out 	 
        bit 9 : TP readout on/off 	 
        bit 10: Trigger in/out 	 
        bit 11: Temperature ok/not ok 	 
        bit 12: channel next to a dead channel 
     */
  uint16_t status_;

  COND_SERIALIZABLE;
};
#endif
