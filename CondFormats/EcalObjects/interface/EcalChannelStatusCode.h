#ifndef CondFormats_EcalObjects_EcalChannelStatusCode_H
#define CondFormats_EcalObjects_EcalChannelStatusCode_H
/**
 * Author: Paolo Meridiani
 * Created: 14 Nov 2006
 * $Id: EcalChannelStatusCode.h,v 1.4 2011/05/19 14:24:08 argiro Exp $
 **/


#include <iostream>
#include <boost/cstdint.hpp>

/**
   

 */

class EcalChannelStatusCode {

  public:
    EcalChannelStatusCode();
    EcalChannelStatusCode(const EcalChannelStatusCode & codeStatus);
    EcalChannelStatusCode(const uint16_t& encodedStatus) : status_(encodedStatus) {};
    ~EcalChannelStatusCode();

    //get Methods to be defined according to the final definition

    void print(std::ostream& s) const { s << "status is: " << status_; }

    EcalChannelStatusCode& operator=(const EcalChannelStatusCode& rhs);
    uint16_t getStatusCode() const { return status_; }

    /// Return the decoded status, i.e. the value giving the status code
    uint16_t getDecodedStatusCode() const { return status_&chStatusMask; }

    bool isHVon() const {return status_& HVbitMask;}
    bool isLVon() const {return status_& LVbitMask;}
    
    static const int chStatusMask      = 0x1F;
    static const int HVbitMask         = 0x1<<5;
    static const int LVbitMask         = 0x1<<6;

  private:
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
};
#endif
