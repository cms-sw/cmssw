// -*- C++ -*-
//
// Package:    EventFilter/SiStripRawToDigi
// 
/*
 Description: handles conversion of pipeline address to temporal location using a look-up table
*/
//
// Original Author:  A.-M. Magnan
//         Created:  2009/11/23
//

#ifndef EventFilter_SiStripRawToDigi_PipeAddrToTimeLookupTable_H
#define EventFilter_SiStripRawToDigi_PipeAddrToTimeLookupTable_H

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"


namespace sistrip {
  
  //pipeline address for a given timeLocation (number between 0 and 191)
  //return a number between 0 and 255
  static const uint8_t PIPEADDR[APV_MAX_ADDRESS] = {
    48,49,51,50,54,55,53,52,
    60,61,63,62,58,59,57,56,
    40,41,43,42,46,47,45,44,
    36,37,39,38,34,35,33,32,
    96,97,99,98,102,103,101,100,
    108,109,111,110,106,107,105,104,
    120,121,123,122,126,127,125,124,
    116,117,119,118,114,115,113,112,
    80,81,83,82,86,87,85,84,
    92,93,95,94,90,91,89,88,
    72,73,75,74,78,79,77,76,
    68,69,71,70,66,67,65,64,
    192,193,195,194,198,199,197,196,
    204,205,207,206,202,203,201,200,
    216,217,219,218,222,223,221,220,
    212,213,215,214,210,211,209,208,
    240,241,243,242,246,247,245,244,
    252,253,255,254,250,251,249,248,
    232,233,235,234,238,239,237,236,
    228,229,231,230,226,227,225,224,
    160,161,163,162,166,167,165,164,
    172,173,175,174,170,171,169,168,
    184,185,187,186,190,191,189,188,
    180,181,183,182,178,179,177,176
  };
  //timeLoc for a given pipeline address (number between 0 and 255)
  //return a number between 0 and 191 if valid.
  //set 200 as invalid value.
  static const uint8_t TIMELOC[256] = {
    200,200,200,200,200,200,200,200,
    200,200,200,200,200,200,200,200,
    200,200,200,200,200,200,200,200,
    200,200,200,200,200,200,200,200,
    31,30,28,29,24,25,27,26,
    16,17,19,18,23,22,20,21,
    0,1,3,2,7,6,4,5,
    15,14,12,13,8,9,11,10,
    95,94,92,93,88,89,91,90,
    80,81,83,82,87,86,84,85,
    64,65,67,66,71,70,68,69,
    79,78,76,77,72,73,75,74,
    32,33,35,34,39,38,36,37,
    47,46,44,45,40,41,43,42,
    63,62,60,61,56,57,59,58,
    48,49,51,50,55,54,52,53,
    200,200,200,200,200,200,200,200,
    200,200,200,200,200,200,200,200,
    200,200,200,200,200,200,200,200,
    200,200,200,200,200,200,200,200,
    160,161,163,162,167,166,164,165,
    175,174,172,173,168,169,171,170,
    191,190,188,189,184,185,187,186,
    176,177,179,178,183,182,180,181,
    96,97,99,98,103,102,100,101,
    111,110,108,109,104,105,107,106,
    127,126,124,125,120,121,123,122,
    112,113,115,114,119,118,116,117,
    159,158,156,157,152,153,155,154,
    144,145,147,146,151,150,148,149,
    128,129,131,130,135,134,132,133,
    143,142,140,141,136,137,139,138
  };

  class FEDAddressConversion
  {
  public:
    
    static const uint8_t pipelineAddress(const uint8_t aTimeLocation);
    static const uint8_t timeLocation(const uint8_t aPipelineAddress);

  private:
 
    
  };

  //FEDAddressConversion
    
  inline const uint8_t FEDAddressConversion::pipelineAddress(const uint8_t aTimeLocation){
    if (aTimeLocation<APV_MAX_ADDRESS) return PIPEADDR[aTimeLocation];
    else return 0;
  }

  inline const uint8_t FEDAddressConversion::timeLocation(const uint8_t aPipelineAddress){
    return TIMELOC[aPipelineAddress];
  }



}//namespace
#endif
