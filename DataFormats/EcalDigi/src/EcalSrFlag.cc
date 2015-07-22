#include "DataFormats/EcalDigi/interface/EcalSrFlag.h"

const char* const EcalSrFlag::srfNames[] = {
  "Suppress",           //SRF_SUPPRESS
  "Zs1",                //SRF_ZS1
  "Zs2",                //SRF_ZS2
  "Full Readout",       //SRF_FULL      
  "Forced Suppress",    //SRF_FORCED_MASK|SRF_SUPPRESS  
  "Forced Zs1",	  //SRF_FORCED_MASK|SRF_ZS1	   
  "Forced Zs2",	  //SRF_FORCED_MASK|SRF_ZS2	   
  "Forced Full Readout" //SRF_FORCED_MASK|SRF_FULL      
};


