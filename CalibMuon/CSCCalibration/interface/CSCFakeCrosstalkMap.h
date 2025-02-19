#ifndef _CSC_FAKE_CROSSTALK_MAP
#define _CSC_FAKE_CROSSTALK_MAP

#include "CondFormats/CSCObjects/interface/CSCcrosstalk.h"

class CSCFakeCrosstalkMap{
 public:
  CSCFakeCrosstalkMap();
  
  const CSCcrosstalk & get(){
    return (*cncrosstalk);
  }
    
  void smear();

 private:
  float theMean;
  float theMin;
  float theMinChi;
  int theSeed;
  long int theM;
  
  CSCcrosstalk *cncrosstalk ;
  
};

#endif
