#ifndef ReducedTrackerTreeVariables_h
#define ReducedTrackerTreeVariables_h


// For ROOT types with '_t':
#include <Rtypes.h>



// container to hold those static module parameters needed for correct xPrime-Residual calculation (and others), determined with ideal geometry
struct ReducedTrackerTreeVariables{
  
  ReducedTrackerTreeVariables(){this->clear();}
  
  void clear(){
    subdetId = 0;
    nStrips = 0;
    uDirection = vDirection = wDirection = 0;
  }
  
  UInt_t subdetId,
         nStrips;
  Int_t uDirection, vDirection, wDirection;
};



#endif
