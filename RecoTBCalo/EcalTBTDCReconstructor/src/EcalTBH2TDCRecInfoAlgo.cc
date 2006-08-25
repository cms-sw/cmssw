#include "RecoTBCalo/EcalTBTDCReconstructor/interface/EcalTBH2TDCRecInfoAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <list>
#include <iostream>

EcalTBH2TDCRecInfoAlgo::EcalTBH2TDCRecInfoAlgo(const double& tdcZero):
  tdcZero_(tdcZero)
{
  
}

EcalTBTDCRecInfo EcalTBH2TDCRecInfoAlgo::reconstruct(const HcalTBTiming& TDCRawInfo) const 
{

  double tdcd = TDCRawInfo.ttcL1Atime() - TDCRawInfo.beamCoincidence();

  if( //!tdcRangeErrorMessageAlreadyDisplayed_  && 
     (tdcd < tdcZero_  || tdcd > tdcZero_ + 25) ){
    std::cout << " ============================" <<std::endl;
    std::cout << " tdc value out of range = "<< tdcd 
	      << " tdcZero = " << tdcZero_ 
	      << std::endl;
    std::cout << " ============================" <<std::endl;
    tdcRangeErrorMessageAlreadyDisplayed_ = true;
  }

  double offset = ( (double)tdcd - (double)tdcZero_ )
    / 25.; //
//   if (use2004OffsetConvention)
//     offset = (1. - offset) ;
  return EcalTBTDCRecInfo(offset); 
}

