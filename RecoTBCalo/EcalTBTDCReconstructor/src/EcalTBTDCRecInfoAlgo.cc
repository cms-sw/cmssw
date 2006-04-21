#include "RecoTBCalo/EcalTBTDCReconstructor/interface/EcalTBTDCRecInfoAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <list>
#include <iostream>

EcalTBTDCRecInfoAlgo::EcalTBTDCRecInfoAlgo(const std::vector<int>& tdcMin, const std::vector<int>& tdcMax) :
  tdcMin_(tdcMin), tdcMax_(tdcMax), tdcRangeErrorMessageAlreadyDisplayed_(false)
{
  
}

EcalTBTDCRecInfo EcalTBTDCRecInfoAlgo::reconstruct(const EcalTBTDCRawInfo& TDCRawInfo,const EcalTBEventHeader& eventHeader) const 
{
  int eventType;
  eventType=eventHeader.dbEventType();

  int tdcd = TDCRawInfo[0].tdcValue();

  if( !tdcRangeErrorMessageAlreadyDisplayed_ 
      && (tdcd < tdcMin_[eventType] || tdcd > tdcMax_[eventType]) ){
    std::cout << " ============================" <<std::endl;
    std::cout << " Event type: " << eventType << std::endl;
    std::cout << " TDC values not correctly set Raw TDC = "<< tdcd 
	      << " (Min/Max = "<< tdcMin_[eventType] << "/" <<tdcMax_[eventType]
	      << std::endl;
    std::cout << " ============================" <<std::endl;
    tdcRangeErrorMessageAlreadyDisplayed_ = true;
  }

  double offset = ( (double)tdcd - (double)tdcMin_[eventType] )
    / ((double)tdcMax_[eventType]-(double)tdcMin_[eventType]);
  offset = (1. - offset) ;
  return EcalTBTDCRecInfo(offset); 
}

