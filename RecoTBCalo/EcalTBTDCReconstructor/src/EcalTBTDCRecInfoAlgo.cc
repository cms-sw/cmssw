#include "RecoTBCalo/EcalTBTDCReconstructor/interface/EcalTBTDCRecInfoAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <list>
#include <iostream>

EcalTBTDCRecInfoAlgo::EcalTBTDCRecInfoAlgo(const std::vector<EcalTBTDCRanges>& tdcRanges) :
  tdcRanges_(tdcRanges), tdcRangeErrorMessageAlreadyDisplayed_(false), actualRun_(-1) , actualRange_(-1)
{
  
}

EcalTBTDCRecInfo EcalTBTDCRecInfoAlgo::reconstruct(const EcalTBTDCRawInfo& TDCRawInfo,const EcalTBEventHeader& eventHeader, bool use2004OffsetConvention) const 
{
  if (actualRun_ != eventHeader.runNumber() )
    {
      actualRun_ = eventHeader.runNumber();
      actualRange_ = -1;
      for (unsigned int i=0; i<tdcRanges_.size();i++)
	if ( eventHeader.runNumber() <= tdcRanges_[i].runRanges.second && eventHeader.runNumber() >= tdcRanges_[i].runRanges.first )
	  actualRange_ = i;
      
      if (actualRange_ == -1)
	{
	  edm::LogError("TDCRange not found") << "TDC range not found";
	  return EcalTBTDCRecInfo(-1);
	}
    }
  
  int eventType;
  eventType=( (eventHeader.dbEventType() == 0) ? 0 : (eventHeader.dbEventType()-1));

  int tdcd = TDCRawInfo[0].tdcValue();

  if( !tdcRangeErrorMessageAlreadyDisplayed_ 
      && (tdcd < tdcRanges_[actualRange_].tdcMin[eventType] || tdcd > tdcRanges_[actualRange_].tdcMax[eventType]) ){
    std::cout << " ============================" <<std::endl;
    std::cout << " Event type: " << eventType << std::endl;
    std::cout << " TDC values not correctly set Raw TDC = "<< tdcd 
	      << " (Min/Max = "<< tdcRanges_[actualRange_].tdcMin[eventType] << "/" << tdcRanges_[actualRange_].tdcMax[eventType]
	      << std::endl;
    std::cout << " ============================" <<std::endl;
    tdcRangeErrorMessageAlreadyDisplayed_ = true;
  }

  double offset = ( (double)tdcd - (double)tdcRanges_[actualRange_].tdcMin[eventType] )
    / ((double)tdcRanges_[actualRange_].tdcMax[eventType]-(double)tdcRanges_[actualRange_].tdcMin[eventType]);
  if (use2004OffsetConvention)
    offset = (1. - offset) ;
  return EcalTBTDCRecInfo(offset); 
}

