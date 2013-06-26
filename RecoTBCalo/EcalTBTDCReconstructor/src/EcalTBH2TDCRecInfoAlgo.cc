#include "RecoTBCalo/EcalTBTDCReconstructor/interface/EcalTBH2TDCRecInfoAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <list>
#include <vector>
#include <iostream>

EcalTBH2TDCRecInfoAlgo::EcalTBH2TDCRecInfoAlgo(const std::vector<EcalTBH2TDCRanges>& tdcZeros):
  tdcZeros_(tdcZeros), actualRun_(-1) , actualRange_(-1)
{
  
}

EcalTBTDCRecInfo EcalTBH2TDCRecInfoAlgo::reconstruct(const int& runNumber, const HcalTBTiming& TDCRawInfo) const 
{
  if (actualRun_ != runNumber )
    {
      actualRun_ = runNumber;
      actualRange_ = -1;
      for (unsigned int i=0; i<tdcZeros_.size();i++)
	if ( runNumber <= tdcZeros_[i].runRanges.second && runNumber >= tdcZeros_[i].runRanges.first )
	  actualRange_ = i;
      
      if (actualRange_ == -1)
	{
	  edm::LogError("TDCRange not found") << "TDC range not found";
	  return EcalTBTDCRecInfo(-1);
	}
    }

  int ntdc = TDCRawInfo.BeamCoincidenceCount();
  edm::LogInfo("") << "EcalTBH2TDCRecInfoAlgo::reconstruct # tdc hits: " << ntdc << std::endl;;
  if(ntdc>1) {
    for(int i=0; i<ntdc; ++i) {
      edm::LogInfo("") << "hit i: " << i << " tdc: " << TDCRawInfo.BeamCoincidenceHits(i) << std::endl;
    }
  }


  if(ntdc==0) {
     edm::LogError("NoTDCHits") << "no TDC hits. TDC info not reliable" << std::endl;
     return EcalTBTDCRecInfo(-999.);
  }


  //double tdcd = TDCRawInfo.ttcL1Atime() - TDCRawInfo.beamCoincidence();
  double tdcd = TDCRawInfo.ttcL1Atime() - TDCRawInfo.BeamCoincidenceHits(0);

  if( //!tdcRangeErrorMessageAlreadyDisplayed_  && 
     (tdcd < tdcZeros_[actualRange_].tdcZero -1 || tdcd > tdcZeros_[actualRange_].tdcZero + 26) )
    {
      edm::LogError("TDCOutOfRange") << " ============================\n" 
				     << " tdc value out of range = "<< tdcd 
				     << " tdcZero = " << tdcZeros_[actualRange_].tdcZero
				     << "\n" 
				     << " ============================\n" <<std::endl;
      tdcRangeErrorMessageAlreadyDisplayed_ = true;
      return EcalTBTDCRecInfo(-999.);
    }
  
  double offset = ( (double)tdcd - (double)tdcZeros_[actualRange_].tdcZero )
    / 25.; //
  //   if (use2004OffsetConvention)
  //  offset = (1. - offset) ;
  return EcalTBTDCRecInfo(offset); 
}

