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
     (tdcd < tdcZero_ -1 || tdcd > tdcZero_ + 26) )
    {
      edm::LogError("TDCOutOfRange") << " ============================\n" 
				     << " tdc value out of range = "<< tdcd 
				     << " tdcZero = " << tdcZero_ 
				     << "\n" 
				     << " ============================\n" <<std::endl;
      tdcRangeErrorMessageAlreadyDisplayed_ = true;
      return EcalTBTDCRecInfo(-999.);
    }
  
  double offset = ( (double)tdcd - (double)tdcZero_ )
    / 25.; //
  //   if (use2004OffsetConvention)
  //  offset = (1. - offset) ;
  return EcalTBTDCRecInfo(offset); 
}

