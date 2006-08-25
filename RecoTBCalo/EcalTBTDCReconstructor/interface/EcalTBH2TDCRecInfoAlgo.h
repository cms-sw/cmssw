#ifndef RecoTBCalo_EcalTBTDCReconstructor_EcalTBH2TDCRecInfoAlgo_HH
#define RecoTBCalo_EcalTBTDCReconstructor_EcalTBH2TDCRecInfoAlgo_HH

#include "TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRecInfo.h"

#include <vector>

class EcalTBH2TDCRecInfoAlgo {

 public:
  EcalTBH2TDCRecInfoAlgo() : tdcRangeErrorMessageAlreadyDisplayed_(false) {};

  explicit EcalTBH2TDCRecInfoAlgo(const double& tdcZero);

  ~EcalTBH2TDCRecInfoAlgo() 
    {
    };

  EcalTBTDCRecInfo reconstruct(const HcalTBTiming& TDCRawInfo) const;

 private:

  double tdcZero_;
  mutable bool tdcRangeErrorMessageAlreadyDisplayed_;

};

#endif
