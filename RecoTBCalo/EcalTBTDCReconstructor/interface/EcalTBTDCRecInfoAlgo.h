#ifndef RecoTBCalo_EcalTBTDCReconstructor_EcalTBTDCRecInfoAlgo_HH
#define RecoTBCalo_EcalTBTDCReconstructor_EcalTBTDCRecInfoAlgo_HH

#include "TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRawInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRecInfo.h"

#include <vector>


class EcalTBTDCRecInfoAlgo {

 public:
  EcalTBTDCRecInfoAlgo() : tdcRangeErrorMessageAlreadyDisplayed_(false) {};

  explicit EcalTBTDCRecInfoAlgo(const std::vector<int>& tdcMin, const std::vector<int>& tdcMax);

  ~EcalTBTDCRecInfoAlgo() 
    {
    };

  EcalTBTDCRecInfo reconstruct(const EcalTBTDCRawInfo& TDCRawInfo,const EcalTBEventHeader& eventHeader, bool use2004OffsetConvention) const;

 private:

  std::vector<int> tdcMin_;
  std::vector<int> tdcMax_;
  mutable bool tdcRangeErrorMessageAlreadyDisplayed_;

};

#endif
