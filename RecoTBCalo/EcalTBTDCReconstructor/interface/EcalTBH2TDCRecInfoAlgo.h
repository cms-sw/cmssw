#ifndef RecoTBCalo_EcalTBTDCReconstructor_EcalTBH2TDCRecInfoAlgo_HH
#define RecoTBCalo_EcalTBTDCReconstructor_EcalTBH2TDCRecInfoAlgo_HH

#include "TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRecInfo.h"

#include <vector>

class EcalTBH2TDCRecInfoAlgo {
public:
  EcalTBH2TDCRecInfoAlgo() : tdcRangeErrorMessageAlreadyDisplayed_(false){};

  struct EcalTBH2TDCRanges {
    std::pair<int, int> runRanges;
    double tdcZero;
  };

  explicit EcalTBH2TDCRecInfoAlgo(const std::vector<EcalTBH2TDCRanges>& tdcZero);

  ~EcalTBH2TDCRecInfoAlgo(){};

  EcalTBTDCRecInfo reconstruct(const int& runNumber, const HcalTBTiming& TDCRawInfo) const;

private:
  std::vector<EcalTBH2TDCRanges> tdcZeros_;
  mutable bool tdcRangeErrorMessageAlreadyDisplayed_;
  mutable int actualRun_;
  mutable int actualRange_;
};

#endif
