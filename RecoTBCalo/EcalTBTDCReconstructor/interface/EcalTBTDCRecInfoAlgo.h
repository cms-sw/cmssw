#ifndef RecoTBCalo_EcalTBTDCReconstructor_EcalTBTDCRecInfoAlgo_HH
#define RecoTBCalo_EcalTBTDCReconstructor_EcalTBTDCRecInfoAlgo_HH

#include "TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRawInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRecInfo.h"

#include <vector>


class EcalTBTDCRecInfoAlgo {
  

 public:


  EcalTBTDCRecInfoAlgo() : tdcRangeErrorMessageAlreadyDisplayed_(false) {};

  struct EcalTBTDCRanges
  {
    std::pair<int,int> runRanges;
    std::vector<double> tdcMin;
    std::vector<double> tdcMax;
  };

  explicit EcalTBTDCRecInfoAlgo(const std::vector<EcalTBTDCRanges>& tdcRanges);

  ~EcalTBTDCRecInfoAlgo() 
    {
    };

  EcalTBTDCRecInfo reconstruct(const EcalTBTDCRawInfo& TDCRawInfo,const EcalTBEventHeader& eventHeader, bool use2004OffsetConvention) const;

 private:

  std::vector<EcalTBTDCRanges> tdcRanges_;
  mutable bool tdcRangeErrorMessageAlreadyDisplayed_;
  mutable int actualRun_;
  mutable int actualRange_;
};

#endif
