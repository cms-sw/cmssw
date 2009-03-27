#ifndef RecoLocalTracker_SiStripZeroSuppression_SiStripRawProcessingAlgorithms_h
#define RecoLocalTracker_SiStripZeroSuppression_SiStripRawProcessingAlgorithms_h

#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripPedestalsSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripFedZeroSuppression.h"

struct SiStripRawProcessingAlgorithms {
  friend class SiStripRawProcessingFactory;

  const std::auto_ptr<SiStripPedestalsSubtractor> subtractorPed;
  const std::auto_ptr<SiStripCommonModeNoiseSubtractor> subtractorCMN;
  const std::auto_ptr<SiStripFedZeroSuppression> suppressor;

  void initialize(const edm::EventSetup& es) {
    subtractorPed->init(es);
    subtractorCMN->init(es);
    suppressor->init(es);
  }

  private:
  SiStripRawProcessingAlgorithms(std::auto_ptr<SiStripPedestalsSubtractor> ped,
				 std::auto_ptr<SiStripCommonModeNoiseSubtractor> cmn,
				 std::auto_ptr<SiStripFedZeroSuppression> zs) 
    :  subtractorPed(ped),
       subtractorCMN(cmn),
       suppressor(zs)
  {}
};
#endif
