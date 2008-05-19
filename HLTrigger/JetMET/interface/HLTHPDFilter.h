#ifndef HLTHPDFilter_h
#define HLTHPDFilter_h

/** \class HLTHPDFilter
 *
 *  \author Fedor Ratnikov (UMd)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

class HLTHPDFilter : public HLTFilter {

   public:
      explicit HLTHPDFilter(const edm::ParameterSet&);
      ~HLTHPDFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag mInputTag; // input tag for HCAL HBHE digis
      double mSeedThresholdEnergy;
      double mShoulderThresholdEnergy;
      double mShoulderToSeedRatio;
};

#endif //HLTHPDFilter_h
