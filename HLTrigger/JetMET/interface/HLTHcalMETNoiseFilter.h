#ifndef HLTHcalMETNoiseFilter_h
#define HLTHcalMETNoiseFilter_h

/** \class HLTHcalNoiseFilter
 *
 *  \author Leonard Apanasevich (UIC)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

class HLTHcalMETNoiseFilter : public HLTFilter {

   public:
      explicit HLTHcalMETNoiseFilter(const edm::ParameterSet&);
      ~HLTHcalMETNoiseFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag HcalNoiseSummaryTag;
      int severity;

      bool useLooseFilter;
      bool useTightFilter;
      bool useHighLevelFilter;
      bool useCustomFilter;

      double minE2Over10TS;
      double min25GeVHitTime;
      double max25GeVHitTime;
      int maxZeros;
      int maxHPDHits;
      int maxRBXHits;
      double minHPDEMF;
      double minRBXEMF;
};

#endif //HLTHcalMETNoiseFilter_h
