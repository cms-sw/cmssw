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
      edm::InputTag HcalNoiseRBXCollectionTag;
      edm::InputTag HcalNoiseSummaryTag;
      int severity;
      double EMFractionMin;
      int nRBXhitsMax;
      double RBXhitThresh; // energy threshold for hits  
};

#endif //HLTHcalMETNoiseFilter_h
