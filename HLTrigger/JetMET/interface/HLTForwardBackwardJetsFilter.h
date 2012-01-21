#ifndef HLTForwardBackwardJetsFilter_h
#define HLTForwardBackwardJetsFilter_h

/** \class HLTForwardBackwardJetsFilter
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

namespace edm {
   class ConfigurationDescriptions;
}

//
// class decleration
//

class HLTForwardBackwardJetsFilter : public HLTFilter {

   public:
      explicit HLTForwardBackwardJetsFilter(const edm::ParameterSet&);
      ~HLTForwardBackwardJetsFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag inputTag_; // input tag identifying jets
      double minPt_;
      double minEta_;
      double maxEta_;
};

#endif //HLTForwardBackwardJetsFilter_h
