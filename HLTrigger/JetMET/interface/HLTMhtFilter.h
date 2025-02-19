#ifndef HLTMhtFilter_h
#define HLTMhtFilter_h

/** \class HLTMhtFilter
 *
 *  \author Gheorghe Lungu
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

namespace edm {
   class ConfigurationDescriptions;
}


//
// class declaration
//

class HLTMhtFilter : public HLTFilter {

   public:
      explicit HLTMhtFilter(const edm::ParameterSet&);
      ~HLTMhtFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag inputMhtTag_; // input tag identifying mht
      double minMht_;
};

#endif //HLTMhtFilter_h
