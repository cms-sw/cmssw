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
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputMhtTag_; // input tag identifying mht
      bool saveTags_;              // whether to save this tag
      double minMht_;
            
};

#endif //HLTMhtFilter_h
