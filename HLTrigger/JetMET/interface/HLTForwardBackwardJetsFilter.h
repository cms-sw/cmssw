#ifndef HLTForwardBackwardJetsFilter_h
#define HLTForwardBackwardJetsFilter_h

/** \class HLTForwardBackwardJetsFilter
 *
 */

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

namespace edm {
   class ConfigurationDescriptions;
}

//
// class decleration
//
template<typename T>
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
      unsigned int nNeg_;
      unsigned int nPos_;
      unsigned int nTot_;
      int    triggerType_;
};

#endif //HLTForwardBackwardJetsFilter_h
