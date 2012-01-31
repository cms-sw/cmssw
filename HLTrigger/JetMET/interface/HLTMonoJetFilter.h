#ifndef HLTMonoJetFilter_h
#define HLTMonoJetFilter_h

/** \class HLTMonoJetFilter
 *
 *  \author Srimanobhas Phat
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

namespace edm {
   class ConfigurationDescriptions;
}


//
// class declaration
//

class HLTMonoJetFilter : public HLTFilter {

   public:
      explicit HLTMonoJetFilter(const edm::ParameterSet&);
      ~HLTMonoJetFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag inputJetTag_;   // input tag identifying jets
      double max_PtSecondJet_;
      double max_DeltaPhi_;
};

#endif //HLTMonoJetFilter_h
