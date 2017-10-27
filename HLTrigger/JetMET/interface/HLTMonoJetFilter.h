#ifndef HLTMonoJetFilter_h
#define HLTMonoJetFilter_h

/** \class HLTMonoJetFilter
 *
 *  \author Srimanobhas Phat
 *
 */

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

namespace edm {
   class ConfigurationDescriptions;
}

//
// class declaration
//

template<typename T>
class HLTMonoJetFilter : public HLTFilter {

   public:
      explicit HLTMonoJetFilter(const edm::ParameterSet&);
      ~HLTMonoJetFilter() override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

   private:
      edm::InputTag inputJetTag_;   // input tag identifying jets
      edm::EDGetTokenT<std::vector<T>> m_theObjectToken;
      double maxPtSecondJet_;
      double maxDeltaPhi_;
      int    triggerType_;
};

#endif //HLTMonoJetFilter_h
