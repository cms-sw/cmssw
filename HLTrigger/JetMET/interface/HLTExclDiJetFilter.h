#ifndef HLTExclDiJetFilter_h
#define HLTExclDiJetFilter_h

/** \class HLTExclDiJetFilter
 *
 *  \author Dominique J. Mangeol
 *
 */

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

namespace edm {
   class ConfigurationDescriptions;
}

//
// class declaration
//
template<typename T>
class HLTExclDiJetFilter : public HLTFilter {

   public:
      explicit HLTExclDiJetFilter(const edm::ParameterSet&);
      ~HLTExclDiJetFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag inputJetTag_; // input tag identifying jets
      double minPtJet_;
      double minHFe_;
      bool   HF_OR_;
      int    triggerType_;
};

#endif //HLTExclDiJetFilter_h
