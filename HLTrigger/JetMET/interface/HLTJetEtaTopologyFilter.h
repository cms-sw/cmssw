#ifndef HLTJetEtaTopologyFilter_h
#define HLTJetEtaTopologyFilter_h

/** \class HLTJetEtaTopologyFilter
 *
 *  \author Tomasz Fruboes
 *    based on HLTDiJetAveFilter
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
class HLTJetEtaTopologyFilter : public HLTFilter {

   public:
      explicit HLTJetEtaTopologyFilter(const edm::ParameterSet&);
      ~HLTJetEtaTopologyFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

   private:
      const edm::InputTag inputJetTag_; // input tag identifying jets
      const edm::EDGetTokenT<std::vector<T>> m_theJetToken;
      const double minPtJet_;
      //double minPtJet3_;
      const double jetEtaMin_;
      const double jetEtaMax_;
      const bool applyAbsToJet_;
      const int    triggerType_;
};

#endif //HLTJetEtaTopologyFilter_h
