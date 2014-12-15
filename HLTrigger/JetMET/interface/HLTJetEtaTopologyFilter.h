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
      edm::EDGetTokenT<std::vector<T>> m_theJetToken;
      edm::InputTag inputJetTag_; // input tag identifying jets
      double minPtJet_;
      //double minPtJet3_;
      double jetEtaMin_;
      double jetEtaMax_;
      bool applyAbsToJet_;
      int    triggerType_;
};

#endif //HLTJetEtaTopologyFilter_h
