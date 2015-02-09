#ifndef HLTDiJetEtaTopologyFilter_h
#define HLTDiJetEtaTopologyFilter_h

/** \class HLTDiJetEtaTopologyFilter
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
class HLTDiJetEtaTopologyFilter : public HLTFilter {

   public:
      explicit HLTDiJetEtaTopologyFilter(const edm::ParameterSet&);
      ~HLTDiJetEtaTopologyFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

   private:
      const edm::InputTag inputJetTag_; // input tag identifying jets
      const edm::EDGetTokenT<std::vector<T>> m_theJetToken;
      const double minPtAve_;
      const double atLeastOneJetAbovePT_;
      const double minPtTag_;
      const double minPtProbe_; 
      const double minDphi_;
      const double tagEtaMin_;
      const double tagEtaMax_;
      const double probeEtaMin_;
      const double probeEtaMax_;
      const bool applyAbsToTag_;
      const bool applyAbsToProbe_;
      const bool oppositeEta_;
      const int    triggerType_;
};

#endif //HLTDiJetEtaTopologyFilter_h
