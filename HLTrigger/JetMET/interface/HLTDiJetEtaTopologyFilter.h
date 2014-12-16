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
      edm::EDGetTokenT<std::vector<T>> m_theJetToken;
      edm::InputTag inputJetTag_; // input tag identifying jets
      double minPtAve_;
      double atLeastOneJetAbovePT_;
      double minPtTag_;
      double minPtProbe_; 
      double minDphi_;
      double tagEtaMin_;
      double tagEtaMax_;
      double probeEtaMin_;
      double probeEtaMax_;
      bool applyAbsToTag_;
      bool applyAbsToProbe_;
      bool oppositeEta_;
      int    triggerType_;
};

#endif //HLTDiJetEtaTopologyFilter_h
