#ifndef HLT2jetGapFilter_h
#define HLT2jetGapFilter_h

/** \class HLT2jetGapFilter
 *
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

namespace edm {
   class ConfigurationDescriptions;
}

//
// class declaration
//

class HLT2jetGapFilter : public HLTFilter {

   public:
      explicit HLT2jetGapFilter(const edm::ParameterSet&);
      ~HLT2jetGapFilter() override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

   private:
      edm::EDGetTokenT<reco::CaloJetCollection> m_theCaloJetToken;

      edm::InputTag inputTag_; // input tag identifying jets
      double minEt_;
      double minEta_;
};

#endif //HLT2jetGapFilter_h
