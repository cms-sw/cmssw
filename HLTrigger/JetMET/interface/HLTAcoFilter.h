#ifndef HLTAcoFilter_h
#define HLTAcoFilter_h

/** \class HLTAcoFilter
 *
 *  \author Dominique J. Mangeol
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include <string>
#include <cstring>
#include <cmath>

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"

namespace edm {
   class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTAcoFilter : public HLTFilter {

   public:
      explicit HLTAcoFilter(const edm::ParameterSet&);
      ~HLTAcoFilter() override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

   private:

      edm::EDGetTokenT<reco::CaloJetCollection> m_theJetToken;
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> m_theMETToken;

      edm::InputTag inputJetTag_; // input tag identifying jets
      edm::InputTag inputMETTag_; // input tag identifying for MET
      double minEtjet1_;
      double minEtjet2_;
      double minDPhi_;
      double maxDPhi_;
      std::string AcoString_;
};

#endif //HLTAcoFilter_h
