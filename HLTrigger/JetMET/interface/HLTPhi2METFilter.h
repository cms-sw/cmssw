#ifndef HLTPhi2METFilter_h
#define HLTPhi2METFilter_h

/** \class HLTPhi2METFilter
 *
 *  \author Dominique J. Mangeol
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include <cmath>
//
// class declaration
//

class HLTPhi2METFilter : public HLTFilter {

   public:
      explicit HLTPhi2METFilter(const edm::ParameterSet&);
      ~HLTPhi2METFilter() override;
      bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

   private:
      edm::InputTag inputJetTag_; // input tag identifying jets
      edm::EDGetTokenT<reco::CaloJetCollection> m_theJetToken;
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> m_theMETToken;
      edm::InputTag inputMETTag_; // input tag identifying for MET
      double minEtjet1_;
      double minEtjet2_;
      double minDPhi_;
      double maxDPhi_;
};

#endif //HLTPhi2METFilter_h
