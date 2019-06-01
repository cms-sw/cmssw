#ifndef HLTNVFilter_h
#define HLTNVFilter_h

/** \class HLTNVFilter
 *
 *  \author Dominique J. Mangeol
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

//
// class declaration
//

class HLTNVFilter : public HLTFilter {
public:
  explicit HLTNVFilter(const edm::ParameterSet&);
  ~HLTNVFilter() override;
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<reco::CaloJetCollection> m_theJetToken;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> m_theMETToken;

  edm::InputTag inputJetTag_;  // input tag identifying jets
  edm::InputTag inputMETTag_;  // input tag identifying for MET
  double minEtjet1_;
  double minEtjet2_;
  double minNV_;
};

#endif  //HLTNVFilter_h
