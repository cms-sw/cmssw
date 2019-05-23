#ifndef HLTJetCollectionsVBFFilter_h
#define HLTJetCollectionsVBFFilter_h

/** \class HLTJetCollectionsVBFFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class decleration
//

template <typename T>
class HLTJetCollectionsVBFFilter : public HLTFilter {
public:
  explicit HLTJetCollectionsVBFFilter(const edm::ParameterSet&);
  ~HLTJetCollectionsVBFFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  edm::EDGetTokenT<std::vector<edm::RefVector<std::vector<T>, T, edm::refhelper::FindUsingAdvance<std::vector<T>, T>>>>
      m_theJetToken;
  edm::InputTag inputTag_;     // input tag identifying jet collections
  edm::InputTag originalTag_;  // input tag original jet collection
  double softJetPt_;
  double hardJetPt_;
  double minDeltaEta_;
  double thirdJetPt_;
  double maxAbsJetEta_;
  double maxAbsThirdJetEta_;
  unsigned int minNJets_;  // number of required jets passing cuts after cleaning
  int triggerType_;
};

#endif  //HLTJetCollectionsVBFFilter_h
