#ifndef HLT2L1TkMuonL1TkMuonMuRefDR_h
#define HLT2L1TkMuonL1TkMuonMuRefDR_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"

#include <string>
#include <vector>
namespace trigger {
  class TriggerFilterObjectWithRefs;
}

//
// class declaration
//

class HLT2L1TkMuonL1TkMuonMuRefDR : public HLTFilter {
public:
  explicit HLT2L1TkMuonL1TkMuonMuRefDR(const edm::ParameterSet&);
  ~HLT2L1TkMuonL1TkMuonMuRefDR() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;
  bool getCollections(edm::Event& iEvent,
                      std::vector<l1t::TrackerMuonRef>& coll1,
                      std::vector<l1t::TrackerMuonRef>& coll2,
                      trigger::TriggerFilterObjectWithRefs& filterproduct) const;
  bool computeDR(edm::Event& iEvent, l1t::TrackerMuonRef& c1, l1t::TrackerMuonRef& c2) const;

private:
  // configuration
  const std::vector<edm::InputTag> originTag1_;  // input tag identifying originals 1st product
  const std::vector<edm::InputTag> originTag2_;  // input tag identifying originals 2nd product
  const edm::InputTag inputTag1_;                // input tag identifying filtered 1st product
  const edm::InputTag inputTag2_;                // input tag identifying filtered 2nd product
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> inputToken1_;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> inputToken2_;
  const double minDR_;  // minimum dR between two muon regional candidates linked to L1TkMuon
  const int min_N_;     // number of pairs passing cuts required
  const bool same_;     // 1st and 2nd product are one and the same

  // eta and phi scaling for RegionalMuonCand
  static constexpr unsigned int emtfRegion_{3};
  static constexpr float etaScale_{0.010875};
  static constexpr float phiScale_{2. * M_PI / 576.};
};

#endif  //HLT2L1TkMuonL1TkMuonMuRefDR_h
