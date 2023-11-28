#ifndef HLTrigger_special_HLTPixelIsolTrackL1TFilter_h
#define HLTrigger_special_HLTPixelIsolTrackL1TFilter_h

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

namespace edm {
  class ConfigurationDescriptions;
  class ParameterSet;
}  // namespace edm

class HLTPixelIsolTrackL1TFilter : public HLTFilter {
public:
  explicit HLTPixelIsolTrackL1TFilter(const edm::ParameterSet&);
  ~HLTPixelIsolTrackL1TFilter() override = default;

  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::InputTag const candTag_;
  edm::EDGetTokenT<reco::IsolatedPixelTrackCandidateCollection> const candToken_;
  double const maxptnearby_;
  double const minEnergy_;
  double const minpttrack_;
  double const maxetatrack_;
  double const minetatrack_;
  bool const filterE_;
  int const nMaxTrackCandidates_;
  bool const dropMultiL2Event_;
};

#endif  // HLTrigger_special_HLTPixelIsolTrackL1TFilter_h
