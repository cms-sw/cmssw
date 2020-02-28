#ifndef HLTEcalIsolationFilter_h
#define HLTEcalIsolationFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HLTEcalIsolationFilter : public HLTFilter {
public:
  explicit HLTEcalIsolationFilter(const edm::ParameterSet&);
  ~HLTEcalIsolationFilter() override;
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::InputTag candTag_;
  edm::EDGetTokenT<reco::IsolatedPixelTrackCandidateCollection> candToken_;
  double maxennearby;
  double minen;
  int maxhitout;
  int maxhitin;
  double maxenin;
  double maxenout;
  double maxetacand;
};

#endif
