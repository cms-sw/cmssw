#ifndef HLTPixelIsolTrackFilter_h
#define HLTPixelIsolTrackFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HLTPixelIsolTrackFilter : public HLTFilter {

   public:
      explicit HLTPixelIsolTrackFilter(const edm::ParameterSet&);
      ~HLTPixelIsolTrackFilter() override;
      bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

   private:
      edm::EDGetTokenT<reco::IsolatedPixelTrackCandidateCollection> candToken_;
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> hltGTseedToken_;
      edm::InputTag candTag_;
      edm::InputTag hltGTseedlabel_;
      double maxptnearby_;
      double minpttrack_;
      double minetatrack_;
      double maxetatrack_;
      bool filterE_;
      double minEnergy_;
      int nMaxTrackCandidates_;
      bool dropMultiL2Event_;
      double minDeltaPtL1Jet_;
};

#endif
