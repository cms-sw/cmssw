#ifndef HLTPixelIsolTrackL1TFilter_h
#define HLTPixelIsolTrackL1TFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/L1Trigger/interface/Jet.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HLTPixelIsolTrackL1TFilter : public HLTFilter {

   public:
      explicit HLTPixelIsolTrackL1TFilter(const edm::ParameterSet&);
      ~HLTPixelIsolTrackL1TFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
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
