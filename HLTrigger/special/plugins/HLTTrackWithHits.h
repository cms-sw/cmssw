#ifndef HLTrigger_HLTTrackWithHits_H
/**\class HLTTrackWithHits
 * Description:
 * templated EDFilter to count the number of tracks with a given hit requirement
 * \author Jean-Roch Vlimant
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class HLTTrackWithHits : public HLTFilter {
public:
  explicit HLTTrackWithHits(const edm::ParameterSet& iConfig)
      : HLTFilter(iConfig),
        src_(iConfig.getParameter<edm::InputTag>("src")),
        minN_(iConfig.getParameter<int>("MinN")),
        maxN_(iConfig.getParameter<int>("MaxN")),
        MinBPX_(iConfig.getParameter<int>("MinBPX")),
        MinFPX_(iConfig.getParameter<int>("MinFPX")),
        MinPXL_(iConfig.getParameter<int>("MinPXL")),
        MinPT_(iConfig.getParameter<double>("MinPT")) {
    srcToken_ = consumes<reco::TrackCollection>(src_);
  }

  ~HLTTrackWithHits() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    makeHLTFilterDescription(desc);
    desc.add<edm::InputTag>("src", edm::InputTag(""));
    desc.add<int>("MinN", 0);
    desc.add<int>("MaxN", 99999);
    desc.add<int>("MinBPX", 0);
    desc.add<int>("MinFPX", 0);
    desc.add<int>("MinPXL", 0);
    desc.add<double>("MinPT", 0.);
    descriptions.add("hltTrackWithHits", desc);
  }

private:
  bool hltFilter(edm::Event& iEvent,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override {
    edm::Handle<reco::TrackCollection> oHandle;
    iEvent.getByToken(srcToken_, oHandle);
    int s = oHandle->size();
    int count = 0;
    for (int i = 0; i != s; ++i) {
      const reco::Track& track = (*oHandle)[i];
      if (track.pt() < MinPT_)
        continue;
      const reco::HitPattern& hits = track.hitPattern();
      if (MinBPX_ > 0 && hits.numberOfValidPixelBarrelHits() >= MinBPX_) {
        ++count;
        continue;
      }
      if (MinFPX_ > 0 && hits.numberOfValidPixelEndcapHits() >= MinFPX_) {
        ++count;
        continue;
      }
      if (MinPXL_ > 0 && hits.numberOfValidPixelHits() >= MinPXL_) {
        ++count;
        continue;
      }
    }

    bool answer = (count >= minN_ && count <= maxN_);
    LogDebug("HLTTrackWithHits") << module(iEvent) << " sees: " << s << " objects. Only: " << count
                                 << " satisfy the hit requirement. Filter answer is: " << (answer ? "true" : "false")
                                 << std::endl;
    return answer;
  }

  edm::InputTag src_;
  edm::EDGetTokenT<reco::TrackCollection> srcToken_;
  int minN_, maxN_, MinBPX_, MinFPX_, MinPXL_;
  double MinPT_;
};

#endif
