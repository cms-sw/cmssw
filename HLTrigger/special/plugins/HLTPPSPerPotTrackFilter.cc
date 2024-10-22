// <author>Laurent Forthomme</author>
// <email>lforthom@cern.ch</email>
// <created>2020-02-17</created>
// <description>
// HLT filter module to select events with tracks in the PPS detector
// </description>

#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"

#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"    // pixel
#include "DataFormats/CTPPSReco/interface/TotemRPLocalTrack.h"       // strip
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondLocalTrack.h"  // diamond

#include <unordered_map>

class HLTPPSPerPotTrackFilter : public edm::global::EDFilter<> {
public:
  explicit HLTPPSPerPotTrackFilter(const edm::ParameterSet&);
  ~HLTPPSPerPotTrackFilter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelLocalTrack>> pixelLocalTrackToken_;
  edm::EDGetTokenT<edm::DetSetVector<TotemRPLocalTrack>> stripLocalTrackToken_;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSDiamondLocalTrack>> diamondLocalTrackToken_;

  struct PerPotFilter {
    int minTracks;
    int maxTracks;
  };
  std::unordered_map<uint32_t, PerPotFilter> pixel_filter_;
  std::unordered_map<uint32_t, PerPotFilter> strip_filter_;
  std::unordered_map<uint32_t, PerPotFilter> diam_filter_;

  // Helper tool to count valid tracks
  static constexpr auto valid_trks_ = [](const auto& trk) { return trk.isValid(); };
};

void HLTPPSPerPotTrackFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("pixelLocalTrackInputTag", edm::InputTag("ctppsPixelLocalTracks"))
      ->setComment("input tag of the pixel local track collection");
  desc.add<edm::InputTag>("stripLocalTrackInputTag", edm::InputTag("totemRPLocalTrackFitter"))
      ->setComment("input tag of the strip local track collection");
  desc.add<edm::InputTag>("diamondLocalTrackInputTag", edm::InputTag("ctppsDiamondLocalTracks"))
      ->setComment("input tag of the diamond local track collection");

  edm::ParameterSetDescription filterValid;
  filterValid.add<unsigned int>("detid", 0)->setComment("station/pot raw DetId");
  filterValid.add<int>("minTracks", -1)->setComment("minimum number of tracks in pot");
  filterValid.add<int>("maxTracks", -1)->setComment("maximum number of tracks in pot");

  std::vector<edm::ParameterSet> vPixelDefault;
  auto& near_pix45 = vPixelDefault.emplace_back();
  near_pix45.addParameter<unsigned int>("detid", CTPPSPixelDetId(0, 2, 2));  // arm-station-rp
  near_pix45.addParameter<int>("minTracks", 2);
  near_pix45.addParameter<int>("maxTracks", -1);
  auto& far_pix45 = vPixelDefault.emplace_back();
  far_pix45.addParameter<unsigned int>("detid", CTPPSPixelDetId(0, 2, 3));  // arm-station-rp
  far_pix45.addParameter<int>("minTracks", 2);
  far_pix45.addParameter<int>("maxTracks", -1);
  auto& near_pix56 = vPixelDefault.emplace_back();
  near_pix56.addParameter<unsigned int>("detid", CTPPSPixelDetId(1, 2, 2));  // arm-station-rp
  near_pix56.addParameter<int>("minTracks", 2);
  near_pix56.addParameter<int>("maxTracks", -1);
  auto& far_pix56 = vPixelDefault.emplace_back();
  far_pix56.addParameter<unsigned int>("detid", CTPPSPixelDetId(1, 2, 3));  // arm-station-rp
  far_pix56.addParameter<int>("minTracks", 2);
  far_pix56.addParameter<int>("maxTracks", -1);
  desc.addVPSet("pixelFilter", filterValid, vPixelDefault);

  std::vector<edm::ParameterSet> vStripDefault;
  desc.addVPSet("stripFilter", filterValid, vStripDefault);

  std::vector<edm::ParameterSet> vDiamDefault;
  desc.addVPSet("diamondFilter", filterValid, vDiamDefault);

  desc.add<int>("triggerType", trigger::TriggerTrack);

  descriptions.add("hltPPSPerPotTrackFilter", desc);
}

HLTPPSPerPotTrackFilter::HLTPPSPerPotTrackFilter(const edm::ParameterSet& iConfig) {
  // First define pixels (2017+) selection
  const auto& pixelVPset = iConfig.getParameter<std::vector<edm::ParameterSet>>("pixelFilter");
  if (!pixelVPset.empty()) {
    pixelLocalTrackToken_ = consumes<edm::DetSetVector<CTPPSPixelLocalTrack>>(
        iConfig.getParameter<edm::InputTag>("pixelLocalTrackInputTag"));
    for (const auto& pset : pixelVPset)
      pixel_filter_[pset.getParameter<unsigned int>("detid")] =
          PerPotFilter{pset.getParameter<int>("minTracks"), pset.getParameter<int>("maxTracks")};
  }
  // Then define strips (2016-17) selection
  const auto& stripVPset = iConfig.getParameter<std::vector<edm::ParameterSet>>("stripFilter");
  if (!stripVPset.empty()) {
    stripLocalTrackToken_ =
        consumes<edm::DetSetVector<TotemRPLocalTrack>>(iConfig.getParameter<edm::InputTag>("stripLocalTrackInputTag"));
    for (const auto& pset : stripVPset)
      strip_filter_[pset.getParameter<unsigned int>("detid")] =
          PerPotFilter{pset.getParameter<int>("minTracks"), pset.getParameter<int>("maxTracks")};
  }
  // Finally define diamond (2016+) selection
  const auto& diamVPset = iConfig.getParameter<std::vector<edm::ParameterSet>>("diamondFilter");
  if (!diamVPset.empty()) {
    diamondLocalTrackToken_ = consumes<edm::DetSetVector<CTPPSDiamondLocalTrack>>(
        iConfig.getParameter<edm::InputTag>("diamondLocalTrackInputTag"));
    for (const auto& pset : diamVPset)
      diam_filter_[pset.getParameter<unsigned int>("detid")] =
          PerPotFilter{pset.getParameter<int>("minTracks"), pset.getParameter<int>("maxTracks")};
  }
}

bool HLTPPSPerPotTrackFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // First filter on pixels
  if (!pixel_filter_.empty()) {
    edm::Handle<edm::DetSetVector<CTPPSPixelLocalTrack>> pixelTracks;
    iEvent.getByToken(pixelLocalTrackToken_, pixelTracks);

    for (const auto& rpv : *pixelTracks) {
      if (pixel_filter_.count(rpv.id) == 0)
        continue;
      const auto& fltr = pixel_filter_.at(rpv.id);

      const auto ntrks = std::count_if(rpv.begin(), rpv.end(), valid_trks_);
      if (fltr.minTracks > 0 && ntrks < fltr.minTracks)
        return false;
      if (fltr.maxTracks > 0 && ntrks > fltr.maxTracks)
        return false;
    }
  }

  // Then filter on strips
  if (!strip_filter_.empty()) {
    edm::Handle<edm::DetSetVector<TotemRPLocalTrack>> stripTracks;
    iEvent.getByToken(stripLocalTrackToken_, stripTracks);

    for (const auto& rpv : *stripTracks) {
      if (strip_filter_.count(rpv.id) == 0)
        continue;
      const auto& fltr = strip_filter_.at(rpv.id);

      const auto ntrks = std::count_if(rpv.begin(), rpv.end(), valid_trks_);
      if (fltr.minTracks > 0 && ntrks < fltr.minTracks)
        return false;
      if (fltr.maxTracks > 0 && ntrks > fltr.maxTracks)
        return false;
    }
  }

  // Finally filter on diamond
  if (!diam_filter_.empty()) {
    edm::Handle<edm::DetSetVector<CTPPSDiamondLocalTrack>> diamondTracks;
    iEvent.getByToken(diamondLocalTrackToken_, diamondTracks);

    for (const auto& rpv : *diamondTracks) {
      if (diam_filter_.count(rpv.id) == 0)
        continue;
      const auto& fltr = diam_filter_.at(rpv.id);

      const auto ntrks = std::count_if(rpv.begin(), rpv.end(), valid_trks_);
      if (fltr.minTracks > 0 && ntrks < fltr.minTracks)
        return false;
      if (fltr.maxTracks > 0 && ntrks > fltr.maxTracks)
        return false;
    }
  }

  return true;
}

// define as a framework module
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTPPSPerPotTrackFilter);
