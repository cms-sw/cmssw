// <author>Mariana Araujo</author>
// <email>mariana.araujo@cern.ch</email>
// <created>2020-12-30</created>
// <description>
// HLT filter module to select events with min and max multiplicity
// in each tracker of the PPS for PCL
// Adapted from a preexisting filter code by Laurent Forthomme
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

class HLTPPSCalFilter : public edm::global::EDFilter<> {
public:
  explicit HLTPPSCalFilter(const edm::ParameterSet&);
  ~HLTPPSCalFilter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  edm::InputTag pixelLocalTrackInputTag_;  // Input tag identifying the pixel detector
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelLocalTrack>> pixelLocalTrackToken_;

  edm::InputTag stripLocalTrackInputTag_;  // Input tag identifying the strip detector
  edm::EDGetTokenT<edm::DetSetVector<TotemRPLocalTrack>> stripLocalTrackToken_;

  edm::InputTag diamondLocalTrackInputTag_;  // Input tag identifying the diamond detector
  edm::EDGetTokenT<edm::DetSetVector<CTPPSDiamondLocalTrack>> diamondLocalTrackToken_;

  int minTracks_;
  int maxTracks_;

  bool do_express_;
};

void HLTPPSCalFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("pixelLocalTrackInputTag", edm::InputTag("ctppsPixelLocalTracks"))
      ->setComment("input tag of the pixel local track collection");
  desc.add<edm::InputTag>("stripLocalTrackInputTag", edm::InputTag("totemRPLocalTrackFitter"))
      ->setComment("input tag of the strip local track collection");
  desc.add<edm::InputTag>("diamondLocalTrackInputTag", edm::InputTag("ctppsDiamondLocalTracks"))
      ->setComment("input tag of the diamond local track collection");

  desc.add<int>("minTracks", 1)->setComment("minimum number of tracks in pot");
  desc.add<int>("maxTracks", -1)->setComment("maximum number of tracks in pot");

  desc.add<bool>("do_express", true)->setComment("toggle on filter type; true for Express, false for Prompt");
  
  desc.add<int>("triggerType", trigger::TriggerTrack);

  descriptions.add("hltPPSCalFilter", desc);
}

HLTPPSCalFilter::HLTPPSCalFilter(const edm::ParameterSet& iConfig)
    : pixelLocalTrackInputTag_(iConfig.getParameter<edm::InputTag>("pixelLocalTrackInputTag")),
      pixelLocalTrackToken_(consumes<edm::DetSetVector<CTPPSPixelLocalTrack>>(pixelLocalTrackInputTag_)),

      stripLocalTrackInputTag_(iConfig.getParameter<edm::InputTag>("stripLocalTrackInputTag")),
      stripLocalTrackToken_(consumes<edm::DetSetVector<TotemRPLocalTrack>>(stripLocalTrackInputTag_)),

      diamondLocalTrackInputTag_(iConfig.getParameter<edm::InputTag>("diamondLocalTrackInputTag")),
      diamondLocalTrackToken_(consumes<edm::DetSetVector<CTPPSDiamondLocalTrack>>(diamondLocalTrackInputTag_)),
    
      minTracks_(iConfig.getParameter<int>("minTracks")),
      maxTracks_(iConfig.getParameter<int>("maxTracks")),
      
      do_express_(iConfig.getParameter<bool>("do_express"))
{
}

bool HLTPPSCalFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // Helper tool to count valid tracks
  const auto valid_trks = [](const auto& trk) { return trk.isValid(); };

  // maps for assigning filter pass / fail
  std::unordered_map<uint32_t, bool> pixel_filter_;
  std::unordered_map<uint32_t, bool> strip_filter_;
  std::unordered_map<uint32_t, bool> diam_filter_;

  // pixel map definition
  pixel_filter_[CTPPSPixelDetId(0, 0, 3)] = false;
  pixel_filter_[CTPPSPixelDetId(0, 2, 3)] = false;
  pixel_filter_[CTPPSPixelDetId(1, 0, 3)] = false;
  pixel_filter_[CTPPSPixelDetId(1, 2, 3)] = false;

  // strip map definition
  // diamond map definition
  
  // First filter on pixels (2017+) selection
  if (!pixel_filter_.empty()) {
    edm::Handle<edm::DetSetVector<CTPPSPixelLocalTrack>> pixelTracks;
    iEvent.getByToken(pixelLocalTrackToken_, pixelTracks);

    for (const auto& rpv : *pixelTracks) {
      if (pixel_filter_.count(rpv.id) == 0) {
        continue;
      }
      auto& fltr = pixel_filter_.at(rpv.id);
      fltr = true;

      const auto ntrks = std::count_if(rpv.begin(), rpv.end(), valid_trks);
      if (minTracks_ > 0 && ntrks < minTracks_)
        fltr = false;
      if (maxTracks_ > 0 && ntrks > maxTracks_)
        fltr = false;

    }

    // compilation of filter conditions
    if(do_express_) {
      return (pixel_filter_.at(CTPPSPixelDetId(0,0,3)) && pixel_filter_.at(CTPPSPixelDetId(0,2,3))) || (pixel_filter_.at(CTPPSPixelDetId(1,0,3)) && pixel_filter_.at(CTPPSPixelDetId(1,2,3)));
    }
    else {
      return (pixel_filter_.at(CTPPSPixelDetId(0,0,3)) || pixel_filter_.at(CTPPSPixelDetId(0,2,3))) || (pixel_filter_.at(CTPPSPixelDetId(1,0,3)) || pixel_filter_.at(CTPPSPixelDetId(1,2,3))); 
    }

  }

  // Then filter on strips (2016-17) selection
  if (!strip_filter_.empty()) {
    edm::Handle<edm::DetSetVector<TotemRPLocalTrack>> stripTracks;
    iEvent.getByToken(stripLocalTrackToken_, stripTracks);

    for (const auto& rpv : *stripTracks) {
      if (strip_filter_.count(rpv.id) == 0)
        continue;
      auto& fltr = strip_filter_.at(rpv.id);
      fltr = true;
      
      const auto ntrks = std::count_if(rpv.begin(), rpv.end(), valid_trks);
      if (minTracks_ > 0 && ntrks < minTracks_)
        fltr = false;
      if (maxTracks_ > 0 && ntrks > maxTracks_)
        fltr = false;
    }
  }

  // Finally filter on diamond (2016+) selection
  if (!diam_filter_.empty()) {
    edm::Handle<edm::DetSetVector<CTPPSDiamondLocalTrack>> diamondTracks;
    iEvent.getByToken(diamondLocalTrackToken_, diamondTracks);

    for (const auto& rpv : *diamondTracks) {
      if (diam_filter_.count(rpv.id) == 0)
        continue;
      auto& fltr = diam_filter_.at(rpv.id);
      fltr = true;
      
      const auto ntrks = std::count_if(rpv.begin(), rpv.end(), valid_trks);
      if (minTracks_ > 0 && ntrks < minTracks_)
        fltr = false;
      if (maxTracks_ > 0 && ntrks > maxTracks_)
        fltr = false;
    }
  }

  return false;
}

// define as a framework module
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTPPSCalFilter);
