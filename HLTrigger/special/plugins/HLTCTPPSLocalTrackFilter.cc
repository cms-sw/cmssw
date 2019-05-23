// <author>Cristovao Beirao da Cruz e Silva</author>
// <email>cbeiraod@cern.ch</email>
// <created>2017-10-26</created>
// <description>
// HLT filter module to select events with tracks in the CTPPS detector
// </description>

// system include files

// user include files
#include "HLTCTPPSLocalTrackFilter.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"

//
// fill discriptions
//
void HLTCTPPSLocalTrackFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("pixelLocalTrackInputTag", edm::InputTag("ctppsPixelLocalTracks"))
      ->setComment("input tag of the pixel local track collection");
  desc.add<edm::InputTag>("stripLocalTrackInputTag", edm::InputTag("totemRPLocalTrackFitter"))
      ->setComment("input tag of the strip local track collection");
  desc.add<edm::InputTag>("diamondLocalTrackInputTag", edm::InputTag("ctppsDiamondLocalTracks"))
      ->setComment("input tag of the diamond local track collection");

  desc.add<bool>("usePixel", true)->setComment("whether to consider the pixel detectors");
  desc.add<bool>("useStrip", false)->setComment("whether to consider the strip detectors");
  desc.add<bool>("useDiamond", false)->setComment("whether to consider the diamond detectors");

  desc.add<int>("minTracks", 2)->setComment("minimum number of tracks");
  desc.add<int>("minTracksPerArm", 1)->setComment("minimum number of tracks per arm of the CTPPS detector");

  desc.add<int>("maxTracks", -1)->setComment("maximum number of tracks, if smaller than minTracks it will be ignored");
  desc.add<int>("maxTracksPerArm", -1)
      ->setComment(
          "maximum number of tracks per arm of the CTPPS detector, if smaller than minTrackPerArm it will be ignored");
  desc.add<int>("maxTracksPerPot", -1)
      ->setComment("maximum number of tracks per roman pot of the CTPPS detector, if negative it will be ignored");

  desc.add<int>("triggerType", trigger::TriggerTrack);

  descriptions.add("hltCTPPSLocalTrackFilter", desc);
  return;
}

//
// destructor and constructor
//
HLTCTPPSLocalTrackFilter::~HLTCTPPSLocalTrackFilter() = default;

HLTCTPPSLocalTrackFilter::HLTCTPPSLocalTrackFilter(const edm::ParameterSet& iConfig)
    : pixelLocalTrackInputTag_(iConfig.getParameter<edm::InputTag>("pixelLocalTrackInputTag")),
      stripLocalTrackInputTag_(iConfig.getParameter<edm::InputTag>("stripLocalTrackInputTag")),
      diamondLocalTrackInputTag_(iConfig.getParameter<edm::InputTag>("diamondLocalTrackInputTag")),
      usePixel_(iConfig.getParameter<bool>("usePixel")),
      useStrip_(iConfig.getParameter<bool>("useStrip")),
      useDiamond_(iConfig.getParameter<bool>("useDiamond")),
      minTracks_(iConfig.getParameter<int>("minTracks")),
      minTracksPerArm_(iConfig.getParameter<int>("minTracksPerArm")),
      maxTracks_(iConfig.getParameter<int>("maxTracks")),
      maxTracksPerArm_(iConfig.getParameter<int>("maxTracksPerArm")),
      maxTracksPerPot_(iConfig.getParameter<int>("maxTracksPerPot")) {
  if (usePixel_)
    pixelLocalTrackToken_ = consumes<edm::DetSetVector<CTPPSPixelLocalTrack>>(pixelLocalTrackInputTag_);
  if (useStrip_)
    stripLocalTrackToken_ = consumes<edm::DetSetVector<TotemRPLocalTrack>>(stripLocalTrackInputTag_);
  if (useDiamond_)
    diamondLocalTrackToken_ = consumes<edm::DetSetVector<CTPPSDiamondLocalTrack>>(diamondLocalTrackInputTag_);

  LogDebug("") << "HLTCTPPSLocalTrackFilter: "
                  "pixelTag/stripTag/diamondTag/usePixel/useStrip/useDiamond/minTracks/minTracksPerArm/maxTracks/"
                  "maxTracksPerArm/maxTracksPerPot : "
               << pixelLocalTrackInputTag_.encode() << " " << stripLocalTrackInputTag_.encode() << " "
               << diamondLocalTrackInputTag_.encode() << " " << usePixel_ << " " << useStrip_ << " " << useDiamond_
               << " " << minTracks_ << " " << minTracksPerArm_ << " " << maxTracks_ << " " << maxTracksPerArm_ << " "
               << maxTracksPerPot_;
}

//
// member functions
//
bool HLTCTPPSLocalTrackFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  int arm45Tracks = 0;
  int arm56Tracks = 0;
  std::map<uint32_t, int> tracksPerPot;

  //   Note that there is no matching between the tracks from the several roman pots
  // so tracks from separate pots might correspond to the same particle.
  // When the pixels are used in more than one RP (in 2018), then the same situation can
  // happen within the pixels themselves.
  if (usePixel_)  // Pixels correspond to RP 220 in 2017 data
  {
    edm::Handle<edm::DetSetVector<CTPPSPixelLocalTrack>> pixelTracks;
    iEvent.getByToken(pixelLocalTrackToken_, pixelTracks);

    for (const auto& rpv : (*pixelTracks)) {
      const CTPPSPixelDetId id(rpv.id);
      if (tracksPerPot.count(rpv.id) == 0)
        tracksPerPot[rpv.id] = 0;

      for (auto& track : rpv) {
        if (track.isValid()) {
          if (id.arm() == 0)
            ++arm45Tracks;
          if (id.arm() == 1)
            ++arm56Tracks;
          ++tracksPerPot[rpv.id];
        }
      }
    }
  }

  if (useStrip_)  // Strips correspond to RP 210 in 2017 data
  {
    edm::Handle<edm::DetSetVector<TotemRPLocalTrack>> stripTracks;
    iEvent.getByToken(stripLocalTrackToken_, stripTracks);

    for (const auto& rpv : (*stripTracks)) {
      const TotemRPDetId id(rpv.id);
      if (tracksPerPot.count(rpv.id) == 0)
        tracksPerPot[rpv.id] = 0;

      for (auto& track : rpv) {
        if (track.isValid()) {
          if (id.arm() == 0)
            ++arm45Tracks;
          if (id.arm() == 1)
            ++arm56Tracks;
          ++tracksPerPot[rpv.id];
        }
      }
    }
  }

  if (useDiamond_) {
    edm::Handle<edm::DetSetVector<CTPPSDiamondLocalTrack>> diamondTracks;
    iEvent.getByToken(diamondLocalTrackToken_, diamondTracks);

    for (const auto& rpv : (*diamondTracks)) {
      const CTPPSDiamondDetId id(rpv.id);
      if (tracksPerPot.count(rpv.id) == 0)
        tracksPerPot[rpv.id] = 0;

      for (auto& track : rpv) {
        if (track.isValid()) {
          if (id.arm() == 0)
            ++arm45Tracks;
          if (id.arm() == 1)
            ++arm56Tracks;
          ++tracksPerPot[rpv.id];
        }
      }
    }
  }

  if (arm45Tracks + arm56Tracks < minTracks_ || arm45Tracks < minTracksPerArm_ || arm56Tracks < minTracksPerArm_)
    return false;

  if (maxTracks_ >= minTracks_ && arm45Tracks + arm56Tracks > maxTracks_)
    return false;

  if (maxTracksPerArm_ >= minTracksPerArm_ && (arm45Tracks > maxTracksPerArm_ || arm56Tracks > maxTracksPerArm_))
    return false;

  if (maxTracksPerPot_ >= 0) {
    for (auto& pot : tracksPerPot) {
      if (pot.second > maxTracksPerPot_) {
        return false;
      }
    }
  }

  return true;
}

// define as a framework module
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTCTPPSLocalTrackFilter);
