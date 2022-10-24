// -*- C++ -*-
//
// Package:    L1Trigger/DemonstratorTools
// Class:      GTTFileWriter
//
/**\class GTTFileWriter GTTFileWriter.cc L1Trigger/DemonstratorTools/plugins/GTTFileWriter.cc

 Description: Example EDAnalyzer class, illustrating how BoardDataWriter can be used to
   write I/O buffer files for hardware/firmware tests

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Thomas Williams <thomas.williams@stfc.ac.uk>
//         Created:  Mon, 15 Feb 2021 00:39:44 GMT
//
//

// system include files
#include <memory>

#include "ap_int.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/VertexWord.h"

#include "L1Trigger/DemonstratorTools/interface/BoardDataWriter.h"
#include "L1Trigger/DemonstratorTools/interface/codecs/tracks.h"
#include "L1Trigger/DemonstratorTools/interface/codecs/vertices.h"
#include "L1Trigger/DemonstratorTools/interface/codecs/tkjets.h"
#include "L1Trigger/DemonstratorTools/interface/codecs/htsums.h"
#include "L1Trigger/DemonstratorTools/interface/codecs/etsums.h"
#include "L1Trigger/DemonstratorTools/interface/utilities.h"

//
// class declaration
//

class GTTFileWriter : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit GTTFileWriter(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // ----------constants, enums and typedefs ---------
  // NOTE: At least some of the info from these constants will eventually come from config files
  static constexpr size_t kFramesPerTMUXPeriod = 9;
  static constexpr size_t kGapLengthInput = 6;
  static constexpr size_t kGapLengthOutputToCorrelator = 44;
  static constexpr size_t kGapLengthOutputToGlobalTriggerSums = 3;
  static constexpr size_t kGapLengthOutputToGlobalTriggerTaus = 36;
  static constexpr size_t kGapLengthOutputToGlobalTriggerMesons = 15;
  static constexpr size_t kGapLengthOutputToGlobalTriggerVertices = 6;
  static constexpr size_t kTrackTMUX = 18;
  static constexpr size_t kGTTBoardTMUX = 6;
  static constexpr size_t kMaxLinesPerFile = 1024;

  const std::map<l1t::demo::LinkId, std::vector<size_t>> kChannelIdsInput = {
      /* logical channel within time slice -> vector of channel indices (one entry per time slice) */
      {{"tracks", 0}, {0, 18, 36}},
      {{"tracks", 1}, {1, 19, 37}},
      {{"tracks", 2}, {2, 20, 38}},
      {{"tracks", 3}, {3, 21, 39}},
      {{"tracks", 4}, {4, 22, 40}},
      {{"tracks", 5}, {5, 23, 41}},
      {{"tracks", 6}, {6, 24, 42}},
      {{"tracks", 7}, {7, 25, 43}},
      {{"tracks", 8}, {8, 26, 44}},
      {{"tracks", 9}, {9, 27, 45}},
      {{"tracks", 10}, {10, 28, 46}},
      {{"tracks", 11}, {11, 29, 47}},
      {{"tracks", 12}, {12, 30, 48}},
      {{"tracks", 13}, {13, 31, 49}},
      {{"tracks", 14}, {14, 32, 50}},
      {{"tracks", 15}, {15, 33, 51}},
      {{"tracks", 16}, {16, 34, 52}},
      {{"tracks", 17}, {17, 35, 53}}};

  const std::map<std::string, l1t::demo::ChannelSpec> kChannelSpecsInput = {
      /* interface name -> {link TMUX, inter-packet gap} */
      {"tracks", {kTrackTMUX, kGapLengthInput}}};

  const std::map<l1t::demo::LinkId, std::pair<l1t::demo::ChannelSpec, std::vector<size_t>>>
      kChannelSpecsOutputToCorrelator = {
          /* logical channel within time slice -> {{link TMUX, inter-packet gap}, vector of channel indices} */
          {{"vertices", 0}, {{kGTTBoardTMUX, kGapLengthOutputToCorrelator}, {0}}}};

  const std::map<l1t::demo::LinkId, std::vector<size_t>> kChannelIdsOutputToGlobalTrigger = {
      /* logical channel within time slice -> vector of channel indices (one entry per time slice) */
      {{"sums", 0}, {0}},
      {{"taus", 1}, {1}},
      {{"mesons", 2}, {2}},
      {{"vertices", 3}, {3}}};

  const std::map<std::string, l1t::demo::ChannelSpec> kChannelSpecsOutputToGlobalTrigger = {
      /* interface name -> {link TMUX, inter-packet gap} */
      {"sums", {kGTTBoardTMUX, kGapLengthOutputToGlobalTriggerSums}},
      {"taus", {kGTTBoardTMUX, kGapLengthOutputToGlobalTriggerTaus}},
      {"mesons", {kGTTBoardTMUX, kGapLengthOutputToGlobalTriggerMesons}},
      {"vertices", {kGTTBoardTMUX, kGapLengthOutputToGlobalTriggerVertices}}};

  typedef TTTrack<Ref_Phase2TrackerDigi_> Track_t;
  typedef std::vector<Track_t> TrackCollection_t;
  typedef edm::RefVector<TrackCollection_t> TrackRefCollection_t;

  // ----------member functions ----------------------
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<edm::View<Track_t>> tracksToken_;
  const edm::EDGetTokenT<edm::View<Track_t>> convertedTracksToken_;
  const edm::EDGetTokenT<TrackCollection_t> convertedTrackCollectionToken_;
  const edm::EDGetTokenT<TrackRefCollection_t> selectedTracksToken_;
  const edm::EDGetTokenT<TrackRefCollection_t> vertexAssociatedTracksToken_;
  const edm::EDGetTokenT<edm::View<l1t::VertexWord>> verticesToken_;
  const edm::EDGetTokenT<edm::View<l1t::TkJetWord>> jetsToken_;
  const edm::EDGetTokenT<edm::View<l1t::EtSum>> htMissToken_;
  const edm::EDGetTokenT<edm::View<l1t::EtSum>> etMissToken_;

  l1t::demo::BoardDataWriter fileWriterInputTracks_;
  l1t::demo::BoardDataWriter fileWriterConvertedTracks_;
  l1t::demo::BoardDataWriter fileWriterSelectedTracks_;
  l1t::demo::BoardDataWriter fileWriterVertexAssociatedTracks_;
  l1t::demo::BoardDataWriter fileWriterOutputToCorrelator_;
  l1t::demo::BoardDataWriter fileWriterOutputToGlobalTrigger_;
};

//
// class implementation
//

GTTFileWriter::GTTFileWriter(const edm::ParameterSet& iConfig)
    : tracksToken_(consumes<edm::View<Track_t>>(iConfig.getUntrackedParameter<edm::InputTag>("tracks"))),
      convertedTracksToken_(
          consumes<edm::View<Track_t>>(iConfig.getUntrackedParameter<edm::InputTag>("convertedTracks"))),
      convertedTrackCollectionToken_(
          consumes<TrackCollection_t>(iConfig.getUntrackedParameter<edm::InputTag>("convertedTracks"))),
      selectedTracksToken_(
          consumes<TrackRefCollection_t>(iConfig.getUntrackedParameter<edm::InputTag>("selectedTracks"))),
      vertexAssociatedTracksToken_(
          consumes<TrackRefCollection_t>(iConfig.getUntrackedParameter<edm::InputTag>("vertexAssociatedTracks"))),
      verticesToken_(consumes<edm::View<l1t::VertexWord>>(iConfig.getUntrackedParameter<edm::InputTag>("vertices"))),
      jetsToken_(consumes<edm::View<l1t::TkJetWord>>(iConfig.getUntrackedParameter<edm::InputTag>("jets"))),
      htMissToken_(consumes<edm::View<l1t::EtSum>>(iConfig.getUntrackedParameter<edm::InputTag>("htmiss"))),
      etMissToken_(consumes<edm::View<l1t::EtSum>>(iConfig.getUntrackedParameter<edm::InputTag>("etmiss"))),
      fileWriterInputTracks_(l1t::demo::parseFileFormat(iConfig.getUntrackedParameter<std::string>("format")),
                             iConfig.getUntrackedParameter<std::string>("inputFilename"),
                             kFramesPerTMUXPeriod,
                             kGTTBoardTMUX,
                             kMaxLinesPerFile,
                             kChannelIdsInput,
                             kChannelSpecsInput),
      fileWriterConvertedTracks_(l1t::demo::parseFileFormat(iConfig.getUntrackedParameter<std::string>("format")),
                                 iConfig.getUntrackedParameter<std::string>("inputConvertedFilename"),
                                 kFramesPerTMUXPeriod,
                                 kGTTBoardTMUX,
                                 kMaxLinesPerFile,
                                 kChannelIdsInput,
                                 kChannelSpecsInput),
      fileWriterSelectedTracks_(l1t::demo::parseFileFormat(iConfig.getUntrackedParameter<std::string>("format")),
                                iConfig.getUntrackedParameter<std::string>("selectedTracksFilename"),
                                kFramesPerTMUXPeriod,
                                kGTTBoardTMUX,
                                kMaxLinesPerFile,
                                kChannelIdsInput,
                                kChannelSpecsInput),
      fileWriterVertexAssociatedTracks_(
          l1t::demo::parseFileFormat(iConfig.getUntrackedParameter<std::string>("format")),
          iConfig.getUntrackedParameter<std::string>("vertexAssociatedTracksFilename"),
          kFramesPerTMUXPeriod,
          kGTTBoardTMUX,
          kMaxLinesPerFile,
          kChannelIdsInput,
          kChannelSpecsInput),
      fileWriterOutputToCorrelator_(l1t::demo::parseFileFormat(iConfig.getUntrackedParameter<std::string>("format")),
                                    iConfig.getUntrackedParameter<std::string>("outputCorrelatorFilename"),
                                    kFramesPerTMUXPeriod,
                                    kGTTBoardTMUX,
                                    kMaxLinesPerFile,
                                    kChannelSpecsOutputToCorrelator),
      fileWriterOutputToGlobalTrigger_(l1t::demo::parseFileFormat(iConfig.getUntrackedParameter<std::string>("format")),
                                       iConfig.getUntrackedParameter<std::string>("outputGlobalTriggerFilename"),
                                       kFramesPerTMUXPeriod,
                                       kGTTBoardTMUX,
                                       kMaxLinesPerFile,
                                       kChannelIdsOutputToGlobalTrigger,
                                       kChannelSpecsOutputToGlobalTrigger) {}

void GTTFileWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace l1t::demo::codecs;

  // 0) Gather the necessary collections
  const auto& tracksCollection = iEvent.get(tracksToken_);
  const auto& convertedTracksCollection = iEvent.get(convertedTracksToken_);
  const auto& verticesCollection = iEvent.get(verticesToken_);
  const auto& jetsCollection = iEvent.get(jetsToken_);
  const auto& htMissCollection = iEvent.get(htMissToken_);
  const auto& etMissCollection = iEvent.get(etMissToken_);

  edm::Handle<TrackCollection_t> convertedTracksHandle;
  edm::Handle<TrackRefCollection_t> selectedTracksHandle;
  edm::Handle<TrackRefCollection_t> vertexAssociatedTracksHandle;
  iEvent.getByToken(convertedTrackCollectionToken_, convertedTracksHandle);
  iEvent.getByToken(selectedTracksToken_, selectedTracksHandle);
  iEvent.getByToken(vertexAssociatedTracksToken_, vertexAssociatedTracksHandle);

  // 1) Encode 'object' information onto vectors containing link data
  const auto trackData(encodeTracks(tracksCollection));
  const auto convertedTrackData(encodeTracks(convertedTracksCollection));
  const auto selectedTrackData(encodeTracks(convertedTracksHandle, selectedTracksHandle));
  const auto vertexAssociatedTrackData(encodeTracks(convertedTracksHandle, vertexAssociatedTracksHandle));
  const auto vertexData(encodeVertices(verticesCollection));
  const auto jetsData(encodeTkJets(jetsCollection));
  const auto htMissData(encodeHtSums(htMissCollection));
  const auto etMissData(encodeEtSums(etMissCollection));

  // 2) Pack 'object' information into 'event data' object
  l1t::demo::EventData eventDataTracks;
  l1t::demo::EventData eventDataConvertedTracks;
  l1t::demo::EventData eventDataSelectedTracks;
  l1t::demo::EventData eventDataVertexAssociatedTracks;
  for (size_t i = 0; i < 18; i++) {
    eventDataTracks.add({"tracks", i}, trackData.at(i));
    eventDataConvertedTracks.add({"tracks", i}, convertedTrackData.at(i));
    eventDataSelectedTracks.add({"tracks", i}, selectedTrackData.at(i));
    eventDataVertexAssociatedTracks.add({"tracks", i}, vertexAssociatedTrackData.at(i));
  }

  l1t::demo::EventData eventDataVertices;
  eventDataVertices.add({"vertices", 0}, vertexData.at(0));

  // 2b) For the global trigger 'event data' combine different objects into one 'logical' link
  std::vector<ap_uint<64>> sumsData;
  sumsData.insert(sumsData.end(), jetsData.at(0).begin(), jetsData.at(0).end());
  sumsData.insert(sumsData.end(), 24, 0);
  sumsData.insert(sumsData.end(), htMissData.at(0).begin(), htMissData.at(0).end());
  sumsData.insert(sumsData.end(), 1, 0);
  sumsData.insert(sumsData.end(), etMissData.at(0).begin(), etMissData.at(0).end());

  std::vector<ap_uint<64>> tracksVerticesData;
  tracksVerticesData.insert(tracksVerticesData.end(), 36, 0);
  tracksVerticesData.insert(tracksVerticesData.end(), vertexData.at(0).begin(), vertexData.at(0).end());
  tracksVerticesData.insert(tracksVerticesData.end(), 2, 0);

  l1t::demo::EventData eventDataGlobalTrigger;
  eventDataGlobalTrigger.add({"sums", 0}, sumsData);
  eventDataGlobalTrigger.add({"taus", 1}, std::vector<ap_uint<64>>(18, 0));  // Placeholder until tau object is written
  eventDataGlobalTrigger.add({"mesons", 2},
                             std::vector<ap_uint<64>>(39, 0));  // Placeholder until light meson objects are written
  eventDataGlobalTrigger.add({"vertices", 3}, tracksVerticesData);

  // 3) Pass the 'event data' object to the file writer

  fileWriterInputTracks_.addEvent(eventDataTracks);
  fileWriterConvertedTracks_.addEvent(eventDataConvertedTracks);
  fileWriterSelectedTracks_.addEvent(eventDataSelectedTracks);
  fileWriterVertexAssociatedTracks_.addEvent(eventDataVertexAssociatedTracks);
  fileWriterOutputToCorrelator_.addEvent(eventDataVertices);
  fileWriterOutputToGlobalTrigger_.addEvent(eventDataGlobalTrigger);
}

// ------------ method called once each job just after ending the event loop  ------------
void GTTFileWriter::endJob() {
  // Writing pending events to file before exiting
  fileWriterInputTracks_.flush();
  fileWriterConvertedTracks_.flush();
  fileWriterOutputToCorrelator_.flush();
  fileWriterOutputToGlobalTrigger_.flush();
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void GTTFileWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // GTTFileWriter
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>("tracks", edm::InputTag("l1tTTTracksFromTrackletEmulation", "Level1TTTracks"));
  desc.addUntracked<edm::InputTag>("convertedTracks", edm::InputTag("l1tGTTInputProducer", "Level1TTTracksConverted"));
  desc.addUntracked<edm::InputTag>("selectedTracks",
                                   edm::InputTag("l1tTrackSelectionProducer", "Level1TTTracksSelectedEmulation"));
  desc.addUntracked<edm::InputTag>(
      "vertexAssociatedTracks",
      edm::InputTag("l1tTrackSelectionProducer", "Level1TTTracksSelectedAssociatedEmulation"));
  desc.addUntracked<edm::InputTag>("vertices", edm::InputTag("l1tVertexProducer", "l1verticesEmulation"));
  desc.addUntracked<edm::InputTag>("jets", edm::InputTag("l1tTrackJetsEmulation", "L1TrackJets"));
  desc.addUntracked<edm::InputTag>("htmiss", edm::InputTag("l1tTrackerEmuHTMiss", "L1TrackerEmuHTMiss"));
  desc.addUntracked<edm::InputTag>("etmiss", edm::InputTag("l1tTrackerEmuEtMiss", "L1TrackerEmuEtMiss"));
  desc.addUntracked<std::string>("inputFilename", "L1GTTInputFile");
  desc.addUntracked<std::string>("inputConvertedFilename", "L1GTTInputConvertedFile");
  desc.addUntracked<std::string>("selectedTracksFilename", "L1GTTSelectedTracksFile");
  desc.addUntracked<std::string>("vertexAssociatedTracksFilename", "L1GTTVertexAssociatedTracksFile");
  desc.addUntracked<std::string>("outputCorrelatorFilename", "L1GTTOutputToCorrelatorFile");
  desc.addUntracked<std::string>("outputGlobalTriggerFilename", "L1GTTOutputToGlobalTriggerFile");
  desc.addUntracked<std::string>("format", "APx");
  descriptions.add("GTTFileWriter", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GTTFileWriter);
