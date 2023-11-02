// -*- C++ -*-
//
// Package:    L1Trigger/DemonstratorTools
// Class:      GTTFileReader
//
/**\class GTTFileReader GTTFileReader.cc L1Trigger/DemonstratorTools/plugins/GTTFileReader.cc

 Description: Example EDProducer class, illustrating how BoardDataReader can be used to
   read I/O buffer files (that have been created in hardware/firmware tests), decode
   the contained data, and store this in EDM collections.

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Thomas Williams <thomas.williams@stfc.ac.uk>
//         Created:  Fri, 19 Feb 2021 01:10:55 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/L1Trigger/interface/Vertex.h"
#include "L1Trigger/DemonstratorTools/interface/GTTInterface.h"
#include "L1Trigger/DemonstratorTools/interface/BoardDataReader.h"
#include "L1Trigger/DemonstratorTools/interface/codecs/vertices.h"
#include "L1Trigger/DemonstratorTools/interface/codecs/tracks.h"
#include "L1Trigger/DemonstratorTools/interface/utilities.h"

//
// class declaration
//

class GTTFileReader : public edm::stream::EDProducer<> {
public:
  explicit GTTFileReader(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // ----------constants, enums and typedefs ---------
  // NOTE: At least some of the info from these constants will eventually come from config files
  // static constexpr size_t kFramesPerTMUXPeriod = 9;
  // static constexpr size_t kGapLength = 44;
  // static constexpr size_t kGapLengthInput = 6;
  // static constexpr size_t kVertexTMUX = 6;
  // static constexpr size_t kGTTBoardTMUX = 6;
  // static constexpr size_t kTrackTMUX = 18;
  // static constexpr size_t kVertexChanIndex = 0;
  static constexpr size_t kEmptyFramesOutputToCorrelator = 0;  // 10 does not match current file writing configuration
  static constexpr size_t kEmptyFramesInput = 0;
  static constexpr size_t kEmptyFramesOutputToGlobalTrigger = 0;

  // const l1t::demo::BoardDataReader::ChannelMap_t kChannelSpecs = {
  //     /* logical channel within time slice -> {{link TMUX, inter-packet gap}, vector of channel indices} */
  //     {{"vertices", 0}, {{kGTTBoardTMUX, kGapLengthOutputToCorrelator}, {kVertexChanIndex}}}};

  // const l1t::demo::BoardDataReader::ChannelMap_t kChannelSpecsInput = {
  //     /* logical channel within time slice -> {{link TMUX, inter-packet gap}, vector of channel indices} */
  //     {{"tracks", 0}, {{kTrackTMUX, kGapLengthInput}, {0, 18, 36}}},
  //     {{"tracks", 1}, {{kTrackTMUX, kGapLengthInput}, {1, 19, 37}}},
  //     {{"tracks", 2}, {{kTrackTMUX, kGapLengthInput}, {2, 20, 38}}},
  //     {{"tracks", 3}, {{kTrackTMUX, kGapLengthInput}, {3, 21, 39}}},
  //     {{"tracks", 4}, {{kTrackTMUX, kGapLengthInput}, {4, 22, 40}}},
  //     {{"tracks", 5}, {{kTrackTMUX, kGapLengthInput}, {5, 23, 41}}},
  //     {{"tracks", 6}, {{kTrackTMUX, kGapLengthInput}, {6, 24, 42}}},
  //     {{"tracks", 7}, {{kTrackTMUX, kGapLengthInput}, {7, 25, 43}}},
  //     {{"tracks", 8}, {{kTrackTMUX, kGapLengthInput}, {8, 26, 44}}},
  //     {{"tracks", 9}, {{kTrackTMUX, kGapLengthInput}, {9, 27, 45}}},
  //     {{"tracks", 10}, {{kTrackTMUX, kGapLengthInput}, {10, 28, 46}}},
  //     {{"tracks", 11}, {{kTrackTMUX, kGapLengthInput}, {11, 29, 47}}},
  //     {{"tracks", 12}, {{kTrackTMUX, kGapLengthInput}, {12, 30, 48}}},
  //     {{"tracks", 13}, {{kTrackTMUX, kGapLengthInput}, {13, 31, 49}}},
  //     {{"tracks", 14}, {{kTrackTMUX, kGapLengthInput}, {14, 32, 50}}},
  //     {{"tracks", 15}, {{kTrackTMUX, kGapLengthInput}, {15, 33, 51}}},
  //     {{"tracks", 16}, {{kTrackTMUX, kGapLengthInput}, {16, 34, 52}}},
  //     {{"tracks", 17}, {{kTrackTMUX, kGapLengthInput}, {17, 35, 53}}}};

  typedef TTTrack<Ref_Phase2TrackerDigi_> L1Track;
  typedef std::vector<L1Track> TTTrackCollection;

  // ----------member functions ----------------------
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  l1t::demo::BoardDataReader fileReaderOutputToCorrelator_;
  std::string l1VertexCollectionName_;
  l1t::demo::BoardDataReader fileReaderInputTracks_;
  std::string l1TrackCollectionName_;
};

//
// class implementation
//

GTTFileReader::GTTFileReader(const edm::ParameterSet& iConfig)
    : fileReaderOutputToCorrelator_(l1t::demo::parseFileFormat(iConfig.getUntrackedParameter<std::string>("format")),
                                    iConfig.getParameter<std::vector<std::string>>("files"),
                                    l1t::demo::kFramesPerTMUXPeriod,
                                    l1t::demo::kGTTBoardTMUX,
                                    kEmptyFramesOutputToCorrelator,
                                    l1t::demo::kChannelMapOutputToCorrelator),
      l1VertexCollectionName_(iConfig.getParameter<std::string>("l1VertexCollectionName")),
      fileReaderInputTracks_(l1t::demo::parseFileFormat(iConfig.getUntrackedParameter<std::string>("format")),
                             iConfig.getParameter<std::vector<std::string>>("filesInputTracks"),
                             l1t::demo::kFramesPerTMUXPeriod,
                             l1t::demo::kGTTBoardTMUX,
                             kEmptyFramesInput,
                             l1t::demo::kChannelMapInput),
      l1TrackCollectionName_(iConfig.getParameter<std::string>("l1TrackCollectionName")) {
  produces<l1t::VertexWordCollection>(l1VertexCollectionName_);
  produces<TTTrackCollection>(l1TrackCollectionName_);
}

// ------------ method called to produce the data  ------------
void GTTFileReader::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace l1t::demo::codecs;

  l1t::demo::EventData correlatorEventData(fileReaderOutputToCorrelator_.getNextEvent());
  l1t::demo::EventData inputEventData(fileReaderInputTracks_.getNextEvent());

  l1t::VertexWordCollection vertices(decodeVertices(correlatorEventData.at({"vertices", 0})));
  auto inputTracks = std::make_unique<TTTrackCollection>();
  for (size_t i = 0; i < 18; i++) {
    auto iTracks = decodeTracks(inputEventData.at({"tracks", i}));
    for (auto& trackword : iTracks) {
      if (!trackword.getValidWord())
        continue;
      L1Track track = L1Track(trackword.getValidWord(),
                              trackword.getRinvWord(),
                              trackword.getPhiWord(),
                              trackword.getTanlWord(),
                              trackword.getZ0Word(),
                              trackword.getD0Word(),
                              trackword.getChi2RPhiWord(),
                              trackword.getChi2RZWord(),
                              trackword.getBendChi2Word(),
                              trackword.getHitPatternWord(),
                              trackword.getMVAQualityWord(),
                              trackword.getMVAOtherWord());
      //retrieve the eta (first) and phi (second) sectors for GTT, encoded in an std::pair
      auto sectors = (l1t::demo::codecs::sectorsEtaPhiFromGTTLinkID(i));
      track.setEtaSector(sectors.first);
      track.setPhiSector(sectors.second);
      track.trackWord_ = trackword.trackWord_;
      inputTracks->push_back(track);
    }
  }

  edm::LogInfo("GTTFileReader") << vertices.size() << " vertices found";

  iEvent.put(std::make_unique<l1t::VertexWordCollection>(vertices), l1VertexCollectionName_);
  iEvent.put(std::move(inputTracks), l1TrackCollectionName_);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void GTTFileReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // GTTFileReader
  edm::ParameterSetDescription desc;
  desc.add<std::vector<std::string>>("files",
                                     {
                                         "L1GTTOutputToCorrelator_0.txt",
                                     });
  desc.add<std::string>("l1VertexCollectionName", "L1VerticesFirmware");
  desc.add<std::vector<std::string>>("filesInputTracks",
                                     {
                                         "L1GTTInputFile_0.txt",
                                     });
  desc.add<std::string>("l1TrackCollectionName", "Level1TTTracks");
  desc.addUntracked<std::string>("format", "APx");
  descriptions.add("GTTFileReader", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GTTFileReader);
