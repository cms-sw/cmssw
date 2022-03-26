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

#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1Trigger/interface/VertexWord.h"
#include "DataFormats/Common/interface/View.h"

#include "L1Trigger/DemonstratorTools/interface/BoardDataWriter.h"
#include "L1Trigger/DemonstratorTools/interface/codecs/tracks.h"
#include "L1Trigger/DemonstratorTools/interface/codecs/vertices.h"
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
  static constexpr size_t kGapLengthOutput = 44;
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
          {{"vertices", 0}, {{kGTTBoardTMUX, kGapLengthOutput}, {0}}}};

  typedef TTTrack<Ref_Phase2TrackerDigi_> Track_t;

  // ----------member functions ----------------------
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<edm::View<Track_t>> tracksToken_;
  edm::EDGetTokenT<edm::View<Track_t>> convertedTracksToken_;
  edm::EDGetTokenT<edm::View<l1t::VertexWord>> verticesToken_;

  l1t::demo::BoardDataWriter fileWriterInputTracks_;
  l1t::demo::BoardDataWriter fileWriterConvertedTracks_;
  l1t::demo::BoardDataWriter fileWriterOutputToCorrelator_;
};

//
// class implementation
//

GTTFileWriter::GTTFileWriter(const edm::ParameterSet& iConfig)
    : tracksToken_(consumes<edm::View<Track_t>>(iConfig.getUntrackedParameter<edm::InputTag>("tracks"))),
      convertedTracksToken_(
          consumes<edm::View<Track_t>>(iConfig.getUntrackedParameter<edm::InputTag>("convertedTracks"))),
      verticesToken_(consumes<edm::View<l1t::VertexWord>>(iConfig.getUntrackedParameter<edm::InputTag>("vertices"))),
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
      fileWriterOutputToCorrelator_(l1t::demo::parseFileFormat(iConfig.getUntrackedParameter<std::string>("format")),
                                    iConfig.getUntrackedParameter<std::string>("outputFilename"),
                                    kFramesPerTMUXPeriod,
                                    kGTTBoardTMUX,
                                    kMaxLinesPerFile,
                                    kChannelSpecsOutputToCorrelator) {}

void GTTFileWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace l1t::demo::codecs;

  // 1) Encode track information onto vectors containing link data
  const auto trackData(encodeTracks(iEvent.get(tracksToken_)));
  const auto convertedTrackData(encodeTracks(iEvent.get(convertedTracksToken_)));
  const auto outputData(encodeVertices(iEvent.get(verticesToken_)));

  // 2) Pack track information into 'event data' object, and pass that to file writer
  l1t::demo::EventData eventDataTracks;
  l1t::demo::EventData eventDataConvertedTracks;
  for (size_t i = 0; i < 18; i++) {
    eventDataTracks.add({"tracks", i}, trackData.at(i));
    eventDataConvertedTracks.add({"tracks", i}, convertedTrackData.at(i));
  }

  l1t::demo::EventData eventDataVertices;
  eventDataVertices.add({"vertices", 0}, outputData.at(0));

  fileWriterInputTracks_.addEvent(eventDataTracks);
  fileWriterConvertedTracks_.addEvent(eventDataConvertedTracks);
  fileWriterOutputToCorrelator_.addEvent(eventDataVertices);
}

// ------------ method called once each job just after ending the event loop  ------------
void GTTFileWriter::endJob() {
  // Writing pending events to file before exiting
  fileWriterInputTracks_.flush();
  fileWriterConvertedTracks_.flush();
  fileWriterOutputToCorrelator_.flush();
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void GTTFileWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);

  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  //ParameterSetDescription desc;
  //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
  //descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GTTFileWriter);
