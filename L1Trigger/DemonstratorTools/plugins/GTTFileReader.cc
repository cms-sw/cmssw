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
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1Track;
  typedef std::vector<L1Track> TTTrackCollection;

  // ----------member functions ----------------------
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  const bool processOutputToCorrelator_;
  const bool processInputTracks_;
  const bool processOutputToGlobalTrigger_;
  const size_t kEmptyFramesOutputToCorrelator_;
  const size_t kEmptyFramesInputTracks_;
  const size_t kEmptyFramesOutputToGlobalTrigger_;
  l1t::demo::BoardDataReader fileReaderOutputToCorrelator_;
  std::string l1VertexCollectionName_;
  l1t::demo::BoardDataReader fileReaderInputTracks_;
  std::string l1TrackCollectionName_;
  l1t::demo::BoardDataReader fileReaderOutputToGlobalTrigger_;
};

GTTFileReader::GTTFileReader(const edm::ParameterSet& iConfig)
    : processOutputToCorrelator_(iConfig.getParameter<bool>("processOutputToCorrelator")),
      processInputTracks_(iConfig.getParameter<bool>("processInputTracks")),
      processOutputToGlobalTrigger_(iConfig.getParameter<bool>("processOutputToGlobalTrigger")),
      kEmptyFramesOutputToCorrelator_(iConfig.getUntrackedParameter<unsigned int>("kEmptyFramesOutputToCorrelator")),
      kEmptyFramesInputTracks_(iConfig.getUntrackedParameter<unsigned int>("kEmptyFramesInputTracks")),
      kEmptyFramesOutputToGlobalTrigger_(
          iConfig.getUntrackedParameter<unsigned int>("kEmptyFramesOutputToGlobalTrigger")),
      l1VertexCollectionName_(iConfig.getParameter<std::string>("l1VertexCollectionName")),
      l1TrackCollectionName_(iConfig.getParameter<std::string>("l1TrackCollectionName")) {
  if (processOutputToCorrelator_) {
    fileReaderOutputToCorrelator_ =
        l1t::demo::BoardDataReader(l1t::demo::parseFileFormat(iConfig.getUntrackedParameter<std::string>("format")),
                                   iConfig.getParameter<std::vector<std::string>>("filesOutputToCorrelator"),
                                   l1t::demo::kFramesPerTMUXPeriod,
                                   l1t::demo::kGTTBoardTMUX,
                                   kEmptyFramesOutputToCorrelator_,
                                   l1t::demo::kChannelMapOutputToCorrelator);
    produces<l1t::VertexWordCollection>(l1VertexCollectionName_);
  }
  if (processInputTracks_) {
    fileReaderInputTracks_ =
        l1t::demo::BoardDataReader(l1t::demo::parseFileFormat(iConfig.getUntrackedParameter<std::string>("format")),
                                   iConfig.getParameter<std::vector<std::string>>("filesInputTracks"),
                                   l1t::demo::kFramesPerTMUXPeriod,
                                   l1t::demo::kGTTBoardTMUX,
                                   kEmptyFramesInputTracks_,
                                   l1t::demo::kChannelMapInput);
    produces<TTTrackCollection>(l1TrackCollectionName_);
  }
  if (processOutputToGlobalTrigger_) {
    // fileReaderOutputToGlobalTrigger_ =
    //   l1t::demo::BoardDataReader(l1t::demo::parseFileFormat(iConfig.getUntrackedParameter<std::string>("format")),
    // 				 iConfig.getParameter<std::vector<std::string>>("filesOutputToGlobalTrigger"),
    // 				 l1t::demo::kFramesPerTMUXPeriod,
    // 				 l1t::demo::kGTTBoardTMUX,
    // 				 kEmptyFramesOutputToGlobalTrigger_,
    // 				 l1t::demo::kChannelMapInput);
    throw std::invalid_argument("Processing OutputToGlobalTrigger files has not been fully implemented and validated.");
    // need to produce output collections for Prompt and Displaced Jets, HTMiss, ETMiss, Taus, Mesons, Vertices, and Isolated Tracks
  }
}

// ------------ method called to produce the data  ------------
void GTTFileReader::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace l1t::demo::codecs;
  if (processOutputToCorrelator_) {
    l1t::demo::EventData correlatorEventData(fileReaderOutputToCorrelator_.getNextEvent());
    l1t::VertexWordCollection vertices(decodeVertices(correlatorEventData.at({"vertices", 0})));
    edm::LogInfo("GTTFileReader") << vertices.size() << " vertices found";

    iEvent.put(std::make_unique<l1t::VertexWordCollection>(vertices), l1VertexCollectionName_);
  }  // end if ( processOutputToCorrelator_ )

  if (processInputTracks_) {
    l1t::demo::EventData inputEventData(fileReaderInputTracks_.getNextEvent());
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
      }  //end loop over trackwoards
    }    // end loop over GTT input links
    iEvent.put(std::move(inputTracks), l1TrackCollectionName_);
  }  // end if ( processInputTracks_ )
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void GTTFileReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // GTTFileReader
  edm::ParameterSetDescription desc;
  desc.add<bool>("processOutputToCorrelator")
      ->setComment("boolean flag to load Correlator outputs via BoardDataReader and produce vertex collection");
  desc.add<bool>("processInputTracks")
      ->setComment("boolean flag to load track inputs via BoardDataReader and produce a TTTrack collection");
  desc.add<bool>("processOutputToGlobalTrigger")
      ->setComment(
          "boolean flag to load Global Trigger outputs via BoardDataReader and produce Track Object collections");
  desc.addUntracked<unsigned int>("kEmptyFramesOutputToCorrelator", 0)
      ->setComment("empty frames to expect in OutputToCorrelator");
  desc.addUntracked<unsigned int>("kEmptyFramesInputTracks", 0)->setComment("empty frames to expect in Track Input");
  desc.addUntracked<unsigned int>("kEmptyFramesOutputToGlobalTrigger", 0)
      ->setComment("empty frames to expect in OutputToGlobalTrigger");
  desc.add<std::vector<std::string>>("filesOutputToCorrelator",
                                     {
                                         "L1GTTOutputToCorrelator_0.txt",
                                     });
  desc.add<std::vector<std::string>>("filesInputTracks",
                                     {
                                         "L1GTTInputFile_0.txt",
                                     });
  desc.add<std::vector<std::string>>("filesOutputToGlobalTrigger",
                                     {
                                         "L1GTTOutputToGlobalTriggerFile_0.txt",
                                     });
  desc.addUntracked<std::string>("format", "APx");
  desc.add<std::string>("l1VertexCollectionName", "L1VerticesFirmware");
  desc.add<std::string>("l1TrackCollectionName", "Level1TTTracks");
  descriptions.add("GTTFileReader", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GTTFileReader);
