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

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/L1Trigger/interface/Vertex.h"
#include "L1Trigger/DemonstratorTools/interface/BoardDataReader.h"
#include "L1Trigger/DemonstratorTools/interface/codecs/vertices.h"
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
  static constexpr size_t kFramesPerTMUXPeriod = 9;
  static constexpr size_t kGapLength = 44;
  static constexpr size_t kVertexTMUX = 6;
  static constexpr size_t kVertexChanIndex = 0;
  static constexpr size_t kEmptyFrames = 10;

  const l1t::demo::BoardDataReader::ChannelMap_t kChannelSpecs = {
      /* logical channel within time slice -> {{link TMUX, inter-packet gap}, vector of channel indices} */
      {{"vertices", 0}, {{kVertexTMUX, kGapLength}, {kVertexChanIndex}}}};

  // ----------member functions ----------------------
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  l1t::demo::BoardDataReader fileReader_;
};

//
// class implementation
//

GTTFileReader::GTTFileReader(const edm::ParameterSet& iConfig)
    : fileReader_(l1t::demo::parseFileFormat(iConfig.getUntrackedParameter<std::string>("format")),
                  iConfig.getParameter<std::vector<std::string>>("files"),
                  kFramesPerTMUXPeriod,
                  kVertexTMUX,
                  kEmptyFrames,
                  kChannelSpecs) {
  produces<l1t::VertexWordCollection>();
}

// ------------ method called to produce the data  ------------
void GTTFileReader::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace l1t::demo::codecs;

  l1t::demo::EventData eventData(fileReader_.getNextEvent());

  l1t::VertexWordCollection vertices(decodeVertices(eventData.at({"vertices", 0})));

  std::cout << vertices.size() << " vertices found" << std::endl;

  iEvent.put(std::make_unique<l1t::VertexWordCollection>(vertices));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void GTTFileReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GTTFileReader);
