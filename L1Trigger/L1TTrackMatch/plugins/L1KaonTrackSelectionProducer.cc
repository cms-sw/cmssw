// -*- C++ -*-
//
// Package:    L1Trigger/L1TTrackMatch
// Class:      L1KaonTrackSelectionProducer
//
/**\class L1KaonTrackSelectionProducer L1KaonTrackSelectionProducer.cc L1Trigger/L1TTrackMatch/plugins/L1KaonTrackSelectionProducer.cc

 Description: Selects positively and negatively charged L1Tracks which pass the Level 1 track selection criteria into two separate collections 

 Implementation:
     Inputs:
         std::vector<TTTrack> - Each floating point TTTrack inside this collection inherits from
                                a bit-accurate TTTrack_TrackWord, used for emulation purposes.
     Outputs:
         std::vector<TTTrack> - Positively and negatively charged collection of TTTracks selected from cuts on the TTTrack properties
         std::vector<TTTrack> - Positively and negatively charged collection of TTTracks selected from cuts on the TTTrack_TrackWord properties
*/
//----------------------------------------------------------------------------
// Authors: Alexx Perloff, Pritam Palit (original version, 2021),
//          Sweta Baradia, Suchandra Dutta, Subir Sarkar (February 2025)
//----------------------------------------------------------------------------
// system include files
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

// Xilinx HLS includes
#include <ap_fixed.h>
#include <ap_int.h>

// user include files
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1Trigger/interface/Vertex.h"
#include "DataFormats/L1Trigger/interface/VertexWord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "CommonTools/Utils/interface/AndSelector.h"
#include "CommonTools/Utils/interface/EtaRangeSelector.h"
#include "CommonTools/Utils/interface/MinSelector.h"
#include "CommonTools/Utils/interface/MinFunctionSelector.h"
#include "CommonTools/Utils/interface/MinNumberSelector.h"
#include "CommonTools/Utils/interface/PtMinSelector.h"
#include "CommonTools/Utils/interface/Selection.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "L1TrackWordUnpacker.h"

//
// class declaration
//
class L1KaonTrackSelectionProducer : public edm::global::EDProducer<> {
public:
  explicit L1KaonTrackSelectionProducer(const edm::ParameterSet&);
  ~L1KaonTrackSelectionProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // ----------constants, enums and typedefs ---------
  using L1Track = TTTrack<Ref_Phase2TrackerDigi_>;
  using TTTrackCollection = std::vector<L1Track>;
  using TTTrackRef = edm::Ref<TTTrackCollection>;
  using TTTrackRefCollection = edm::RefVector<TTTrackCollection>;
  using TTTrackCollectionHandle = edm::Handle<TTTrackRefCollection>;
  using TTTrackRefCollectionUPtr = std::unique_ptr<TTTrackRefCollection>;

  // ----------meprintDebugInfomber functions ----------------------
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------selectors -----------------------------
  // Based on recommendations from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGenericSelectors
  // Track charge selection

  bool TTTrackChargeSelector(const L1Track& t) const { return std::signbit(t.rInv()); };

  bool TTTrackWordChargeSelector(const L1Track& t) const {
    ap_uint<1> chargeEmulationBits = t.getTrackWord()(TTTrack_TrackWord::TrackBitLocations::kRinvMSB,
                                                      TTTrack_TrackWord::TrackBitLocations::kRinvMSB);
    return chargeEmulationBits.to_uint();
  };

  // ----------member data ---------------------------
  const edm::EDGetTokenT<TTTrackRefCollection> l1TracksToken_;
  const std::string outputCollectionName_;
  bool processSimulatedTracks_, processEmulatedTracks_;
  int debug_;
};

//
// constructors and destructor
//
L1KaonTrackSelectionProducer::L1KaonTrackSelectionProducer(const edm::ParameterSet& iConfig)
    : l1TracksToken_(consumes<TTTrackRefCollection>(iConfig.getParameter<edm::InputTag>("l1TracksInputTag"))),
      outputCollectionName_(iConfig.getParameter<std::string>("outputCollectionName")),
      processSimulatedTracks_(iConfig.getParameter<bool>("processSimulatedTracks")),
      processEmulatedTracks_(iConfig.getParameter<bool>("processEmulatedTracks")),
      debug_(iConfig.getParameter<int>("debug")) {
  // Ensure that the configuration makes sense
  if (!processSimulatedTracks_ && !processEmulatedTracks_) {
    throw cms::Exception("You must process at least one of the track collections (simulated or emulated).");
  }

  // Get additional input tags and define the EDM output based on the previous configuration parameters
  if (processSimulatedTracks_) {
    produces<TTTrackRefCollection>(outputCollectionName_ + "Positivecharge");
    produces<TTTrackRefCollection>(outputCollectionName_ + "Negativecharge");
  }
  if (processEmulatedTracks_) {
    produces<TTTrackRefCollection>(outputCollectionName_ + "EmulationPositivecharge");
    produces<TTTrackRefCollection>(outputCollectionName_ + "EmulationNegativecharge");
  }
}

L1KaonTrackSelectionProducer::~L1KaonTrackSelectionProducer() {}

//
// member functions
//
// ------------ method called to produce the data  ------------
void L1KaonTrackSelectionProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto vTTPosTrackOutput = std::make_unique<TTTrackRefCollection>();
  auto vTTPosTrackEmulationOutput = std::make_unique<TTTrackRefCollection>();

  auto vTTNegTrackOutput = std::make_unique<TTTrackRefCollection>();
  auto vTTNegTrackEmulationOutput = std::make_unique<TTTrackRefCollection>();

  TTTrackCollectionHandle l1TracksHandle;
  iEvent.getByToken(l1TracksToken_, l1TracksHandle);
  size_t nTracks = l1TracksHandle->size();
  if (!nTracks)
    return;

  if (processSimulatedTracks_) {
    vTTPosTrackOutput->reserve(static_cast<std::size_t>(std::round(nTracks * 0.6)));
    vTTNegTrackOutput->reserve(static_cast<std::size_t>(std::round(nTracks * 0.6)));
  }
  if (processEmulatedTracks_) {
    vTTPosTrackEmulationOutput->reserve(static_cast<std::size_t>(std::round(nTracks * 0.6)));
    vTTNegTrackEmulationOutput->reserve(static_cast<std::size_t>(std::round(nTracks * 0.6)));
  }

  for (size_t i = 0; i < nTracks; i++) {
    const auto& trackRef = l1TracksHandle->at(i);
    const auto& track = *trackRef;

    // Select tracks based on the floating point TTTrack
    if (processSimulatedTracks_) {
      (!TTTrackChargeSelector(track) ? vTTPosTrackOutput->push_back(trackRef) : vTTNegTrackOutput->push_back(trackRef));
    }
    // Select tracks based on the bitwise accurate TTTrack_TrackWord
    if (processEmulatedTracks_) {
      (!TTTrackWordChargeSelector(track) ? vTTPosTrackEmulationOutput->push_back(trackRef)
                                         : vTTNegTrackEmulationOutput->push_back(trackRef));
    }
  }
  // Put the outputs into the event
  if (processSimulatedTracks_) {
    iEvent.put(std::move(vTTPosTrackOutput), outputCollectionName_ + "Positivecharge");
    iEvent.put(std::move(vTTNegTrackOutput), outputCollectionName_ + "Negativecharge");
  }
  if (processEmulatedTracks_) {
    iEvent.put(std::move(vTTPosTrackEmulationOutput), outputCollectionName_ + "EmulationPositivecharge");
    iEvent.put(std::move(vTTNegTrackEmulationOutput), outputCollectionName_ + "EmulationNegativecharge");
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1KaonTrackSelectionProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //L1KaonTrackSelectionProducer
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("l1TracksInputTag", edm::InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"));
  desc.add<std::string>("outputCollectionName", "Level1TTKaonTracksSelected");
  desc.add<bool>("processSimulatedTracks", true)
      ->setComment("return selected tracks after cutting on the floating point values");
  desc.add<bool>("processEmulatedTracks", true)
      ->setComment("return selected tracks after cutting on the bitwise emulated values");
  desc.add<int>("debug", 4)->setComment("Verbosity levels: 0, 1, 2, 3");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1KaonTrackSelectionProducer);
