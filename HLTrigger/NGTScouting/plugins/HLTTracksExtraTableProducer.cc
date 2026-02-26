#include <memory>

// user include files
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

//
// class declaration
//

class HLTTracksExtraTableProducer : public edm::stream::EDProducer<> {
public:
  explicit HLTTracksExtraTableProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  const bool skipNonExistingSrc_;
  const std::string tableName_;
  const unsigned int precision_;
  const edm::EDGetTokenT<std::vector<reco::Track>> tracks_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpot_;
};

//
// constructors
//

HLTTracksExtraTableProducer::HLTTracksExtraTableProducer(const edm::ParameterSet& params)
    : skipNonExistingSrc_(params.getParameter<bool>("skipNonExistingSrc")),
      tableName_(params.getParameter<std::string>("tableName")),
      precision_(params.getParameter<int>("precision")),
      tracks_(consumes<std::vector<reco::Track>>(params.getParameter<edm::InputTag>("tracksSrc"))),
      beamSpot_(consumes<reco::BeamSpot>(params.getParameter<edm::InputTag>("beamSpot"))) {
  produces<nanoaod::FlatTable>(tableName_);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void HLTTracksExtraTableProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  //vertex collection
  auto tracksIn = iEvent.getHandle(tracks_);
  auto beamSpotIn = iEvent.getHandle(beamSpot_);
  const size_t nTracks = tracksIn.isValid() ? (*tracksIn).size() : 0;

  static constexpr float default_value = std::numeric_limits<float>::quiet_NaN();

  // initialize to quiet Nans
  std::vector<float> v_dxyBS(nTracks, default_value);
  std::vector<float> v_dzBS(nTracks, default_value);

  if ((tracksIn.isValid() && beamSpotIn.isValid()) || !(this->skipNonExistingSrc_)) {
    const auto& tracks = *tracksIn;
    const auto& beamSpot = *beamSpotIn;
    math::XYZPoint point(beamSpot.x0(), beamSpot.y0(), beamSpot.z0());
    for (size_t tk_index = 0; tk_index < nTracks; ++tk_index) {
      v_dxyBS[tk_index] = tracks[tk_index].dxy(point);
      v_dzBS[tk_index] = tracks[tk_index].dz(point);
    }
  } else {
    edm::LogWarning("HLTTracksExtraTableProducer")
        << " Invalid handle for " << tableName_ << " in tracks input collection";
  }

  //table for all primary vertices
  auto tracksTable = std::make_unique<nanoaod::FlatTable>(nTracks, tableName_, /*singleton*/ false, /*extension*/ true);
  tracksTable->addColumn<float>("dzBS", v_dzBS, "tracks dz() w.r.t Beam Spot", precision_);
  tracksTable->addColumn<float>("dxyBS", v_dxyBS, "tracks dxy() w.r.t Beam Spot", precision_);
  iEvent.put(std::move(tracksTable), tableName_);
}

// ------------ fill 'descriptions' with the allowed parameters for the module ------------
void HLTTracksExtraTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<bool>("skipNonExistingSrc", false)
      ->setComment("whether or not to skip producing the table on absent input product");
  desc.add<std::string>("tableName")->setComment("name of the flat table ouput");
  desc.add<edm::InputTag>("tracksSrc")->setComment("std::vector<reco::Track> input collections");
  desc.add<edm::InputTag>("beamSpot")->setComment("input BeamSpot collection");
  desc.add<int>("precision", 7);
  descriptions.addWithDefaultLabel(desc);
}

// ------------ define this as a plug-in ------------
DEFINE_FWK_MODULE(HLTTracksExtraTableProducer);
