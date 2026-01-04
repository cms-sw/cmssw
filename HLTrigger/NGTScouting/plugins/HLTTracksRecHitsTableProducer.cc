/**  \class HLTTracksRecHitsTableProducer
 *
 *   \brief Produces a nanoAOD flat table with recHits information for HLT tracks
 *
 *   This producer creates a nanoAOD flat table containing the recHits global positions and errors
 *   starting from a collection of reco::Track.
 *   This data can be added as an extension to the HLTTracks table.
 *   The maximum number of recHits per track is fixed to a configurable value;
 *   if a track has more recHits, a warning is issued and the extra recHits are ignored.
 *
 *   \author Luca Ferragina (INFN BO), 2025
 */

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/StreamID.h"

class HLTTracksRecHitsTableProducer : public edm::stream::EDProducer<> {
public:
  explicit HLTTracksRecHitsTableProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<std::vector<reco::Track>> tracks_;

  const std::string tableName_;
  const unsigned int maxRecHits_;
  const unsigned int precision_;
  const bool skipNonExistingSrc_;
};

HLTTracksRecHitsTableProducer::HLTTracksRecHitsTableProducer(const edm::ParameterSet& params)
    : tracks_(consumes<std::vector<reco::Track>>(params.getParameter<edm::InputTag>("tracksSrc"))),
      tableName_(params.getParameter<std::string>("tableName")),
      maxRecHits_(params.getParameter<uint>("maxRecHits")),
      precision_(params.getParameter<int>("precision")),
      skipNonExistingSrc_(params.getParameter<bool>("skipNonExistingSrc")) {
  produces<nanoaod::FlatTable>(tableName_);
}

void HLTTracksRecHitsTableProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto tracksIn = iEvent.getHandle(tracks_);
  const size_t nTracks = tracksIn.isValid() ? (*tracksIn).size() : 0;

  static constexpr float default_value = std::numeric_limits<float>::quiet_NaN();

  std::vector<float> globalX(maxRecHits_ * nTracks, default_value);
  std::vector<float> globalY(maxRecHits_ * nTracks, default_value);
  std::vector<float> globalZ(maxRecHits_ * nTracks, default_value);
  std::vector<float> globalErrX(maxRecHits_ * nTracks, default_value);
  std::vector<float> globalErrY(maxRecHits_ * nTracks, default_value);
  std::vector<float> globalErrZ(maxRecHits_ * nTracks, default_value);

  if (tracksIn.isValid() || !(this->skipNonExistingSrc_)) {
    const auto& tracks = *tracksIn;
    for (size_t tkIndex = 0; tkIndex < nTracks; ++tkIndex) {
      const auto& track = tracks[tkIndex];
      for (auto it = track.recHitsBegin(); it != track.recHitsEnd(); ++it) {
        auto hit = *it;
        auto globalPoint = hit->globalPosition();
        auto globalError = hit->globalPositionError();
        auto hitIndex = std::distance(track.recHitsBegin(), it);
        if (hitIndex >= maxRecHits_) {
          edm::LogWarning("HLTTracksRecHitsTableProducer")
              << " Track " << tkIndex << " has more (" << track.recHitsSize() << ") than " << maxRecHits_
              << " recHits, skipping the rest.";
          break;
        }
        globalX[tkIndex * maxRecHits_ + hitIndex] = globalPoint.x();
        globalY[tkIndex * maxRecHits_ + hitIndex] = globalPoint.y();
        globalZ[tkIndex * maxRecHits_ + hitIndex] = globalPoint.z();
        globalErrX[tkIndex * maxRecHits_ + hitIndex] = globalError.cxx();
        globalErrY[tkIndex * maxRecHits_ + hitIndex] = globalError.cyy();
        globalErrZ[tkIndex * maxRecHits_ + hitIndex] = globalError.czz();
      }
    }
  } else {
    edm::LogWarning("HLTTracksRecHitsTableProducer")
        << " Invalid handle for " << tableName_ << " in tracks input collection";
  }

  assert(globalX.size() == globalY.size() && globalX.size() == globalZ.size() && globalX.size() == globalErrX.size() &&
         globalX.size() == globalErrY.size() && globalX.size() == globalErrZ.size());

  // Table for tracks recHits
  auto tracksTable =
      std::make_unique<nanoaod::FlatTable>(nTracks * maxRecHits_, tableName_, /*singleton*/ false, /*extension*/ false);
  tracksTable->addColumn<float>("globalX", globalX, "RecHits global x coordinate", precision_);
  tracksTable->addColumn<float>("globalY", globalY, "RecHits global y coordinate", precision_);
  tracksTable->addColumn<float>("globalZ", globalZ, "RecHits global z coordinate", precision_);
  tracksTable->addColumn<float>("globalErrX", globalErrX, "RecHits global x error", precision_);
  tracksTable->addColumn<float>("globalErrY", globalErrY, "RecHits global y error", precision_);
  tracksTable->addColumn<float>("globalErrZ", globalErrZ, "RecHits global z error", precision_);

  iEvent.put(std::move(tracksTable), tableName_);
}

void HLTTracksRecHitsTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<uint>("maxRecHits", 16)->setComment("maximum number of recHits per track to be stored in the table");
  desc.add<bool>("skipNonExistingSrc", false)
      ->setComment("whether or not to skip producing the table on absent input product");
  desc.add<std::string>("tableName")->setComment("name of the flat table ouput");
  desc.add<edm::InputTag>("tracksSrc")->setComment("std::vector<reco::Track> input collection");
  desc.add<int>("precision", 7);
  descriptions.addWithDefaultLabel(desc);
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTTracksRecHitsTableProducer);
