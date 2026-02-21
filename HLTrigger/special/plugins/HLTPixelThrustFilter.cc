#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "PhysicsTools/CandUtils/interface/Thrust.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationForHLTService.h"

//
// class declaration
//

class HLTPixelThrustFilter : public edm::stream::EDFilter<> {
public:
  explicit HLTPixelThrustFilter(const edm::ParameterSet&);
  ~HLTPixelThrustFilter() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool filter(edm::Event&, edm::EventSetup const&) final;

private:
  bool isSaturatedPixel(const SiPixelCluster&, const DetId&) const;
  std::unique_ptr<SiPixelGainCalibrationForHLTService> getPixelCalib(const edm::ParameterSet& conf, edm::ConsumesCollector iC) const {
    if (min_nSatPixels_ > 0 || useSaturatedPixels_)
      return std::make_unique<SiPixelGainCalibrationForHLTService>(conf, iC);
    return nullptr;
  };
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryRcdToken_;
  const edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > inputToken_;
  const bool useSaturatedPixels_;
  const unsigned int min_nPixels_, min_nSatPixels_;
  const std::unique_ptr<SiPixelGainCalibrationForHLTService> pixelCalib_;
  const double min_thrust_;  // minimum thrust
  const double max_thrust_;  // maximum thrust
};

//
// constructors and destructor
//

HLTPixelThrustFilter::HLTPixelThrustFilter(const edm::ParameterSet& config)
    : trackerGeometryRcdToken_(esConsumes()),
      inputToken_(consumes<edmNew::DetSetVector<SiPixelCluster> >(config.getParameter<edm::InputTag>("inputTag"))),
      useSaturatedPixels_(config.getParameter<bool>("useSaturatedPixels")),
      min_nPixels_(config.getParameter<unsigned int>("minNPixels")),
      min_nSatPixels_(config.getParameter<unsigned int>("minNSaturatedPixels")),
      pixelCalib_(getPixelCalib(config, consumesCollector())),
      min_thrust_(config.getParameter<double>("minThrust")),
      max_thrust_(config.getParameter<double>("maxThrust")) {}

void HLTPixelThrustFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTag", edm::InputTag("hltSiPixelClusters"));
  desc.add<bool>("useSaturatedPixels", false);
  desc.add<unsigned int>("minNPixels", 2);
  desc.add<unsigned int>("minNSaturatedPixels", 0);
  desc.add<double>("minThrust", 0);
  desc.add<double>("maxThrust", 0);
  descriptions.add("hltPixelThrustFilter", desc);
}

//
// member functions
//
// ------------ method called to produce the data  ------------
bool HLTPixelThrustFilter::filter(edm::Event& event, edm::EventSetup const& iSetup) {
  // get hold of products from Event
  auto const& clusters = event.get(inputToken_);
  auto const& trackerGeo = iSetup.getData(trackerGeometryRcdToken_);
  if (pixelCalib_)
    pixelCalib_->setESObjects(iSetup);

  size_t nSatPixels(0);
  std::vector<reco::LeafCandidate> vec;
  for (auto DSViter = clusters.begin(); DSViter != clusters.end(); DSViter++) {
    auto const& pgdu = static_cast<const PixelGeomDetUnit*>(trackerGeo.idToDetUnit(DSViter->detId()));
    for (auto const& cluster : *DSViter) {
      if (pixelCalib_ && isSaturatedPixel(cluster, DSViter->detId()))
        nSatPixels += 1;
      else if (useSaturatedPixels_)
        continue;
      auto const& pos = pgdu->surface().toGlobal(pgdu->specificTopology().localPosition({cluster.x(), cluster.y()}));
      auto const mag = std::sqrt(pos.x() * pos.x() + pos.y() * pos.y());
      if (mag > 0)
        vec.emplace_back(0, reco::Particle::LorentzVector(pos.x() / mag, pos.y() / mag, 0, 0));
    }
  }
  if (vec.size() < min_nPixels_ || nSatPixels < min_nSatPixels_)
    return false;
  auto const thrust = Thrust(vec.begin(), vec.end()).thrust();

  bool accept = (thrust >= min_thrust_);
  if (max_thrust_ > 0)
    accept &= (thrust <= max_thrust_);
  // return with final filter decision
  return accept;
}

bool HLTPixelThrustFilter::isSaturatedPixel(const SiPixelCluster& cluster, const DetId& detId) const {
  for (size_t j = 0; j < cluster.pixelADC().size(); j++) {
    const auto& pixel = cluster.pixel(j);
    if (pixel.adc == std::numeric_limits<uint16_t>::max())
      return true;
    // Run3: VCaltoElectronOffset = 0 and VCaltoElectronGain = 1
    const auto& vcal = pixel.adc;
    const auto DBgain = pixelCalib_->getGain(detId, pixel.y, pixel.x);
    const auto DBpedestal = pixelCalib_->getPedestal(detId, pixel.y, pixel.x);
    const auto adc = DBgain > 0. ? std::round(DBpedestal + vcal / DBgain) : 0;
    if (adc >= std::numeric_limits<uint8_t>::max())
      return true;
  }
  return false;
}

// define as a framework module
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTPixelThrustFilter);
