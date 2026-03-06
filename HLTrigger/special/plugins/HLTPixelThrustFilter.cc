#include "FWCore/Framework/interface/global/EDFilter.h"
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

//
// class declaration
//

class HLTPixelThrustFilter : public edm::global::EDFilter<> {
public:
  explicit HLTPixelThrustFilter(const edm::ParameterSet&);
  ~HLTPixelThrustFilter() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const final;

private:
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryRcdToken_;
  const edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > inputToken_;
  const bool useOnlySaturatedPixels_;
  const unsigned int min_nPixels_, max_nPixels_;
  const unsigned int min_nSatPixels_, max_nSatPixels_;
  const double min_thrust_;  // minimum thrust
  const double max_thrust_;  // maximum thrust
};

//
// constructors and destructor
//

HLTPixelThrustFilter::HLTPixelThrustFilter(const edm::ParameterSet& config)
    : trackerGeometryRcdToken_(esConsumes()),
      inputToken_(consumes<edmNew::DetSetVector<SiPixelCluster> >(config.getParameter<edm::InputTag>("inputTag"))),
      useOnlySaturatedPixels_(config.getParameter<bool>("useOnlySaturatedPixels")),
      min_nPixels_(config.getParameter<unsigned int>("minNPixels")),
      max_nPixels_(config.getParameter<unsigned int>("maxNPixels")),
      min_nSatPixels_(config.getParameter<unsigned int>("minNSaturatedPixels")),
      max_nSatPixels_(config.getParameter<unsigned int>("maxNSaturatedPixels")),
      min_thrust_(config.getParameter<double>("minThrust")),
      max_thrust_(config.getParameter<double>("maxThrust")) {}

void HLTPixelThrustFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTag", edm::InputTag("hltSiPixelClusters"));
  desc.add<bool>("useOnlySaturatedPixels", false);
  desc.add<unsigned int>("minNPixels", 2);
  desc.add<unsigned int>("maxNPixels", 0);
  desc.add<unsigned int>("minNSaturatedPixels", 0);
  desc.add<unsigned int>("maxNSaturatedPixels", 0);
  desc.add<double>("minThrust", 0);
  desc.add<double>("maxThrust", 0);
  descriptions.add("hltPixelThrustFilter", desc);
}

//
// member functions
//
// ------------ method called to produce the data  ------------
bool HLTPixelThrustFilter::filter(edm::StreamID, edm::Event& event, edm::EventSetup const& iSetup) const {
  // get hold of products from Event
  auto const& clusters = event.get(inputToken_);
  auto const& trackerGeo = iSetup.getData(trackerGeometryRcdToken_);

  size_t nPixels(0), nSatPixels(0);
  for (auto const& dsv : clusters)
    for (auto const& cluster : dsv) {
      nPixels += 1;
      nSatPixels += cluster.isSaturated();
    }
  if (nPixels < min_nPixels_ || nSatPixels < min_nSatPixels_)
    return false;
  if ((max_nPixels_ > 0 && nPixels > max_nPixels_) || (max_nSatPixels_ > 0 && nSatPixels > max_nSatPixels_))
    return false;

  std::vector<reco::LeafCandidate> vec;
  vec.reserve(nPixels);
  for (auto const& dsv : clusters) {
    auto const& pgdu = static_cast<const PixelGeomDetUnit*>(trackerGeo.idToDetUnit(dsv.detId()));
    for (auto const& cluster : dsv) {
      if (useOnlySaturatedPixels_ && not cluster.isSaturated())
        continue;
      auto const& pos = pgdu->surface().toGlobal(pgdu->specificTopology().localPosition({cluster.x(), cluster.y()}));
      if (auto mag = pos.perp())
        vec.emplace_back(0, reco::Particle::LorentzVector(pos.x() / mag, pos.y() / mag, 0, 0));
    }
  }
  if (vec.size() < min_nPixels_)
    return false;
  auto const thrust = Thrust(vec.begin(), vec.end()).thrust();

  bool accept = (thrust >= min_thrust_);
  if (max_thrust_ > 0)
    accept &= (thrust <= max_thrust_);
  // return with final filter decision
  return accept;
}

// define as a framework module
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTPixelThrustFilter);
