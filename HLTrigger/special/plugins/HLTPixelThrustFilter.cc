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
  const double min_thrust_;  // minimum thrust
  const double max_thrust_;  // maximum thrust
};

//
// constructors and destructor
//

HLTPixelThrustFilter::HLTPixelThrustFilter(const edm::ParameterSet& config)
    : trackerGeometryRcdToken_(esConsumes()),
      inputToken_(consumes<edmNew::DetSetVector<SiPixelCluster> >(config.getParameter<edm::InputTag>("inputTag"))),
      min_thrust_(config.getParameter<double>("minThrust")),
      max_thrust_(config.getParameter<double>("maxThrust")) {}

void HLTPixelThrustFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTag", edm::InputTag("hltSiPixelClusters"));
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

  std::vector<reco::LeafCandidate> vec;
  for (auto DSViter = clusters.begin(); DSViter != clusters.end(); DSViter++) {
    auto const& pgdu = static_cast<const PixelGeomDetUnit*>(trackerGeo.idToDetUnit(DSViter->detId()));
    for (auto const& cluster : *DSViter) {
      auto const& pos = pgdu->surface().toGlobal(pgdu->specificTopology().localPosition({cluster.x(), cluster.y()}));
      auto const mag = std::sqrt(pos.x() * pos.x() + pos.y() * pos.y());
      vec.emplace_back(0, reco::Particle::LorentzVector(pos.x() / mag, pos.y() / mag, 0, 0));
    }
  }
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
