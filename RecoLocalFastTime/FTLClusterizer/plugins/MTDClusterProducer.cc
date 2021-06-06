//---------------------------------------------------------------------------
//! \class MTDClusterProducer
//!
//! \brief EDProducer to cluster FTLRecHits into FTLClusters.
//!
//---------------------------------------------------------------------------
// Our own stuff
#include "RecoLocalFastTime/FTLClusterizer/interface/MTDThresholdClusterizer.h"
#include "RecoLocalFastTime/FTLClusterizer/interface/MTDClusterizerBase.h"

// Data Formats
#include "DataFormats/FTLRecHit/interface/FTLRecHit.h"

// STL
#include <vector>
#include <memory>
#include <string>
#include <iostream>

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"

class MTDClusterProducer : public edm::stream::EDProducer<> {
public:
  //--- Constructor, virtual destructor (just in case)
  explicit MTDClusterProducer(const edm::ParameterSet& conf);
  ~MTDClusterProducer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  //--- The top-level event method.
  void produce(edm::Event& e, const edm::EventSetup& c) override;

  //--- Execute the algorithm(s).
  template <typename T>
  void run(const T& input, FTLClusterCollection& output);

private:
  const edm::EDGetTokenT<FTLRecHitCollection> btlHits_;
  const edm::EDGetTokenT<FTLRecHitCollection> etlHits_;

  const std::string ftlbInstance_;  // instance name of barrel clusters
  const std::string ftleInstance_;  // instance name of endcap clusters

  const std::string clusterMode_;                    // user's choice of the clusterizer
  std::unique_ptr<MTDClusterizerBase> clusterizer_;  // what we got (for now, one ptr to base class)

  const MTDGeometry* geom_;
  const MTDTopology* topo_;
  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;
};

//---------------------------------------------------------------------------
//!  Constructor: set the ParameterSet and defer all thinking to setupClusterizer().
//---------------------------------------------------------------------------
MTDClusterProducer::MTDClusterProducer(edm::ParameterSet const& conf)
    : btlHits_(consumes<FTLRecHitCollection>(conf.getParameter<edm::InputTag>("srcBarrel"))),
      etlHits_(consumes<FTLRecHitCollection>(conf.getParameter<edm::InputTag>("srcEndcap"))),
      ftlbInstance_(conf.getParameter<std::string>("BarrelClusterName")),
      ftleInstance_(conf.getParameter<std::string>("EndcapClusterName")),
      clusterMode_(conf.getParameter<std::string>("ClusterMode")) {
  //--- Declare to the EDM what kind of collections we will be making.
  produces<FTLClusterCollection>(ftlbInstance_);
  produces<FTLClusterCollection>(ftleInstance_);

  mtdgeoToken_ = esConsumes<MTDGeometry, MTDDigiGeometryRecord>();
  mtdtopoToken_ = esConsumes<MTDTopology, MTDTopologyRcd>();

  //--- Make the algorithm(s) according to what the user specified
  //--- in the ParameterSet.
  if (clusterMode_ == "MTDThresholdClusterizer") {
    clusterizer_ = std::make_unique<MTDThresholdClusterizer>(conf);
  } else {
    throw cms::Exception("MTDClusterProducer") << "[MTDClusterProducer]:"
                                               << " choice " << clusterMode_ << " is invalid.\n"
                                               << "Possible choices:\n"
                                               << "    MTDThresholdClusterizer";
  }
}

// Configuration descriptions
void MTDClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcBarrel", edm::InputTag("mtdRecHits:FTLBarrel"));
  desc.add<edm::InputTag>("srcEndcap", edm::InputTag("mtdRecHits:FTLEndcap"));
  desc.add<std::string>("BarrelClusterName", "FTLBarrel");
  desc.add<std::string>("EndcapClusterName", "FTLEndcap");
  desc.add<std::string>("ClusterMode", "MTDThresholdClusterizer");
  MTDThresholdClusterizer::fillPSetDescription(desc);
  descriptions.add("mtdClusterProducer", desc);
}

//---------------------------------------------------------------------------
//! The "Event" entrypoint: gets called by framework for every event
//---------------------------------------------------------------------------
void MTDClusterProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  // Step A.1: get input data
  edm::Handle<FTLRecHitCollection> inputBarrel;
  edm::Handle<FTLRecHitCollection> inputEndcap;
  e.getByToken(btlHits_, inputBarrel);
  e.getByToken(etlHits_, inputEndcap);

  // Step A.2: get event setup
  auto geom = es.getTransientHandle(mtdgeoToken_);
  geom_ = geom.product();

  auto mtdTopo = es.getTransientHandle(mtdtopoToken_);
  topo_ = mtdTopo.product();

  // Step B: create the final output collection
  auto outputBarrel = std::make_unique<FTLClusterCollection>();
  auto outputEndcap = std::make_unique<FTLClusterCollection>();

  run(*inputBarrel, *outputBarrel);
  run(*inputEndcap, *outputEndcap);

  e.put(std::move(outputBarrel), ftlbInstance_);
  e.put(std::move(outputEndcap), ftleInstance_);
}

//---------------------------------------------------------------------------
//!  Iterate over DetUnits, and invoke the PixelClusterizer on each.
//---------------------------------------------------------------------------
template <typename T>
void MTDClusterProducer::run(const T& input, FTLClusterCollection& output) {
  if (!clusterizer_) {
    throw cms::Exception("MTDClusterProducer") << " at least one clusterizer is not ready -- can't run!";
  }

  clusterizer_->clusterize(input, geom_, topo_, output);

  LogDebug("MTDClusterProducer") << " Executing " << clusterMode_ << " resulted in " << output.size()
                                 << " MTDClusters for " << input.size() << " Hits.";
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MTDClusterProducer);
