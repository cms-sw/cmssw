/** SiPixelClusterProducer.cc
 * ---------------------------------------------------------------
 * Description:  see SiPixelClusterProducer.h
 * Author:  P. Maksimovic (porting from original ORCA version)
 * History: Oct 14, 2005, initial version
 * Get rid of the noiseVector. d.k. 28/3/06
 * Implementation of the DetSetVector container.    V.Chiochia, May 06
 * SiPixelClusterCollection typedef of DetSetVector V.Chiochia, June 06
 * Introduce the DetSet local container (cache) for speed. d.k. 05/07
 * 
 * ---------------------------------------------------------------
 */

// Our own stuff
#include "SiPixelClusterProducer.h"
#include "PixelThresholdClusterizer.h"

// Geometry
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

// Data Formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/DetId/interface/DetId.h"

// Database payloads
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationService.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationOfflineService.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationForHLTService.h"

// Framework
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// STL
#include <vector>
#include <memory>
#include <string>
#include <iostream>

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//---------------------------------------------------------------------------
//!  Constructor: set the ParameterSet and defer all thinking to setupClusterizer().
//---------------------------------------------------------------------------
SiPixelClusterProducer::SiPixelClusterProducer(edm::ParameterSet const& conf)
    : tPutPixelClusters(produces<SiPixelClusterCollectionNew>()),
      clusterMode_(conf.getParameter<std::string>("ClusterMode")),
      maxTotalClusters_(conf.getParameter<int32_t>("maxNumberOfClusters")) {
  if (clusterMode_ == "PixelThresholdReclusterizer")
    tPixelClusters = consumes<SiPixelClusterCollectionNew>(conf.getParameter<edm::InputTag>("src"));
  else
    tPixelDigi = consumes<edm::DetSetVector<PixelDigi>>(conf.getParameter<edm::InputTag>("src"));

  trackerTopoToken_ = esConsumes<TrackerTopology, TrackerTopologyRcd>();
  trackerGeomToken_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();

  const auto& payloadType = conf.getParameter<std::string>("payloadType");
  if (payloadType == "HLT")
    theSiPixelGainCalibration_ = std::make_unique<SiPixelGainCalibrationForHLTService>(conf);
  else if (payloadType == "Offline")
    theSiPixelGainCalibration_ = std::make_unique<SiPixelGainCalibrationOfflineService>(conf);
  else if (payloadType == "Full")
    theSiPixelGainCalibration_ = std::make_unique<SiPixelGainCalibrationService>(conf);

  //--- Make the algorithm(s) according to what the user specified
  //--- in the ParameterSet.
  setupClusterizer(conf);
}

// Destructor
SiPixelClusterProducer::~SiPixelClusterProducer() = default;

void SiPixelClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("siPixelDigis"));
  desc.add<std::string>("ClusterMode", "PixelThresholdClusterizer");
  desc.add<int>("maxNumberOfClusters", -1)->setComment("-1 means no limit");
  desc.add<std::string>("payloadType", "Offline")
      ->setComment("Options: HLT - column granularity, Offline - gain:col/ped:pix");

  PixelThresholdClusterizer::fillPSetDescription(desc);
  SiPixelGainCalibrationServiceBase::fillPSetDescription(desc);  // no-op, but in principle the structures are there...

  descriptions.add("SiPixelClusterizerDefault", desc);
}

//---------------------------------------------------------------------------
//! The "Event" entrypoint: gets called by framework for every event
//---------------------------------------------------------------------------
void SiPixelClusterProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  //Setup gain calibration service
  theSiPixelGainCalibration_->setESObjects(es);

  // Step A.1: get input data
  edm::Handle<SiPixelClusterCollectionNew> inputClusters;
  edm::Handle<edm::DetSetVector<PixelDigi>> inputDigi;
  if (clusterMode_ == "PixelThresholdReclusterizer")
    e.getByToken(tPixelClusters, inputClusters);
  else
    e.getByToken(tPixelDigi, inputDigi);

  // Step A.2: get event setup
  edm::ESHandle<TrackerGeometry> geom = es.getHandle(trackerGeomToken_);

  edm::ESHandle<TrackerTopology> trackerTopologyHandle = es.getHandle(trackerTopoToken_);
  tTopo_ = trackerTopologyHandle.product();

  // Step B: create the final output collection
  auto output = std::make_unique<SiPixelClusterCollectionNew>();
  //FIXME: put a reserve() here

  // Step C: Iterate over DetIds and invoke the pixel clusterizer algorithm
  // on each DetUnit
  if (clusterMode_ == "PixelThresholdReclusterizer")
    run(*inputClusters, geom, *output);
  else
    run(*inputDigi, geom, *output);

  // Step D: write output to file
  output->shrink_to_fit();
  e.put(tPutPixelClusters, std::move(output));
}

//---------------------------------------------------------------------------
//!  Set up the specific algorithm we are going to use.
//!  TO DO: in the future, we should allow for a different algorithm for
//!  each detector subset (e.g. barrel vs forward, per layer, etc).
//---------------------------------------------------------------------------
void SiPixelClusterProducer::setupClusterizer(const edm::ParameterSet& conf) {
  if (clusterMode_ == "PixelThresholdReclusterizer" || clusterMode_ == "PixelThresholdClusterizer") {
    clusterizer_ = std::make_unique<PixelThresholdClusterizer>(conf);
    clusterizer_->setSiPixelGainCalibrationService(theSiPixelGainCalibration_.get());
  } else {
    throw cms::Exception("Configuration") << "[SiPixelClusterProducer]:"
                                          << " choice " << clusterMode_ << " is invalid.\n"
                                          << "Possible choices:\n"
                                          << "    PixelThresholdClusterizer";
  }
}

//---------------------------------------------------------------------------
//!  Iterate over DetUnits, and invoke the PixelClusterizer on each.
//---------------------------------------------------------------------------
template <typename T>
void SiPixelClusterProducer::run(const T& input,
                                 const edm::ESHandle<TrackerGeometry>& geom,
                                 edmNew::DetSetVector<SiPixelCluster>& output) {
  int numberOfDetUnits = 0;
  int numberOfClusters = 0;

  // Iterate on detector units
  typename T::const_iterator DSViter = input.begin();
  for (; DSViter != input.end(); DSViter++) {
    ++numberOfDetUnits;

    //  LogDebug takes very long time, get rid off.
    //LogDebug("SiStripClusterizer") << "[SiPixelClusterProducer::run] DetID" << DSViter->id;

    std::vector<short> badChannels;
    DetId detIdObject(DSViter->detId());

    // Comment: At the moment the clusterizer depends on geometry
    // to access information as the pixel topology (number of columns
    // and rows in a detector module).
    // In the future the geometry service will be replaced with
    // a ES service.
    const GeomDetUnit* geoUnit = geom->idToDetUnit(detIdObject);
    const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(geoUnit);
    if (!pixDet) {
      // Fatal error!  TO DO: throw an exception!
      assert(0);
    }
    {
      // Produce clusters for this DetUnit and store them in
      // a DetSet
      edmNew::DetSetVector<SiPixelCluster>::FastFiller spc(output, DSViter->detId());
      clusterizer_->clusterizeDetUnit(*DSViter, pixDet, tTopo_, badChannels, spc);
      if (spc.empty()) {
        spc.abort();
      } else {
        numberOfClusters += spc.size();
      }
    }  // spc is not deleted and detsetvector updated
    if ((maxTotalClusters_ >= 0) && (numberOfClusters > maxTotalClusters_)) {
      edm::LogError("TooManyClusters")
          << "Limit on the number of clusters exceeded. An empty cluster collection will be produced instead.\n";
      edmNew::DetSetVector<SiPixelCluster> empty;
      empty.swap(output);
      break;
    }
  }  // end of DetUnit loop

  //LogDebug ("SiPixelClusterProducer") << " Executing "
  //      << clusterMode_ << " resulted in " << numberOfClusters
  //      << " SiPixelClusters in " << numberOfDetUnits << " DetUnits.";
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiPixelClusterProducer);
