#ifndef FastSimulation_TrackingRecHitProducer_TrackingRecHitAlgorithm_H
#define FastSimulation_TrackingRecHitProducer_TrackingRecHitAlgorithm_H

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitProduct.h"

#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"

// Pixel-related stuff:
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelTemplate.h"

#include <string>
#include <memory>

class TrackingRecHitAlgorithm {
private:
  const std::string _name;
  const std::string _selectionString;
  const TrackerTopology* _trackerTopology;
  const TrackerGeometry* _trackerGeometry;
  const TrackerGeometry* _misalignedTrackerGeometry;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyESToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryESToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> misalignedTrackerGeometryESToken_;
  std::shared_ptr<RandomEngineAndDistribution> _randomEngine;

public:
  TrackingRecHitAlgorithm(const std::string& name,
                          const edm::ParameterSet& config,
                          edm::ConsumesCollector& consumesCollector);

  inline const std::string& getName() const { return _name; }
  inline const std::string& getSelectionString() const { return _selectionString; }

  const TrackerTopology& getTrackerTopology() const;
  const TrackerGeometry& getTrackerGeometry() const;
  const TrackerGeometry& getMisalignedGeometry() const;
  const RandomEngineAndDistribution& getRandomEngine() const;

  //this function will only be called once per stream
  virtual void beginStream(const edm::StreamID& id);

  //this function will only be called once per run
  virtual void beginRun(edm::Run const& run,
                        const edm::EventSetup& eventSetup,
                        const SiPixelTemplateDBObject* pixelTemplateDBObjectPtr,
                        const std::vector<SiPixelTemplateStore>& tempStoreRef);

  //this function will only be called once per event
  virtual void beginEvent(edm::Event& event, const edm::EventSetup& eventSetup);

  //the main action is here
  virtual TrackingRecHitProductPtr process(TrackingRecHitProductPtr product) const;

  //this function will only be called once per event
  virtual void endEvent(edm::Event& event, const edm::EventSetup& eventSetup);

  //this function will only be called once per stream
  virtual void endStream();

  virtual ~TrackingRecHitAlgorithm();
};

#endif
