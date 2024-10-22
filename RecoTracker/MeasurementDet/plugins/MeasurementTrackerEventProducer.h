#ifndef MeasurementTrackerEventProducer_h
#define MeasurementTrackerEventProducer_h
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/MeasurementDet/src/TkMeasurementDetSet.h"
#include "DataFormats/Common/interface/ContainerMask.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/DetId/interface/DetIdVector.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

class dso_hidden MeasurementTrackerEventProducer final : public edm::stream::EDProducer<> {
public:
  explicit MeasurementTrackerEventProducer(const edm::ParameterSet& iConfig);
  ~MeasurementTrackerEventProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

protected:
  void updatePixels(const edm::Event&,
                    PxMeasurementDetSet& thePxDets,
                    std::vector<bool>& pixelClustersToSkip,
                    const TrackerGeometry& trackerGeom,
                    const edm::EventSetup& iSetup) const;
  void updateStrips(const edm::Event&, StMeasurementDetSet& theStDets, std::vector<bool>& stripClustersToSkip) const;
  void updatePhase2OT(const edm::Event&, Phase2OTMeasurementDetSet& thePh2OTDets) const;
  //FIXME:: going to be updated soon
  void updateStacks(const edm::Event&, Phase2OTMeasurementDetSet& theStDets) const {};

  void getInactiveStrips(const edm::Event& event, std::vector<uint32_t>& rawInactiveDetIds) const;

  edm::ESGetToken<MeasurementTracker, CkfComponentsRecord> measurementTrackerToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> thePixelClusterLabel;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster>> theStripClusterLabel;
  edm::EDGetTokenT<edmNew::DetSetVector<Phase2TrackerCluster1D>> thePh2OTClusterLabel;
  edm::EDGetTokenT<VectorHitCollection> thePh2OTVectorHitsLabel;
  edm::EDGetTokenT<VectorHitCollection> thePh2OTVectorHitsRejLabel;
  edm::EDGetTokenT<edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster>>> thePixelClusterMask;
  edm::EDGetTokenT<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster>>> theStripClusterMask;

  std::vector<edm::EDGetTokenT<DetIdCollection>> theInactivePixelDetectorLabels;
  std::vector<edm::EDGetTokenT<PixelFEDChannelCollection>> theBadPixelFEDChannelsLabels;
  edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> pixelCablingMapToken_;
  std::vector<edm::EDGetTokenT<DetIdVector>> theInactiveStripDetectorLabels;

  bool selfUpdateSkipClusters_;
  bool switchOffPixelsIfEmpty_;
  bool isPhase2_;
  bool useVectorHits_;
};

#endif
