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

class MeasurementTrackerEventProducer : public edm::stream::EDProducer<> {
public:
      explicit MeasurementTrackerEventProducer(const edm::ParameterSet &iConfig) ;
      ~MeasurementTrackerEventProducer() {}
private:
      virtual void produce(edm::Event&, const edm::EventSetup&);

protected:
      void updatePixels( const edm::Event&, PxMeasurementDetSet & thePxDets, std::vector<bool> & pixelClustersToSkip ) const;
      void updateStrips( const edm::Event&, StMeasurementDetSet & theStDets, std::vector<bool> & stripClustersToSkip ) const;

      void getInactiveStrips(const edm::Event& event,std::vector<uint32_t> & rawInactiveDetIds) const;

      std::string measurementTrackerLabel_;
      const edm::ParameterSet& pset_;
      edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> thePixelClusterLabel;
      edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster>> theStripClusterLabel;
      edm::EDGetTokenT<edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster>>> thePixelClusterMask;
      edm::EDGetTokenT<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster>>> theStripClusterMask;

      std::vector<edm::EDGetTokenT<DetIdCollection>>      theInactivePixelDetectorLabels;
      std::vector<edm::EDGetTokenT<DetIdCollection>>      theInactiveStripDetectorLabels;

      bool selfUpdateSkipClusters_;
};

#endif
