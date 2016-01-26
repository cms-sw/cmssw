#ifndef RecoLocalTracker_SiPhase2Clusterizer_Phase2TrackerClusterizer_h
#define RecoLocalTracker_SiPhase2Clusterizer_Phase2TrackerClusterizer_h

#include "RecoLocalTracker/SiPhase2Clusterizer/interface/Phase2TrackerClusterizerAlgorithm.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

namespace cms {

    class Phase2TrackerClusterizer : public edm::EDProducer {

        public:
            explicit Phase2TrackerClusterizer(const edm::ParameterSet& conf);
            virtual ~Phase2TrackerClusterizer();
            virtual void produce(edm::Event& event, const edm::EventSetup& eventSetup);

        private:
            edm::ParameterSet conf_;
            Phase2TrackerClusterizerAlgorithm* clusterizer_;
            edm::InputTag src_;
            edm::EDGetTokenT< edm::DetSetVector< Phase2TrackerDigi > > token_;
            unsigned int maxClusterSize_;
            unsigned int maxNumberClusters_;

    };
}

#endif
