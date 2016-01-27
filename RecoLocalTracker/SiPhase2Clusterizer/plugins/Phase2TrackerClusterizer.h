#ifndef RecoLocalTracker_SiPhase2Clusterizer_Phase2TrackerClusterizer_h
#define RecoLocalTracker_SiPhase2Clusterizer_Phase2TrackerClusterizer_h

#include "RecoLocalTracker/SiPhase2Clusterizer/interface/Phase2TrackerClusterizerAlgorithm.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <memory>


    class Phase2TrackerClusterizer : public edm::global::EDProducer<> {

        public:
            explicit Phase2TrackerClusterizer(const edm::ParameterSet& conf);
            virtual ~Phase2TrackerClusterizer();
            void produce(edm::StreamID sid, edm::Event& event, const edm::EventSetup& eventSetup) const override final;

        private:
            std::unique_ptr< Phase2TrackerClusterizerAlgorithm > clusterizer_;
            edm::EDGetTokenT< edm::DetSetVector< Phase2TrackerDigi > > token_;

    };


#endif
