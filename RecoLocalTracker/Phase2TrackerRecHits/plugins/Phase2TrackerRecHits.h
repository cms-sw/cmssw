#ifndef RecoLocalTracker_Phase2TrackerRecHits_Phase2TrackerRecHits_h
#define RecoLocalTracker_Phase2TrackerRecHits_Phase2TrackerRecHits_h

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEDummy.h"


  class Phase2TrackerRecHits : public edm::global::EDProducer<> {

    public:

      explicit Phase2TrackerRecHits(const edm::ParameterSet& conf);
      virtual ~Phase2TrackerRecHits() {};
      void produce(edm::StreamID sid, edm::Event& event, const edm::EventSetup& eventSetup) const override final;

    private:
            
      edm::EDGetTokenT< Phase2TrackerCluster1DCollectionNew > token_;
      edm::ESInputTag cpeTag_;
    
    };

#endif
