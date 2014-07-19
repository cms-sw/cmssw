#ifndef EventFilter_Phase2TrackerRawToDigi_Phase2TrackerCommissioningDigiProducer_H
#define EventFilter_Phase2TrackerRawToDigi_Phase2TrackerCommissioningDigiProducer_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDBuffer.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerCommissioningDigi.h"
#include "boost/cstdint.hpp"
#include <iostream>
#include <string>
#include <vector>

namespace Phase2Tracker {

  class Phase2TrackerCommissioningDigiProducer : public edm::EDProducer
  {
  public:
    /// constructor
    Phase2TrackerCommissioningDigiProducer( const edm::ParameterSet& pset );
    /// default constructor
    ~Phase2TrackerCommissioningDigiProducer();
    void produce( edm::Event& event, const edm::EventSetup& es );
    
  private:
    unsigned int runNumber_;
<<<<<<< HEAD
    edm::InputTag productLabel_;
    const SiStripFedCabling * cabling_;
    edm::EDGetTokenT<FEDRawDataCollection> token_;
    DetIdCollection detids_;

    std::vector<Registry> proc_work_registry_;
    std::vector<Phase2TrackerCommissioningDigi> cond_data_digis_;
  };
}
#endif // EventFilter_Phase2TrackerRawToDigi_Phase2TrackerCommissioningDigiProducer_H
