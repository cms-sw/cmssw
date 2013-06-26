#ifndef EventFilter_SiStripRawToDigi_ExcludedFEDListProducer_H
#define EventFilter_SiStripRawToDigi_ExcludedFEDListProducer_H

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
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "boost/cstdint.hpp"
#include <iostream>
#include <string>
#include <vector>

namespace sistrip {

  class ExcludedFEDListProducer : public edm::EDProducer
  {
  public:
    /// constructor
    ExcludedFEDListProducer( const edm::ParameterSet& pset );
    /// default constructor
    ~ExcludedFEDListProducer();
    void beginRun( const edm::Run & run, const edm::EventSetup & es) override;
    void produce( edm::Event& event, const edm::EventSetup& es ) override;
    
  private:
    unsigned int runNumber_;
    edm::InputTag productLabel_;
    const SiStripFedCabling * cabling_;
    uint32_t cacheId_;
    DetIdCollection detids_;
  };
}
#endif // EventFilter_SiStripRawToDigi_ExcludedFEDListProducer_H
