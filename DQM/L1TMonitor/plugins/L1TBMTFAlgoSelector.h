#ifndef DQM_L1TMonitor_L1TBMTFAlgoSelector_h
#define DQM_L1TMonitor_L1TBMTFAlgoSelector_h

// system requirements
#include <iosfwd>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

// general requirements
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/Handle.h"

// stage2 requirements
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "EventFilter/L1TRawToDigi/interface/AMC13Spec.h"
#include "EventFilter/L1TRawToDigi/interface/Block.h"


// class decleration

namespace dqmBmtfAlgoSelector{

  class  L1TBMTFAlgoSelector: public edm::stream::EDProducer<> {

  public:

    // class constructor
    explicit L1TBMTFAlgoSelector(const edm::ParameterSet & ps);
    // class destructor
    ~L1TBMTFAlgoSelector() override;

    // member functions
  private:
    void produce(edm::Event&, const edm::EventSetup&) override;

    // data members  
    edm::EDGetToken bmtfKalmanToken;
    edm::EDGetToken bmtfLegacyToken;
    edm::EDGetToken fedToken;
  };
}
#endif
