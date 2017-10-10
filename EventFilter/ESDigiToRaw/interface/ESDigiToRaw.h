#ifndef ESDIGITORAW_H
#define ESDIGITORAW_H

#include <memory>
#include <iostream>
#include <string>
#include <fstream>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "EventFilter/ESDigiToRaw/interface/ESDataFormatter.h"

class ESDigiToRaw : public edm::global::EDProducer<> {
public:
  ESDigiToRaw(const edm::ParameterSet& ps);
  ~ESDigiToRaw() override;
  
  void produce(edm::StreamID, edm::Event& e, const edm::EventSetup& es) const override;

  typedef uint32_t Word32;
  typedef uint64_t Word64;
  
  static const int BXMAX = 2808;
  static const int LHC_BX_RANGE = 3564;
  static const int KCHIP_BC_RANGE = 4096;
  static const int KCHIP_EC_RANGE = 256;
  
 private:
  int fedId_[2][2][40][40];
    
  const ESDataFormatter* ESDataFormatter_;
  const std::string label_;
  const std::string instanceName_;
  const edm::EDGetTokenT<ESDigiCollection> ESDigiToken_;  
  const edm::FileInPath lookup_;
  const bool   debug_;
  const int formatMajor_; 
  const int formatMinor_; 

  
};

#endif
