#ifndef ESDIGITORAW_H
#define ESDIGITORAW_H

#include <memory>
#include <iostream>
#include <string>
#include <fstream>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "EventFilter/ESDigiToRaw/interface/ESDataFormatter.h"

class ESDigiToRaw : public edm::EDProducer {
  
 public:
  
  ESDigiToRaw(const edm::ParameterSet& ps);
  virtual ~ESDigiToRaw();
  
  void beginJob() ;
  void produce(edm::Event& e, const edm::EventSetup& es);
  void endJob() ;

  typedef uint32_t Word32;
  typedef uint64_t Word64;
  
  int* GetCounter() {return &counter_ ;}
  int* GetOrbit() {return &orbit_number_ ;}
  int* GetBX() {return &bx_ ;}
  int* GetLV1() {return &lv1_ ;}
  int* GetRunNumber() {return &run_number_ ;}
  
  static const int BXMAX = 2808;
  static const int LHC_BX_RANGE = 3564;
  static const int KCHIP_BC_RANGE = 4096;
  static const int KCHIP_EC_RANGE = 256;
  
 private:
  
  int counter_;
  int orbit_number_;
  int run_number_;
  int bx_;
  int lv1_;
  int kchip_ec_; 
  int kchip_bc_; 
  int fedId_[2][2][40][40];
    
  std::string label_;
  std::string instanceName_;
  edm::EDGetTokenT<ESDigiCollection> ESDigiToken_;  
  edm::FileInPath lookup_;
  bool   debug_;
  int formatMajor_; 
  int formatMinor_; 

  ESDataFormatter* ESDataFormatter_;
  
};

#endif
