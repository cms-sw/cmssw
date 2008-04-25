#ifndef ESDIGITORAWTB_H
#define ESDIGITORAWTB_H

#include <memory>
#include <iostream>
#include <string>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "EventFilter/ESDigiToRaw/interface/ESDataFormatter.h"

class ESDigiToRawTB : public edm::EDProducer {
  
 public:
  
  ESDigiToRawTB(const edm::ParameterSet& ps);
  virtual ~ESDigiToRawTB();
  
  void beginJob(const edm::EventSetup& es) ;
  void produce(edm::Event& e, const edm::EventSetup& es);
  void endJob() ;

  typedef uint64_t  Word64;
  typedef uint32_t  Word32;
  
  int* GetCounter() {return &counter_ ;}
  int* GetOrbit() {return &orbit_number_ ;}
  int* GetBX() {return &bx_ ;}
  int* GetLV1() {return &lv1_ ;}
  int* GetRunNumber() {return &run_number_ ;}
  
  static const int BXMAX = 2808;
  
 private:
  
  int counter_;
  int orbit_number_;
  int run_number_;
  int bx_;
  int lv1_;
    
  std::string label_;
  std::string instanceName_;
  bool   debug_;

  ESDataFormatter* ESDataFormatter_;
  
};

#endif
