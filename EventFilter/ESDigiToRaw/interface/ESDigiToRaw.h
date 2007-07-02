#ifndef ESDIGITORAW_H
#define ESDIGITORAW_H

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

using namespace std;
using namespace edm;

class ESDigiToRaw : public EDProducer {
  
 public:
  
  ESDigiToRaw(const ParameterSet& ps);
  virtual ~ESDigiToRaw();
  
  void beginJob(const EventSetup& es) ;
  void produce(Event& e, const EventSetup& es);
  void endJob() ;
  
  typedef long long Word64;
  typedef unsigned int Word32;
  
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
    
  string label_;
  string instanceName_;
  bool   debug_;

  ESDataFormatter* ESDataFormatter_;
  
};

#endif
