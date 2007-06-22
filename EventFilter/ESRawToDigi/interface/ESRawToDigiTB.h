#ifndef ESRAWTODIGITB_H
#define ESRAWTODIGITB_H 

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "EventFilter/ESRawToDigi/interface/ESUnpackerTB.h"

using namespace std;
using namespace edm;

class ESRawToDigiTB : public EDProducer {
  
 public:
  
  ESRawToDigiTB(const ParameterSet& ps);
  virtual ~ESRawToDigiTB();
  
  void produce(Event& e, const EventSetup& es);
  
 private:

  string label_;
  string instanceName_;
  string ESdigiCollection_;

  bool debug_;

  ESUnpackerTB* ESUnpackerTB_;
  
};

#endif
