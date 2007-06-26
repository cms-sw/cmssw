#ifndef ESRAWTODIGICT_H
#define ESRAWTODIGICT_H 

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "EventFilter/ESRawToDigi/interface/ESUnpackerCT.h"

using namespace std;
using namespace edm;

class ESRawToDigiCT : public EDProducer {
  
 public:
  
  ESRawToDigiCT(const ParameterSet& ps);
  virtual ~ESRawToDigiCT();
  
  void produce(Event& e, const EventSetup& es);
  
 private:

  vector<int> fedUnpackList_;

  string label_;
  string instanceName_;
  string ESdigiCollection_;

  bool debug_;

  ESUnpackerCT* ESUnpackerCT_;
  
};

#endif
