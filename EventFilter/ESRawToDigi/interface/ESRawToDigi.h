#ifndef _ESRAWTODIGI_H_
#define _ESRAWTODIGI_H_ 

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "EventFilter/ESRawToDigi/interface/ESUnpacker.h"

using namespace std;
using namespace edm;

class ESRawToDigi : public EDProducer {
  
 public:
  
  ESRawToDigi(const ParameterSet& ps);
  virtual ~ESRawToDigi();
  
  void produce(Event& e, const EventSetup& es);
  
 private:

  edm::InputTag sourceTag_;
  edm::InputTag fedsListLabel_;
  string ESdigiCollection_;
  bool regional_;

  bool debug_;

  ESUnpacker* ESUnpacker_;
  
};

#endif
