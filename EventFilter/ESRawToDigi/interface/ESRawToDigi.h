#ifndef _ESRAWTODIGI_H_
#define _ESRAWTODIGI_H_ 

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "EventFilter/ESRawToDigi/interface/ESUnpacker.h"




class ESRawToDigi : public edm::EDProducer {
  
 public:
  
  ESRawToDigi(const edm::ParameterSet& ps);
  virtual ~ESRawToDigi();
  
  void produce(edm::Event& e, const edm::EventSetup& es);
  
 private:

  edm::InputTag sourceTag_;
  edm::InputTag fedsListLabel_;
  std::string ESdigiCollection_;
  bool regional_;

  bool debug_;

  ESUnpacker* ESUnpacker_;
  
};

#endif
