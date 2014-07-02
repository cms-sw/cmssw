#ifndef _ESRAWTODIGI_H_
#define _ESRAWTODIGI_H_ 

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "EventFilter/ESRawToDigi/interface/ESUnpacker.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalRawData/interface/ESListOfFEDS.h"


class ESRawToDigi : public edm::stream::EDProducer<> {
  
 public:
  
  ESRawToDigi(const edm::ParameterSet& ps);
  virtual ~ESRawToDigi();
  
  void produce(edm::Event& e, const edm::EventSetup& es);
  
 private:



  std::string ESdigiCollection_;
  edm::EDGetTokenT<FEDRawDataCollection> dataToken_;
  edm::EDGetTokenT<ESListOfFEDS>         fedsToken_;


  bool regional_;

  bool debug_;

  ESUnpacker* ESUnpacker_;
  
};

#endif
