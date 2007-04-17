#ifndef PathTimerInserter_h
#define PathTimerInserter_h

/*
  Author: David Lange, LLNL

*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class PathTimerInserter : public edm::EDProducer
{
 public:
  
  explicit PathTimerInserter(edm::ParameterSet const& ps);
  
  virtual ~PathTimerInserter();
  
  virtual void produce(edm::Event& e, edm::EventSetup const& c);
  
 private:
};

  // ---------------------

#endif
