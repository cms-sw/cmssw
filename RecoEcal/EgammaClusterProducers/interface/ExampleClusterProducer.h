#ifndef RecoEcal_EgammaClusterProducers_ExampleClusterProducer_h
#define RecoEcal_EgammaClusterProducers_ExampleClusterProducer_h
/** \class ExampleClusterProducer
 **   example of producer for BasicCluster from recHits
 **
 **  $Id: ExampleClusterProducer.h,v 1.1 2006/04/13 14:40:05 rahatlou Exp $
 **  $Date: 2006/04/13 14:40:05 $
 **  $Revision: 1.1 $
 **  \author Shahram Rahatlou, University of Rome & INFN, April 2006
 **
 ***/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

class ExampleClusterAlgo;

// ExampleClusterProducer inherits from EDProducer, so it can be a module:
class ExampleClusterProducer : public edm::EDProducer {

 public:

  ExampleClusterProducer (const edm::ParameterSet& ps);
  ~ExampleClusterProducer();

  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

 private:

  ExampleClusterAlgo* algo_; // algorithm doing the real work

  std::string hitProducer_;   // name of module/plugin/producer producing hits
  std::string hitCollection_; // secondary name given to collection of hits by hitProducer
  std::string clusterCollection_;  // secondary name to be given to collection of cluster produced in this module

  int nMaxPrintout_; // max # of printouts
  int nEvt_;         // internal counter of events

  bool counterExceeded() const { return ((nEvt_ > nMaxPrintout_) || (nMaxPrintout_ < 0)); }


};
#endif
