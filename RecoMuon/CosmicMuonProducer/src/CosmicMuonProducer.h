#ifndef CosmicMuonProducer_h
#define CosmicMuonProducer_h

/** \file CosmicMuonProducer
 *
 *  $Date: $
 *  $Revision: $
 *  \author Chang Liu
 */

#include "FWCore/Framework/interface/EDProducer.h"
//#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/EventSetup.h"

//#include "FWCore/ParameterSet/interface/ParameterSet.h"


class CosmicMuonProducer : public edm::EDProducer {
public:
  explicit CosmicMuonProducer(const edm::ParameterSet&);

   ~CosmicMuonProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:
};

#endif


