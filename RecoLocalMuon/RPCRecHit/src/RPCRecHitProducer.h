#ifndef RecoLocalMuon_RPCRecHitProducer_h
#define RecoLocalMuon_RPCRecHitProducer_h

/** \class RPCRecHitProducer
 *  Module for RPCRecHit production. 
 *  
 *  $Date: 2006/05/08 11:21:22 $
 *  $Revision: 1.1 $
 *  \author M. Maggim -- INFN Bari
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class RPCRecHitBaseAlgo;

class RPCRecHitProducer : public edm::EDProducer {
public:
  /// Constructor
  RPCRecHitProducer(const edm::ParameterSet&);

  /// Destructor
  virtual ~RPCRecHitProducer();

  /// The method which produces the rechits
  virtual void produce(edm::Event& event, const edm::EventSetup& setup);

private:

  // The label to be used to retrieve RPC digis from the event
  edm::InputTag theRPCDigiLabel;
  // The reconstruction algorithm
  RPCRecHitBaseAlgo *theAlgo;
//   static string theAlgoName;

};
#endif

