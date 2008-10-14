#ifndef RecoLocalMuon_RPCRecHitProducer_h
#define RecoLocalMuon_RPCRecHitProducer_h

/** \class RPCRecHitProducer
 *  Module for RPCRecHit production. 
 *  
 *  $Date: 2008/01/29 12:53:03 $
 *  $Revision: 1.2 $
 *  \author M. Maggim -- INFN Bari
 */


#include <memory>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <cstdlib>
#include <bitset>
#include <map>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "RecoLocalMuon/RPCRecHit/src/RPCMaskReClusterizer.h"


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

  std::map<RPCDetId,RollMask> MaskMap;
  // Map with masks for all the RPC Detectors

};
#endif

