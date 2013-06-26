#ifndef RecoLocalMuon_RPCRecHitProducer_h
#define RecoLocalMuon_RPCRecHitProducer_h

/** \class RPCRecHitProducer
 *  Module for RPCRecHit production. 
 *  
 *  $Date: 2013/02/25 18:49:16 $
 *  $Revision: 1.7 $
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
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "CondFormats/RPCObjects/interface/RPCMaskedStrips.h"
#include "CondFormats/DataRecord/interface/RPCMaskedStripsRcd.h"
#include "CondFormats/RPCObjects/interface/RPCDeadStrips.h"
#include "CondFormats/DataRecord/interface/RPCDeadStripsRcd.h"

#include "RPCRollMask.h"


namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class RPCRecHitBaseAlgo;

class RPCRecHitProducer : public edm::EDProducer {

public:
  /// Constructor
  RPCRecHitProducer(const edm::ParameterSet& config);

  /// Destructor
  virtual ~RPCRecHitProducer();

  // Method that access the EventSetup for each run
  virtual void beginRun(const edm::Run&, const edm::EventSetup& ) override;

  /// The method which produces the rechits
  virtual void produce(edm::Event& event, const edm::EventSetup& setup) override;

private:

  // The label to be used to retrieve RPC digis from the event
  edm::InputTag theRPCDigiLabel;

  // The reconstruction algorithm
  RPCRecHitBaseAlgo *theAlgo;
//   static std::string theAlgoName;

  RPCMaskedStrips* RPCMaskedStripsObj;
  // Object with mask-strips-vector for all the RPC Detectors

  RPCDeadStrips* RPCDeadStripsObj;
  // Object with dead-strips-vector for all the RPC Detectors

  std::string maskSource;
  std::string deadSource;

  std::vector<RPCMaskedStrips::MaskItem> MaskVec;
  std::vector<RPCDeadStrips::DeadItem> DeadVec;

};

#endif

