#ifndef RecoLocalMuon_RPCRecHitProducer_h
#define RecoLocalMuon_RPCRecHitProducer_h

/** \class RPCRecHitProducer
 *  Module for RPCRecHit production. 
 *  
 *  \author M. Maggim -- INFN Bari
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "CondFormats/RPCObjects/interface/RPCMaskedStrips.h"
#include "CondFormats/RPCObjects/interface/RPCDeadStrips.h"
#include "RecoLocalMuon/RPCRecHit/interface/RPCRecHitBaseAlgo.h"

class RPCRecHitProducer : public edm::stream::EDProducer<> {

public:
  /// Constructor
  RPCRecHitProducer(const edm::ParameterSet& config);

  /// Destructor
  ~RPCRecHitProducer() override {};

  // Method that access the EventSetup for each run
  void beginRun(const edm::Run&, const edm::EventSetup& ) override;

  /// The method which produces the rechits
  void produce(edm::Event& event, const edm::EventSetup& setup) override;

private:
  // The label to be used to retrieve RPC digis from the event
  const edm::EDGetTokenT<RPCDigiCollection> theRPCDigiLabel;
  //  edm::InputTag theRPCDigiLabel;

  // The reconstruction algorithm
  std::unique_ptr<RPCRecHitBaseAlgo> theAlgo;

  std::unique_ptr<RPCMaskedStrips> theRPCMaskedStripsObj;
  // Object with mask-strips-vector for all the RPC Detectors

  std::unique_ptr<RPCDeadStrips> theRPCDeadStripsObj;
  // Object with dead-strips-vector for all the RPC Detectors

  enum class MaskSource { File, EventSetup } maskSource_, deadSource_;

  std::vector<RPCMaskedStrips::MaskItem> MaskVec;
  std::vector<RPCDeadStrips::DeadItem> DeadVec;

};

#endif

