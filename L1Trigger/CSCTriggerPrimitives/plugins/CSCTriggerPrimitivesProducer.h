#ifndef L1Trigger_CSCTriggerPrimitives_CSCTriggerPrimitivesProducer_h
#define L1Trigger_CSCTriggerPrimitives_CSCTriggerPrimitivesProducer_h

/** \class CSCTriggerPrimitivesProducer
 *
 * Implementation of the local Level-1 Cathode Strip Chamber trigger.
 * Simulates functionalities of the anode and cathode Local Charged Tracks
 * (LCT) processors, of the Trigger Mother Board (TMB), and of the Muon Port
 * Card (MPC).
 *
 * Input to the simulation are collections of the CSC wire and comparator
 * digis.
 *
 * Produces four collections of the Level-1 CSC Trigger Primitives (track
 * stubs, or LCTs): anode LCTs (ALCTs), cathode LCTs (CLCTs), correlated
 * LCTs at TMB, and correlated LCTs at MPC.
 *
 * \author Slava Valuev, UCLA.
 *
 *
 */

#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

class CSCTriggerPrimitivesBuilder;

class CSCTriggerPrimitivesProducer : public edm::one::EDProducer<edm::one::SharedResources>
{
 public:
  explicit CSCTriggerPrimitivesProducer(const edm::ParameterSet&);
  ~CSCTriggerPrimitivesProducer();

  //virtual void beginRun(const edm::EventSetup& setup);
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
  int iev; // event number
 
  edm::InputTag compDigiProducer_;
  edm::InputTag wireDigiProducer_;
  edm::InputTag gemPadDigiProducer_;
  edm::InputTag rpcDigiProducer_;
  edm::EDGetTokenT<CSCComparatorDigiCollection> comp_token_;
  edm::EDGetTokenT<CSCWireDigiCollection> wire_token_;
  edm::EDGetTokenT<GEMPadDigiCollection> gem_pad_token_;
  edm::EDGetTokenT<RPCDigiCollection> rpc_digi_token_;
 
  // switch to force the use of parameters from config file rather then from DB
  bool debugParameters_;
  // switch to for enabling checking against the list of bad chambers
  bool checkBadChambers_;
  bool runME11ILT_;
  bool runME21ILT_;
  std::unique_ptr<CSCTriggerPrimitivesBuilder> lctBuilder_;
};

#endif
