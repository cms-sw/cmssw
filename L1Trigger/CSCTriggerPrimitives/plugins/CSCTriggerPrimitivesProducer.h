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
 * The trigger primitive emulator has been expanded with options to
 * use both ALCTs, CLCTs and GEM pads. The GEM-CSC integrated
 * local trigger combines ALCT, CLCT and GEM information to produce integrated
 * stubs. The available stub types can be found in the class definition of
 * CSCCorrelatedLCTDigi (DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h)
 * Either single GEM pads or GEM pad clusters can be used as input. The online
 * system will use GEM pad clusters however.
 *
 * authors: Sven Dildick (TAMU), Tao Huang (TAMU)
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"
#include "L1Trigger/CSCTriggerPrimitives/src/CSCTriggerPrimitivesBuilder.h"

class CSCTriggerPrimitivesProducer : public edm::global::EDProducer<edm::StreamCache<CSCTriggerPrimitivesBuilder>>
{
 public:
  explicit CSCTriggerPrimitivesProducer(const edm::ParameterSet&);
  ~CSCTriggerPrimitivesProducer() override;

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

 private:

  // master configuration
  edm::ParameterSet config_;

  std::unique_ptr<CSCTriggerPrimitivesBuilder> beginStream(edm::StreamID) const override {
    return std::unique_ptr<CSCTriggerPrimitivesBuilder>(new CSCTriggerPrimitivesBuilder(config_));
  }

  // input tags for input collections
  edm::InputTag compDigiProducer_;
  edm::InputTag wireDigiProducer_;
  edm::InputTag gemPadDigiProducer_;
  edm::InputTag gemPadDigiClusterProducer_;

  // tokens
  edm::EDGetTokenT<CSCComparatorDigiCollection> comp_token_;
  edm::EDGetTokenT<CSCWireDigiCollection> wire_token_;
  edm::EDGetTokenT<GEMPadDigiCollection> gem_pad_token_;
  edm::EDGetTokenT<GEMPadDigiClusterCollection> gem_pad_cluster_token_;

  // switch to force the use of parameters from config file rather then from DB
  bool debugParameters_;

  // switch to for enabling checking against the list of bad chambers
  bool checkBadChambers_;

  // switch to enable the integrated local triggers in ME11 and ME21
  bool runME11ILT_;
  bool runME21ILT_;
};

#endif
