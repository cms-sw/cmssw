#ifndef CSCTriggerPrimitives_CSCTriggerPrimitivesProducer_h
#define CSCTriggerPrimitives_CSCTriggerPrimitivesProducer_h

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
 * $Date: 2009/03/24 14:11:36 $
 * $Revision: 1.2 $
 *
 */

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Utilities/interface/InputTag.h>

class CSCTriggerPrimitivesBuilder;

class CSCTriggerPrimitivesProducer : public edm::EDProducer
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
  CSCTriggerPrimitivesBuilder* lctBuilder_;
};

#endif
