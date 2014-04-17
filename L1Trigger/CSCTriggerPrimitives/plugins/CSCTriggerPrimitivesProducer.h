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
  virtual void endJob();

 private:
  int iev; // event number
  edm::InputTag compDigiProducer_;
  edm::InputTag wireDigiProducer_;
  edm::InputTag gemPadProducer_;
  edm::InputTag rpcDigiProducer_;
  // swich to force the use of parameters from config file rather then from DB
  bool debugParameters_;
  // switch to for enabling checking against the list of bad chambers
  bool checkBadChambers_;
  CSCTriggerPrimitivesBuilder* lctBuilder_;

 public:

  // variables for debugging
  int me1bValidAlct_;
  int me1bValidAlctValidClct_;
  int me1bValidAlctClctInBoxWindow_;
  int me1bValidAlctNoValidClct_;
  int me1bMatchAttempts_;
  int me1bMatchAlctClct_;
  int me1bAlctClctLowQ_;
  int me1bAlctClctLowQInEdge_;
  int me1bAlctNoValidClct_;
  int me1bMatchAlctGemCoPad_;
  int me1bValidAlctGemInBXWindow_;
  int me1bAlctGemNoCoPad_;
  int me1bValidAlctGemNoCoPad_;
  int me1bValidAlctGemCoPad_;
  int me1bMatchAlctClctLowQ_;
  int me1bMatchAlctNoClct_;
  int me1bMatchAlctClctLowQInEdge_;
  int me1bAlctClctLowQNoGemPad_;

  int me1aValidAlct_;
  int me1aValidAlctValidClct_;
  int me1aValidAlctClctInBoxWindow_;
  int me1aValidAlctNoValidClct_;
  int me1aMatchAttempts_;
  int me1aMatchAlctClct_;
  int me1aAlctClctLowQ_;
  int me1aAlctClctLowQInEdge_;
  int me1aAlctNoValidClct_;
  int me1aMatchAlctGemCoPad_;
  int me1aValidAlctGemInBXWindow_;
  int me1aAlctGemNoCoPad_;
  int me1aValidAlctGemNoCoPad_;
  int me1aValidAlctGemCoPad_;
  int me1aMatchAlctClctLowQ_;
  int me1aMatchAlctNoClct_;
  int me1aMatchAlctClctLowQInEdge_;
  int me1aAlctClctLowQNoGemPad_;


};

#endif
