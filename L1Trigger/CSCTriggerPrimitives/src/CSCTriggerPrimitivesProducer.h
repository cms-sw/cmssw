#ifndef CSCTriggerPrimitives_CSCTriggerPrimitivesProducer_h
#define CSCTriggerPrimitives_CSCTriggerPrimitivesProducer_h

/** \class CSCTriggerPrimitivesProducer
 *
 * Implementation of the local Level-1 Cathode Strip Chamber trigger.
 * Simulates functionalities of the anode and cathode Local Charged Tracks
 * (LCT) processors and of the Trigger Motherboard.
 *
 * Input to the simulation are collections of the CSC wire and comparator
 * digis.
 *
 * Produces three collections of the Level-1 CSC Trigger Primitives (track
 * stubs, or LCTs): anode LCTs (ALCTs), cathode LCTs (CLCTs), and correlated
 * LCTs.
 *
 * \author Slava Valuev  May 2006
 *
 */

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

class CSCTriggerPrimitivesBuilder;

class CSCTriggerPrimitivesProducer : public edm::EDProducer
{
 public:
  explicit CSCTriggerPrimitivesProducer(const edm::ParameterSet&);
  ~CSCTriggerPrimitivesProducer();

  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
  int iev; // event number
  std::string compDigiProducer_;
  std::string wireDigiProducer_;
  CSCTriggerPrimitivesBuilder* lctBuilder_;
};

#endif
