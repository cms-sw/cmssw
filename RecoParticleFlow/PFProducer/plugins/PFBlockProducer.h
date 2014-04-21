#ifndef RecoParticleFlow_PFProducer_PFBlockProducer_h_
#define RecoParticleFlow_PFProducer_PFBlockProducer_h_

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoParticleFlow/PFProducer/interface/PFBlockAlgo.h"


/**\class PFBlockProducer 
\brief Producer for particle flow blocks

This producer makes use of PFBlockAlgo, the particle flow block algorithm.
Particle flow itself consists in reconstructing particles from the particle 
flow blocks This is done at a later stage, see PFProducer and PFAlgo.

\author Colin Bernet
\date   April 2007
*/

class FSimEvent;



class PFBlockProducer : public edm::EDProducer {
 public:

  explicit PFBlockProducer(const edm::ParameterSet&);

  ~PFBlockProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

 private:
  /// verbose ?
  bool   verbose_;

  /// Particle flow block algorithm 
  PFBlockAlgo            pfBlockAlgo_;

};

#endif
