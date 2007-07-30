#ifndef RecoParticleFlow_PFProducer_h_
#define RecoParticleFlow_PFProducer_h_

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

// useful?
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoParticleFlow/PFAlgo/interface/PFAlgo.h"


/**\class PFProducer 
\brief Producer for particle flow reconstructed particles (PFCandidates)

This producer makes use of PFAlgo, the particle flow algorithm.

\author Colin Bernet
\date   July 2006
*/

class PFProducer : public edm::EDProducer {
 public:
  explicit PFProducer(const edm::ParameterSet&);
  ~PFProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup & c);

 private:

  /// module label for retrieving input collection of PFBlocks 
  std::string blocksModuleLabel_;

  /// instance name for retrieving input collection of PFBlocks  
  std::string blocksInstanceName_;

  /// verbose ?
  bool   verbose_;

  /// particle flow algorithm
  PFAlgo      pfAlgo_;

};

#endif
