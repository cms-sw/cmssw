#ifndef RecoParticleFlow_PFProducer_h_
#define RecoParticleFlow_PFProducer_h_

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"


/**\class PFProducer 
\brief Producer for particle flow tracks, particles and reconstructed particles 

This producer makes use of PFAlgo, the particle flow algorithm.

\author Colin Bernet, Renaud Bruneliere
\date   July 2006
*/

class FSimEvent;

class PFProducer : public edm::EDProducer {
 public:
  explicit PFProducer(const edm::ParameterSet&);
  ~PFProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup & c);

 private:
  // ----------member data ---------------------------

  // input rec track collection name
  std::string recTrackModuleLabel_;

  // module label for input simtrack and simvertex
  std::string simModuleLabel_;  
  
  // output collection name for reconstructed tracks
  std::string pfRecTrackCollection_;

  // output collection name for particles
  std::string pfParticleCollection_;

  // parameters used for track reconstruction
  TrackProducerAlgorithm trackAlgo_;
  std::string       fitterName_;
  std::string       propagatorName_;
  std::string       builderName_;

  // parameters for retrieving true particles information
  edm::ParameterSet vertexGenerator_;
  edm::ParameterSet particleFilter_;
  FSimEvent* mySimEvent;
  
};

#endif
