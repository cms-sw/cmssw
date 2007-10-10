#ifndef RecoParticleFlow_PFSimParticleProducer_h_
#define RecoParticleFlow_PFSimParticleProducer_h_

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "RecoParticleFlow/PFBlockAlgo/interface/PFGeometry.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"

#include "RecoParticleFlow/PFBlockAlgo/interface/PFBlockAlgo.h"


/**\class PFSimParticleProducer 
\brief Producer for PFRecTracks and PFSimParticles

\todo Remove the PFRecTrack part, which is now handled by PFTracking
\author Colin Bernet
\date   April 2007
*/

class FSimEvent;



class PFSimParticleProducer : public edm::EDProducer {
 public:

  explicit PFSimParticleProducer(const edm::ParameterSet&);

  ~PFSimParticleProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

  virtual void beginJob(const edm::EventSetup & c);

 private:
    

  /// module label for retrieving input simtrack and simvertex
  std::string simModuleLabel_;  


  // parameters for retrieving true particles information --

  edm::ParameterSet vertexGenerator_;
  edm::ParameterSet particleFilter_;
  FSimEvent* mySimEvent;

  // flags for the various tasks ---------------------------

  /// process particles on/off
  bool   processParticles_;

  /// verbose ?
  bool   verbose_;

};  

#endif
