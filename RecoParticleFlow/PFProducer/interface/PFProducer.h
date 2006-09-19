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
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "RecoParticleFlow/PFAlgo/interface/PFGeometry.h"

/**\class PFProducer 
\brief Producer for particle flow tracks, particles and 
reconstructed particles 

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

  // ----------member data ---------------------------------
  /// Get position of track on a given surface
  TrajectoryStateOnSurface 
    getStateOnSurface(PFGeometry::Surface_t iSurf, 
		      const TrajectoryStateOnSurface& tsos, 
		      const Propagator& propagator, 
		      int& side);

  /// module label for retrieving input rec tracks 
  std::string recTrackModuleLabel_;

  /// module label for retrieving PFClusters
  std::string pfClusterModuleLabel_;

  /// instance name for retrieving ECAL PFClusters
  std::string pfClusterECALInstanceName_;

  /// instance name for retrieving HCAL PFClusters
  std::string pfClusterHCALInstanceName_;

  /// instance name for retrieving PS PFClusters
  std::string pfClusterPSInstanceName_;

  /// module label for retrieving input simtrack and simvertex
  std::string simModuleLabel_;  

  /// output collection name for reconstructed tracks
  // std::string pfRecTrackCollection_;

  /// output collection name for particles
  // std::string pfParticleCollection_;

  // parameters used for track reconstruction --------------

  TrackProducerAlgorithm trackAlgo_;
  std::string       fitterName_;
  std::string       propagatorName_;
  std::string       builderName_;

  // geometry, for track and particle extrapolation --------

  //Renaud: Surfaces are now accessed from PFAlgo/interface/PFGeometry.h
/*   ReferenceCountingPointer<Surface> beamPipe_; */
/*   ReferenceCountingPointer<Surface> ps1Wall_; */
/*   ReferenceCountingPointer<Surface> ps2Wall_; */
/*   ReferenceCountingPointer<Surface> ecalInnerWall_; */
/*   ReferenceCountingPointer<Surface> hcalInnerWall_; */

  // parameters for retrieving true particles information --

  edm::ParameterSet vertexGenerator_;
  edm::ParameterSet particleFilter_;
  FSimEvent* mySimEvent;

  // flags for the various tasks ---------------------------

  /// process RecTracks on/off
  bool   processRecTracks_;
  
  /// process particles on/off
  bool   processParticles_;
  
  /// particle flow on/off
  bool   doParticleFlow_;

  // particle flow parameters ------------------------------
  int pfReconMethod_;
  
};

#endif
