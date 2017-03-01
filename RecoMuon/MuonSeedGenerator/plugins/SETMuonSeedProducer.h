#ifndef RecoMuon_MuonSeedGenerator_SETMuonSeedProducer_H
#define RecoMuon_MuonSeedGenerator_SETMuonSeedProducer_H

/** \class SETMuonSeedProducer 
     I. Bloch, E. James, S. Stoynev
  */
//---- Despite its name SET is not a Seed producer in a sense that it is supposed to
//---- give the final answer about the STA muon (no pattern recognition needed 
//---- hereafter). For exact parameters (including chi2 estimation) the measurements 
//---- provided need to be fitted properly (starting from the initial estimates also provided).
//---- Technically all this information is stored as a TrajectorySeed. SS 

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoMuon/TrackingTools/interface/RecoMuonEnumerators.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "RecoMuon/MuonSeedGenerator/src/SETFilter.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h" 
#include "RecoMuon/MuonSeedGenerator/src/SETPatternRecognition.h"
#include "RecoMuon/MuonSeedGenerator/src/SETSeedFinder.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class TrajectorySeed;
class STAFilter;

#include "FWCore/Framework/interface/Frameworkfwd.h"

class SETMuonSeedProducer : public edm::stream::EDProducer<> {
  
 public:
  typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;
  typedef std::vector<Trajectory*> TrajectoryContainer;

  /// Constructor with Parameter set 
  SETMuonSeedProducer (const edm::ParameterSet&);
  
  /// Destructor
  virtual ~SETMuonSeedProducer();
  
  // Operations
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

 protected:

 private:
  
  // Returns a vector of measurements sets (for later trajectory seed building)
  /// pre-filter
  SETFilter* filter() const {return theFilter;}
  
  //----

  //private:
  
  SETFilter* theFilter;
  void setEvent(const edm::Event&);
 
  //---- SET
  bool apply_prePruning;
  bool useSegmentsInTrajectory;
  MuonServiceProxy *theService;

  SETPatternRecognition *thePatternRecognition;
  SETSeedFinder theSeedFinder;

  edm::InputTag theBeamSpotTag;
  edm::EDGetTokenT<reco::BeamSpot> beamspotToken;

};
#endif
