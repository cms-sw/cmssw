#ifndef PFTrackTransformer_H
#define PFTrackTransformer_H
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "RecoParticleFlow/PFAlgo/interface/PFGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
typedef std::pair<Trajectory*, reco::GsfTrack*> AlGsfProduct; 
class PFTrackTransformer{
  typedef TrajectoryStateOnSurface TSOS;
 public:
  PFTrackTransformer(const MagneticField * magField);
  ~PFTrackTransformer();
  reco::PFRecTrack  producePFtrackKf(AlgoProduct&,
				     reco::PFRecTrack::AlgoType_t,
				     int index);

  reco::PFRecTrack  producePFtrackKf(Trajectory * traj,
				     reco::GsfTrack *gtrack,
				     reco::PFRecTrack::AlgoType_t,
				     int index);

  TrajectoryStateOnSurface getStateOnSurface(PFGeometry::Surface_t iSurf, 
					     const TrajectoryStateOnSurface& tsos, 
					     const Propagator* propagator, int& side);
 private:
  void addPoints(); 
  const AnalyticalPropagator *fwdPropagator;
  const AnalyticalPropagator *bkwdPropagator;
  math::XYZTLorentzVector momClosest;
  math::XYZPoint posClosest;
  reco::PFRecTrack track;
  Trajectory *tj;
};

#endif
