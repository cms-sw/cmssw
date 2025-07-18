#ifndef MuonAnalysis_MuonAssociators_interface_PropagateToMuon_h
#define MuonAnalysis_MuonAssociators_interface_PropagateToMuon_h
//
//

/**
  \class    PropagateToMuon PropagateToMuon.h "MuonAnalysis/MuonAssociators/interface/PropagateToMuon.h" 

  \brief Propagate an object (usually a track) to the second (default) or first muon station.

*/

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "MuonAnalysis/MuonAssociators/interface/trackStateEnums.h"

class IdealMagneticFieldRecord;
class TrackingComponentsRecord;
class MuonRecoGeometryRecord;

class PropagateToMuon {
public:
  explicit PropagateToMuon() {}
  explicit PropagateToMuon(edm::ESHandle<MagneticField>,
                           edm::ESHandle<Propagator>,
                           edm::ESHandle<Propagator>,
                           edm::ESHandle<Propagator>,
                           edm::ESHandle<MuonDetLayerGeometry>,
                           bool,
                           bool,
                           bool,
                           WhichTrack,
                           WhichState,
                           bool,
                           bool);
  ~PropagateToMuon() {}

  /// Extrapolate reco::Track to the muon station 2, return an invalid TSOS if it fails
  TrajectoryStateOnSurface extrapolate(const reco::Track &tk) const { return extrapolate(startingState(tk)); }

  /// Extrapolate reco::Candidate to the muon station 2, return an invalid TSOS if it fails
  TrajectoryStateOnSurface extrapolate(const reco::Candidate &tk) const { return extrapolate(startingState(tk)); }

  /// Extrapolate a SimTrack to the muon station 2, return an invalid TSOS if it fails; needs the SimVertices too, to know where to start from
  /// Note: it will throw an exception if the SimTrack has no vertex.
  //  don't ask me why SimVertexContainer is in edm namespace
  TrajectoryStateOnSurface extrapolate(const SimTrack &tk, const edm::SimVertexContainer &vtxs) const {
    return extrapolate(startingState(tk, vtxs));
  }

  /// Extrapolate a FreeTrajectoryState to the muon station 2, return an invalid TSOS if it fails
  TrajectoryStateOnSurface extrapolate(const FreeTrajectoryState &state) const;

private:
  // needed services for track propagation
  edm::ESHandle<MagneticField> magfield_;
  edm::ESHandle<Propagator> propagator_, propagatorAny_, propagatorOpposite_;
  edm::ESHandle<MuonDetLayerGeometry> muonGeometry_;

  bool useSimpleGeometry_ = false;

  bool useMB2_ = false;

  /// Fallback to ME1 if propagation to ME2 fails
  bool fallbackToME1_ = false;

  /// Labels for input collections
  WhichTrack whichTrack_ = None;
  WhichState whichState_ = AtVertex;

  /// for cosmics, some things change: the along-opposite is not in-out, nor the innermost/outermost states are in-out really
  bool cosmicPropagation_ = false;

  bool useMB2InOverlap_ = false;

  // simplified geometry for track propagation
  const BoundCylinder *barrelCylinder_ = nullptr;
  const BoundDisk *endcapDiskPos_[3] = {nullptr, nullptr, nullptr};
  const BoundDisk *endcapDiskNeg_[3] = {nullptr, nullptr, nullptr};
  double barrelHalfLength_ = 0.;
  std::pair<float, float> endcapRadii_[3] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};

  /// Starting state for the propagation
  FreeTrajectoryState startingState(const reco::Candidate &reco) const;

  /// Starting state for the propagation
  FreeTrajectoryState startingState(const reco::Track &tk) const;

  /// Starting state for the propagation
  FreeTrajectoryState startingState(const SimTrack &tk, const edm::SimVertexContainer &vtxs) const;

  /// Get the best TSOS on one of the chambres of this DetLayer, or an invalid TSOS if none match
  TrajectoryStateOnSurface getBestDet(const TrajectoryStateOnSurface &tsos, const DetLayer *station) const;
};

#endif
