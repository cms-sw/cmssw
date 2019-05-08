#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayFitter.h"

TwoBodyDecayFitter::TwoBodyDecayFitter(const edm::ParameterSet &config)
    : theVertexFinder(new DefaultLinearizationPointFinder()),
      theLinPointFinder(new TwoBodyDecayLinearizationPointFinder(config)),
      theEstimator(new TwoBodyDecayEstimator(config)) {}

TwoBodyDecayFitter::TwoBodyDecayFitter(const edm::ParameterSet &config,
                                       const LinearizationPointFinder *vf,
                                       const TwoBodyDecayLinearizationPointFinder *lpf,
                                       const TwoBodyDecayEstimator *est)
    : theVertexFinder(vf->clone()), theLinPointFinder(lpf->clone()), theEstimator(est->clone()) {}

TwoBodyDecayFitter::~TwoBodyDecayFitter(void) {}

const TwoBodyDecay TwoBodyDecayFitter::estimate(const std::vector<reco::TransientTrack> &tracks,
                                                const TwoBodyDecayVirtualMeasurement &vm) const {
  // get geometrical linearization point
  GlobalPoint linVertex = theVertexFinder->getLinearizationPoint(tracks);

  // create linearized track states
  std::vector<RefCountedLinearizedTrackState> linTracks;
  linTracks.push_back(theLinTrackStateFactory.linearizedTrackState(linVertex, tracks[0]));
  linTracks.push_back(theLinTrackStateFactory.linearizedTrackState(linVertex, tracks[1]));

  // get full linearization point (geomatrical & kinematical)
  const TwoBodyDecayParameters linPoint =
      theLinPointFinder->getLinearizationPoint(linTracks, vm.primaryMass(), vm.secondaryMass());

  // make the fit
  return theEstimator->estimate(linTracks, linPoint, vm);
}

const TwoBodyDecay TwoBodyDecayFitter::estimate(const std::vector<reco::TransientTrack> &tracks,
                                                const std::vector<TrajectoryStateOnSurface> &tsos,
                                                const TwoBodyDecayVirtualMeasurement &vm) const {
  // get geometrical linearization point
  std::vector<FreeTrajectoryState> freeTrajStates;
  freeTrajStates.push_back(*tsos[0].freeState());
  freeTrajStates.push_back(*tsos[1].freeState());
  GlobalPoint linVertex = theVertexFinder->getLinearizationPoint(freeTrajStates);

  // create linearized track states
  std::vector<RefCountedLinearizedTrackState> linTracks;
  linTracks.push_back(theLinTrackStateFactory.linearizedTrackState(linVertex, tracks[0], tsos[0]));
  linTracks.push_back(theLinTrackStateFactory.linearizedTrackState(linVertex, tracks[1], tsos[1]));

  // get full linearization point (geomatrical & kinematical)
  const TwoBodyDecayParameters linPoint =
      theLinPointFinder->getLinearizationPoint(linTracks, vm.primaryMass(), vm.secondaryMass());

  // make the fit
  return theEstimator->estimate(linTracks, linPoint, vm);
}
