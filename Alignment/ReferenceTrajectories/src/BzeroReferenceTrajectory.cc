
#include "Alignment/ReferenceTrajectories/interface/BzeroReferenceTrajectory.h"

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h" 
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"


BzeroReferenceTrajectory::BzeroReferenceTrajectory(const TrajectoryStateOnSurface& tsos,
                                                   const TransientTrackingRecHit::ConstRecHitContainer& recHits,
                                                   const MagneticField *magField,
                                                   const reco::BeamSpot& beamSpot,
                                                   const ReferenceTrajectoryBase::Config& config) :
  ReferenceTrajectory(tsos.localParameters().mixedFormatVector().kSize, recHits.size(), config),
  theMomentumEstimate(config.momentumEstimate)
{
  // no check against magField == 0

  // No estimate for momentum of cosmics available -> set to default value.
  theParameters = asHepVector(tsos.localParameters().mixedFormatVector());
  theParameters[0] = 1./theMomentumEstimate;

  LocalTrajectoryParameters locParamWithFixedMomentum( asSVector<5>(theParameters),
						       tsos.localParameters().pzSign(),
						       tsos.localParameters().charge() );

  const TrajectoryStateOnSurface refTsosWithFixedMomentum(locParamWithFixedMomentum, tsos.localError(),
							  tsos.surface(), magField,
							  surfaceSide(config.propDir));

  if (config.hitsAreReverse)
  {
    TransientTrackingRecHit::ConstRecHitContainer fwdRecHits;
    fwdRecHits.reserve(recHits.size());

    for (TransientTrackingRecHit::ConstRecHitContainer::const_reverse_iterator it=recHits.rbegin(); it != recHits.rend(); ++it)
      fwdRecHits.push_back(*it);

    theValidityFlag = this->construct(refTsosWithFixedMomentum, fwdRecHits, magField, beamSpot);
  } else {
    theValidityFlag = this->construct(refTsosWithFixedMomentum, recHits, magField, beamSpot);
  }

  // Exclude momentum from the parameters and also the derivatives of the measurements w.r.t. the momentum.
  theParameters = theParameters.sub( 2, 5 );
  theDerivatives = theDerivatives.sub( 1, theDerivatives.num_row(), 2, theDerivatives.num_col() );
}
