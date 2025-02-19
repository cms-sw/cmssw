/** \file BowedSurfaceAlignmentParameters.cc
 *
 *  Version    : $Revision: 1.1 $
 *  last update: $Date: 2010/10/26 20:41:08 $
 *  by         : $Author: flucke $
 */

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"
#include "Alignment/CommonAlignmentParametrization/interface/KarimakiAlignmentDerivatives.h"
#include "Alignment/CommonAlignmentParametrization/interface/AlignmentParametersFactory.h"
#include "CondFormats/Alignment/interface/Definitions.h"
#include "Geometry/CommonTopologies/interface/BowedSurfaceDeformation.h"


// This class's header 
#include "Alignment/CommonAlignmentParametrization/interface/BowedSurfaceAlignmentParameters.h"

#include <iostream>
//_________________________________________________________________________________________________
BowedSurfaceAlignmentParameters::BowedSurfaceAlignmentParameters(Alignable *ali) :
  AlignmentParameters(ali, AlgebraicVector(N_PARAM), AlgebraicSymMatrix(N_PARAM, 0))
{
}

//_________________________________________________________________________________________________
BowedSurfaceAlignmentParameters
::BowedSurfaceAlignmentParameters(Alignable *alignable, 
				  const AlgebraicVector &parameters, 
				  const AlgebraicSymMatrix &covMatrix) :
  AlignmentParameters(alignable, parameters, covMatrix)
{
  if (parameters.num_row() != N_PARAM) {
    throw cms::Exception("BadParameters") << "in BowedSurfaceAlignmentParameters(): "
                                          << parameters.num_row() << " instead of " << N_PARAM 
                                          << " parameters.";
  }
}

//_________________________________________________________________________________________________
BowedSurfaceAlignmentParameters
::BowedSurfaceAlignmentParameters(Alignable *alignable, 
				  const AlgebraicVector &parameters, 
				  const AlgebraicSymMatrix &covMatrix,
				  const std::vector<bool> &selection) :
  AlignmentParameters(alignable, parameters, covMatrix, selection)
{  
  if (parameters.num_row() != N_PARAM) {
    throw cms::Exception("BadParameters") << "in BowedSurfaceAlignmentParameters(): "
                                          << parameters.num_row() << " instead of " << N_PARAM 
                                          << " parameters.";
  }
}

//_________________________________________________________________________________________________
BowedSurfaceAlignmentParameters* 
BowedSurfaceAlignmentParameters::clone(const AlgebraicVector &parameters, 
				       const AlgebraicSymMatrix &covMatrix) const 
{
  BowedSurfaceAlignmentParameters* rbap = 
    new BowedSurfaceAlignmentParameters(this->alignable(), parameters, covMatrix, selector());

  if (this->userVariables()) rbap->setUserVariables(this->userVariables()->clone());
  rbap->setValid(this->isValid());

  return rbap;
}

//_________________________________________________________________________________________________
BowedSurfaceAlignmentParameters* 
BowedSurfaceAlignmentParameters::cloneFromSelected(const AlgebraicVector &parameters,
						   const AlgebraicSymMatrix &covMatrix) const
{
  return this->clone(this->expandVector(parameters, this->selector()),
		     this->expandSymMatrix(covMatrix, this->selector()));

}

//_________________________________________________________________________________________________
AlgebraicMatrix 
BowedSurfaceAlignmentParameters::derivatives(const TrajectoryStateOnSurface &tsos,
					     const AlignableDetOrUnitPtr &alidet) const
{
  const Alignable *ali = this->alignable(); // Alignable of these parameters

  if (ali == alidet) {
    const AlignableSurface &surf = ali->surface();
    return BowedDerivs()(tsos, surf.width(), surf.length());
  } else {
    // We could give this a meaning by applying frame-to-frame derivatives 
    // to the first six parameters (be careful that alpha and beta changed
    // their scale and switched their place compared to RigidBody!) and
    // keep the remaining three untouched in local meaning.
    // In this way we could do higher level alignment and determine 'average'
    // surface structures for the components.
    throw cms::Exception("MisMatch")
      << "BowedSurfaceAlignmentParameters::derivatives: The hit alignable must match the "
      << "aligned one (i.e. bowed surface parameters cannot be used for composed alignables)\n";
    return AlgebraicMatrix(N_PARAM, 2); // please compiler
  }

}

//_________________________________________________________________________________________________
align::LocalVector BowedSurfaceAlignmentParameters::translation() const
{ 
  // align::LocalVector uses double while LocalVector uses float only!
  const AlgebraicVector &params = theData->parameters();
  return align::LocalVector(params[dx], params[dy], params[dz]);
}


//_________________________________________________________________________________________________
align::EulerAngles BowedSurfaceAlignmentParameters::rotation() const
{
  const AlgebraicVector &params = theData->parameters();
  const Alignable *alignable = this->alignable();
  const AlignableSurface &surface = alignable->surface();

  align::EulerAngles eulerAngles(3);
  // Note that dslopeX <-> -beta and dslopeY <-> alpha:
  // Should we use atan of these values? Anyway it is small...
  eulerAngles[0] =  params[dslopeY] * 2. / surface.length();
  eulerAngles[1] = -params[dslopeX] * 2. / surface.width();
  const double aScale = BowedDerivs::gammaScale(surface.width(), surface.length());
  eulerAngles[2] =  params[drotZ] / aScale;

  return eulerAngles;
}

//_________________________________________________________________________________________________
void BowedSurfaceAlignmentParameters::apply()
{
  Alignable *alignable = this->alignable();
  if (!alignable) {
    throw cms::Exception("BadParameters") 
      << "BowedSurfaceAlignmentParameters::apply: parameters without alignable";
  }
  
  // Get translation in local frame, transform to global and apply:
  alignable->move(alignable->surface().toGlobal(this->translation()));

  // Rotation in local frame
  const align::EulerAngles angles(this->rotation());
  // original code:
  //  alignable->rotateInLocalFrame( align::toMatrix(angles) );
  // correct for rounding errors:
  align::RotationType rot(alignable->surface().toGlobal(align::toMatrix(angles)));
  align::rectify(rot);
  alignable->rotateInGlobalFrame(rot);

  const AlgebraicVector &params = theData->parameters();
  const BowedSurfaceDeformation deform(params[dsagittaX], params[dsagittaXY], params[dsagittaY]);

  // FIXME: true to propagate down?
  //        Needed for hierarchy with common deformation parameter,
  //        but that is not possible now anyway.
  alignable->addSurfaceDeformation(&deform, false);
}

//_________________________________________________________________________________________________
int BowedSurfaceAlignmentParameters::type() const
{
  return AlignmentParametersFactory::kBowedSurface;
}

//_________________________________________________________________________________________________
void BowedSurfaceAlignmentParameters::print() const
{
  std::cout << "Contents of BowedSurfaceAlignmentParameters:"
            << "\nParameters: " << theData->parameters()
            << "\nCovariance: " << theData->covariance() << std::endl;
}
