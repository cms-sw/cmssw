/** \file TwoBowedSurfacesAlignmentParameters.cc
 *
 *  Version    : $Revision: 1.14 $
 *  last update: $Date: 2008/09/02 15:08:12 $
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
//#include "DataFormats/Alignment/interface/SurfaceDeformation.h"
#include "Geometry/CommonTopologies/interface/TwoBowedSurfacesDeformation.h"

// This class's header 
#include "Alignment/CommonAlignmentParametrization/interface/TwoBowedSurfacesAlignmentParameters.h"

#include <math.h>
#include <iostream>
//_________________________________________________________________________________________________
TwoBowedSurfacesAlignmentParameters::TwoBowedSurfacesAlignmentParameters(Alignable *ali) :
  AlignmentParameters(ali, AlgebraicVector(N_PARAM), AlgebraicSymMatrix(N_PARAM, 0)),
  ySplit_(this->ySplitFromAlignable(ali))
{
}

//_________________________________________________________________________________________________
TwoBowedSurfacesAlignmentParameters
::TwoBowedSurfacesAlignmentParameters(Alignable *alignable, 
				      const AlgebraicVector &parameters, 
				      const AlgebraicSymMatrix &covMatrix) :
  AlignmentParameters(alignable, parameters, covMatrix),
  ySplit_(this->ySplitFromAlignable(alignable))
{
  if (parameters.num_row() != N_PARAM) {
    throw cms::Exception("BadParameters") << "in TwoBowedSurfacesAlignmentParameters(): "
                                          << parameters.num_row() << " instead of " << N_PARAM 
                                          << " parameters.";
  }
}

//_________________________________________________________________________________________________
TwoBowedSurfacesAlignmentParameters
::TwoBowedSurfacesAlignmentParameters(Alignable *alignable, 
				      const AlgebraicVector &parameters, 
				      const AlgebraicSymMatrix &covMatrix,
				      const std::vector<bool> &selection) :
  AlignmentParameters(alignable, parameters, covMatrix, selection),
  ySplit_(this->ySplitFromAlignable(alignable))
{
  if (parameters.num_row() != N_PARAM) {
    throw cms::Exception("BadParameters") << "in TwoBowedSurfacesAlignmentParameters(): "
                                          << parameters.num_row() << " instead of " << N_PARAM 
                                          << " parameters.";
  }
}

//_________________________________________________________________________________________________
TwoBowedSurfacesAlignmentParameters* 
TwoBowedSurfacesAlignmentParameters::clone(const AlgebraicVector &parameters, 
					   const AlgebraicSymMatrix &covMatrix) const 
{
  TwoBowedSurfacesAlignmentParameters* rbap = 
    new TwoBowedSurfacesAlignmentParameters(this->alignable(), parameters, covMatrix, selector());

  if (this->userVariables()) rbap->setUserVariables(this->userVariables()->clone());
  rbap->setValid(this->isValid());

  return rbap;
}

//_________________________________________________________________________________________________
TwoBowedSurfacesAlignmentParameters* 
TwoBowedSurfacesAlignmentParameters::cloneFromSelected(const AlgebraicVector &parameters,
						       const AlgebraicSymMatrix &covMatrix) const
{
  return this->clone(this->expandVector(parameters, this->selector()),
		     this->expandSymMatrix(covMatrix, this->selector()));
}

//_________________________________________________________________________________________________
AlgebraicMatrix 
TwoBowedSurfacesAlignmentParameters::derivatives(const TrajectoryStateOnSurface &tsos,
						 const AlignableDetOrUnitPtr &alidet) const
{
  const Alignable *ali = this->alignable(); // Alignable of these parameters
  AlgebraicMatrix result(N_PARAM, 2); // initialised with zeros

  if (ali == alidet) {
    const AlignableSurface &surf = ali->surface();
 
    // matrix of dimension BowedDerivs::N_PARAM x 2 
    const AlgebraicMatrix derivs(BowedDerivs()(tsos, surf.width(), surf.length(),
					       true, ySplit_)); // split at ySplit_!

    // Parameters belong to surface part with y < ySplit_ or y >= ySplit_?
    const double localY = tsos.localParameters().mixedFormatVector()[4];
    const unsigned int indexOffset = (localY < ySplit_ ? 0 : dx2);
    // Copy derivatives to relevant part of result
    for (unsigned int i = BowedDerivs::dx; i < BowedDerivs::N_PARAM; ++i) {
      result[indexOffset + i][0] = derivs[i][0];
      result[indexOffset + i][1] = derivs[i][1];
    } 
  } else {
    // The following is even more difficult for TwoBowedSurfacesAlignmentParameters
    // than for BowedSurfaceAlignmentParameters where this text comes from:
    //
    // We could give this a meaning by applying frame-to-frame derivatives 
    // to the rigid body part of the parameters (be careful that alpha ~= dslopeY
    // and beta ~= -dslopeX, but with changed scale!)
    // and keep the surface structure parameters untouched in local meaning.
    // In this way we could do higher level alignment and determine 'average'
    // surface structures for the components.
    throw cms::Exception("MisMatch")
      << "TwoBowedSurfacesAlignmentParameters::derivatives: The hit alignable must match the "
      << "aligned one (i.e. bowed surface parameters cannot be used for composed alignables)\n";
  }

  return result;
}

//_________________________________________________________________________________________________
void TwoBowedSurfacesAlignmentParameters::apply()
{
  Alignable *alignable = this->alignable();
  if (!alignable) {
    throw cms::Exception("BadParameters") 
      << "TwoBowedSurfacesAlignmentParameters::apply: parameters without alignable";
  }
  
  // Some repeatedly needed variables
  const AlignableSurface &surface = alignable->surface();
  const double halfLength  = surface.length() * 0.5; // full module
  const double halfLength1 = (halfLength + ySplit_) * 0.5;        // low-y surface
  const double halfLength2 = (halfLength - ySplit_) * 0.5;        // high-y surface

  // first copy the parameters into separate parts for the two surfaces
  const AlgebraicVector &params = theData->parameters();
  std::vector<double> rigidBowPar1(BowedDerivs::N_PARAM); // 1st surface (y <  ySplit_)
  std::vector<double> rigidBowPar2(BowedDerivs::N_PARAM); // 2nd surface (y >= ySplit_)
  for (unsigned int i = 0; i < BowedDerivs::N_PARAM; ++i) {
    rigidBowPar1[i] = params[i];
    rigidBowPar2[i] = params[i + BowedDerivs::N_PARAM];
  }
  // Now adjust slopes to angles, note that dslopeX <-> -beta & dslopeY <-> alpha,
  // see BowedSurfaceAlignmentParameters::rotation(): FIXME: use atan?
  rigidBowPar1[3] =  params[dslopeY1] / halfLength1; // alpha1
  rigidBowPar2[3] =  params[dslopeY2] / halfLength2; // alpha2 
  rigidBowPar1[4] = -params[dslopeX1] / (surface.width() * 0.5); // beta1
  rigidBowPar2[4] = -params[dslopeX2] / (surface.width() * 0.5); // beta2
  // gamma is simply scaled
  const double gammaScale1 = BowedDerivs::gammaScale(surface.width(), 2.0*halfLength1);
  rigidBowPar1[5] = params[drotZ1] / gammaScale1;
//   const double gammaScale2 = std::sqrt(halfLength2 * halfLength2
// 				       + surface.width() * surface.width()/4.);
  const double gammaScale2 = BowedDerivs::gammaScale(surface.width(), 2.0*halfLength2);
  rigidBowPar2[5] = params[drotZ2] / gammaScale2;

  // Get rigid body rotations of full module as mean of the two surfaces:
  align::EulerAngles angles(3); // to become 'common' rotation in local frame
  for (unsigned int i = 0; i < 3; ++i) {
    angles[i] = (rigidBowPar1[i+3] + rigidBowPar2[i+3]) * 0.5;
  }
  // Module rotations are around other axes than the one we determined,
  // so we have to correct that the surfaces are shifted by the rotation around
  // the module axis - in linear approximation just an additional shift:
  const double yMean1 = -halfLength + halfLength1;// y of alpha1 rotation axis in module frame
  const double yMean2 =  halfLength - halfLength2;// y of alpha2 rotation axis in module frame
  rigidBowPar1[dz1] -= angles[0] * yMean1; // correct w1 for alpha
  rigidBowPar2[dz1] -= angles[0] * yMean2; // correct w2 for alpha
  // Nothing for beta1/2 since anyway both around the y-axis of the module.
  rigidBowPar1[dx1] += angles[2] * yMean1; // correct x1 for gamma
  rigidBowPar2[dx1] += angles[2] * yMean2; // correct x1 for gamma

  // Get rigid body shifts of full module as mean of the two surfaces:
  const align::LocalVector shift((rigidBowPar1[dx1] + rigidBowPar2[dx1]) * 0.5, // dx1!
				 (rigidBowPar1[dy1] + rigidBowPar2[dy1]) * 0.5, // dy1!
				 (rigidBowPar1[dz1] + rigidBowPar2[dz1]) * 0.5);// dz1!
  // Apply module shift and rotation:
  alignable->move(surface.toGlobal(shift));
  // original code:
  //  alignable->rotateInLocalFrame( align::toMatrix(angles) );
  // correct for rounding errors:
  align::RotationType rot(surface.toGlobal(align::toMatrix(angles)));
  align::rectify(rot);
  alignable->rotateInGlobalFrame(rot);

  // Fill surface structures with mean bows and half differences for all parameters:
  std::vector<align::Scalar> deformations; deformations.reserve(13);
  // first part: average bows
  deformations.push_back((params[dsagittaX1 ] + params[dsagittaX2 ]) * 0.5);
  deformations.push_back((params[dsagittaXY1] + params[dsagittaXY2]) * 0.5);
  deformations.push_back((params[dsagittaY1 ] + params[dsagittaY2 ]) * 0.5);
  // second part: half difference of all corrections
  for (unsigned int i = 0; i < BowedDerivs::N_PARAM; ++i) { 
    // sign means that we have to apply e.g.
    // - sagittaX for sensor 1: deformations[0] + deformations[9]
    // - sagittaX for sensor 2: deformations[0] - deformations[9]
    // - additional dx for sensor 1:  deformations[3]
    // - additional dx for sensor 2: -deformations[3]
    deformations.push_back((rigidBowPar1[i] - rigidBowPar2[i]) * 0.5);
  }
  // finally: keep track of where we have split the module
  deformations.push_back(ySplit_); // index is 12

  //  const SurfaceDeformation deform(SurfaceDeformation::kTwoBowedSurfaces, deformations);
  const TwoBowedSurfacesDeformation deform(deformations);

  // FIXME: true to propagate down?
  //        Needed for hierarchy with common deformation parameter,
  //        but that is not possible now anyway.
  alignable->addSurfaceDeformation(&deform, false);
}

//_________________________________________________________________________________________________
int TwoBowedSurfacesAlignmentParameters::type() const
{
  return AlignmentParametersFactory::kTwoBowedSurfaces;
}

//_________________________________________________________________________________________________
void TwoBowedSurfacesAlignmentParameters::print() const
{

  std::cout << "Contents of TwoBowedSurfacesAlignmentParameters:"
            << "\nParameters: " << theData->parameters()
            << "\nCovariance: " << theData->covariance() << std::endl;
}

//_________________________________________________________________________________________________
double TwoBowedSurfacesAlignmentParameters::ySplitFromAlignable(const Alignable *ali) const
{
  if (!ali) return 0.;

  const align::PositionType pos(ali->globalPosition());  
  const double r = pos.perp();

  // The returned numbers come from CMS-NOTE 2003/10.
  // (FIXME: according to Andreas Mussgiller, but GF has not found the page ...)
  // Note that at that time it was planned to use ST sensors for the outer TEC
  // while in the end there are only a few of them in the tracker - the others are HPK.
  // No idea whether there are subtle changes in geometry...
  if (r < 58.) { // Pixel, TIB, TID or TEC ring 1-4
    edm::LogError("Alignment") << "@SUB=TwoBowedSurfacesAlignmentParameters::ySplitFromAlignable"
			       << "There are no split modules for radii < 58, but r = " << r;
    return 0.;
  } else if (fabs(pos.z()) < 118.) { // TOB
    return 0.;
  } else if (r > 90.) { // TEC ring 7
    return 0.9;
  } else if (r > 75.) { // TEC ring 6
    return 0.6;
  } else {              // TEC ring 5 - smaller radii alreay excluded before
    return 0.9;
  }
  //  return 0.;
}
