/** \file RigidBodyAlignmentParameters.cc
 *
 *  Version    : $Revision: 1.1 $
 *  last update: $Date: 2008/12/12 15:58:07 $
 *  by         : $Author: pablom $
 */

#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"
#include "Alignment/CommonAlignmentParametrization/interface/FrameToFrameDerivative.h"
#include "Alignment/CommonAlignmentParametrization/interface/SegmentAlignmentDerivatives4D.h"
#include "Alignment/CommonAlignmentParametrization/interface/AlignmentParametersFactory.h"
#include "CondFormats/Alignment/interface/Definitions.h"

// This class's header 
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters4D.h"

//__________________________________________________________________________________________________
AlgebraicMatrix 
RigidBodyAlignmentParameters4D::derivatives( const TrajectoryStateOnSurface &tsos,
					   const AlignableDetOrUnitPtr &alidet ) const
{
  const Alignable *ali = this->alignable(); // Alignable of these parameters

  if (ali == alidet) { // same alignable => same frame
    return SegmentAlignmentDerivatives4D()(tsos);
  } else { // different alignable => transform into correct frame
    const AlgebraicMatrix deriv = SegmentAlignmentDerivatives4D()(tsos);
    FrameToFrameDerivative ftfd;
    return ftfd.frameToFrameDerivative(alidet, ali) * deriv;
  }
}

//__________________________________________________________________________________________________
RigidBodyAlignmentParameters4D*
RigidBodyAlignmentParameters4D::clone( const AlgebraicVector& parameters,
                                     const AlgebraicSymMatrix& covMatrix ) const
{
  RigidBodyAlignmentParameters4D* rbap =
    new RigidBodyAlignmentParameters4D( alignable(), parameters, covMatrix, selector());

  if (userVariables()) rbap->setUserVariables(userVariables()->clone());
  rbap->setValid(isValid());

  return rbap;
}

//__________________________________________________________________________________________________
RigidBodyAlignmentParameters4D*
RigidBodyAlignmentParameters4D::cloneFromSelected( const AlgebraicVector& parameters,
                                                 const AlgebraicSymMatrix& covMatrix ) const
{
  RigidBodyAlignmentParameters4D* rbap =
    new RigidBodyAlignmentParameters4D(alignable(), expandVector( parameters, selector()),
                                     expandSymMatrix(covMatrix, selector()), selector());

  if ( userVariables() ) rbap->setUserVariables(userVariables()->clone());
  rbap->setValid(isValid());

  return rbap;
}



//__________________________________________________________________________________________________
int RigidBodyAlignmentParameters4D::type() const
{
  return AlignmentParametersFactory::kRigidBody4D;
}

