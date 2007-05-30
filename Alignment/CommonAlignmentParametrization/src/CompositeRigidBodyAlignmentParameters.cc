/** \file CompositeRigidBodyAlignmentParameters.cc
 *
 *  $Date: 2005/07/26 10:13:49 $
 *  $Revision: 1.1 $
 */

#include "Alignment/CommonAlignmentParametrization/interface/KarimakiAlignmentDerivatives.h"
#include "Alignment/CommonAlignmentParametrization/interface/FrameToFrameDerivative.h"

#include "Alignment/CommonAlignmentParametrization/interface/CompositeRigidBodyAlignmentParameters.h"

//__________________________________________________________________________________________________
CompositeRigidBodyAlignmentParameters::
CompositeRigidBodyAlignmentParameters( Alignable* object, 
				       const AlgebraicVector& par, 
				       const AlgebraicSymMatrix& cov ) :
  RigidBodyAlignmentParameters(object,par,cov)
{}


//__________________________________________________________________________________________________
CompositeRigidBodyAlignmentParameters::
CompositeRigidBodyAlignmentParameters(Alignable* object, 
				      const AlgebraicVector& par, 
				      const AlgebraicSymMatrix& cov, 
				      const std::vector<bool>& sel) :
  RigidBodyAlignmentParameters(object,par,cov,sel)
{}


//__________________________________________________________________________________________________
RigidBodyAlignmentParameters* 
CompositeRigidBodyAlignmentParameters::clone( const AlgebraicVector& par, 
					      const AlgebraicSymMatrix& cov ) const 
{

  CompositeRigidBodyAlignmentParameters* rbap = 
    new CompositeRigidBodyAlignmentParameters( alignable(),par,cov, selector() );

  if ( userVariables() ) rbap->setUserVariables(userVariables()->clone());
  rbap->setValid(isValid());

  return rbap;

}

//__________________________________________________________________________________________________
RigidBodyAlignmentParameters* 
CompositeRigidBodyAlignmentParameters::
cloneFromSelected( const AlgebraicVector& par, 
		   const AlgebraicSymMatrix& cov) const
{

  CompositeRigidBodyAlignmentParameters* rbap = 
    new CompositeRigidBodyAlignmentParameters( alignable(),
					       expandVector(par,selector()),
					       expandSymMatrix(cov,selector()),
					       selector() );
  if ( userVariables() ) rbap->setUserVariables(userVariables()->clone());
  rbap->setValid(isValid());

  return rbap;

}


//__________________________________________________________________________________________________
AlgebraicMatrix 
CompositeRigidBodyAlignmentParameters::derivatives( const TrajectoryStateOnSurface& tsos, 
						    AlignableDet* alidet ) const
{

  AlgebraicMatrix dev = KarimakiAlignmentDerivatives()(tsos);
  
  // get alignable belonging to higher level structure to which these
  // alignement parameters are attached 
  Alignable* ali = alignable(); 

  // get derivative of higher level structure w.r.t. det
  FrameToFrameDerivative ftfd;
  AlgebraicMatrix framederiv = ftfd.frameToFrameDerivative( alidet, ali );

  // multiplication with derivatives w.r.t. det
  dev = framederiv * dev;

  return dev;

}


