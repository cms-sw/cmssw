/** \file RigidBodyAlignmentParameters.cc
 *
 *  $Date: 2006/10/19 14:20:59 $
 *  $Revision: 1.2 $
 */

#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/CommonAlignmentParametrization/interface/KarimakiAlignmentDerivatives.h"
#include "Alignment/CommonAlignmentParametrization/interface/AlignmentTransformations.h"

// This class's header 

#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"

//__________________________________________________________________________________________________
RigidBodyAlignmentParameters::RigidBodyAlignmentParameters(Alignable* ali) :
  AlignmentParameters(ali, AlgebraicVector(N_PARAM), AlgebraicSymMatrix(N_PARAM, 0))
{
  AlignmentTransformations trafo; // why does it not work with const?
  const Alignable::RotationType diffRot    (ali->rotation());// a.transform(b) means a = b * a
  const Alignable::RotationType globRotOrig(diffRot.transposed().transform(ali->globalRotation()));
  const AlgebraicVector         globShift  (trafo.algebraicVector(ali->displacement()));
  const AlgebraicVector         locShift   (trafo.algebraicMatrix(globRotOrig) * globShift);
  const Alignable::RotationType locRot     (trafo.globalToLocalMatrix(diffRot, globRotOrig));
  const AlgebraicVector         angles     (trafo.eulerAngles(locRot, 0));

  for (int i = 0; i < N_PARAM; ++i) {
    theParameters[i] = (i < dalpha ? locShift[i] : angles[i-dalpha]);
  }
}

//__________________________________________________________________________________________________
RigidBodyAlignmentParameters::RigidBodyAlignmentParameters(Alignable* alignable, 
							   const AlgebraicVector& parameters, 
							   const AlgebraicSymMatrix& covMatrix) :
  AlignmentParameters( alignable, parameters, covMatrix )
{
 
  if (parameters.num_row() != N_PARAM) {
    throw cms::Exception("BadParameters") << "in RigidBodyAlignmentParameters(): "
                                          << parameters.num_row() << " instead of " << N_PARAM 
                                          << " parameters.";
  }
}

//__________________________________________________________________________________________________
RigidBodyAlignmentParameters::RigidBodyAlignmentParameters(Alignable* alignable, 
                                                           const AlgebraicVector& parameters, 
                                                           const AlgebraicSymMatrix& covMatrix,
                                                           const std::vector<bool>& selection ) :
  AlignmentParameters( alignable, parameters, covMatrix, selection )
{  
  if (parameters.num_row() != N_PARAM) {
    throw cms::Exception("BadParameters") << "in RigidBodyAlignmentParameters(): "
                                          << parameters.num_row() << " instead of " << N_PARAM 
                                          << " parameters.";
  }
}

//__________________________________________________________________________________________________
RigidBodyAlignmentParameters* 
RigidBodyAlignmentParameters::clone( const AlgebraicVector& parameters, 
				     const AlgebraicSymMatrix& covMatrix ) const 
{
  RigidBodyAlignmentParameters* rbap = 
    new RigidBodyAlignmentParameters( alignable(), parameters, covMatrix, selector());

  if (userVariables()) rbap->setUserVariables(userVariables()->clone());
  rbap->setValid(isValid());

  return rbap;
}

//__________________________________________________________________________________________________
RigidBodyAlignmentParameters* 
RigidBodyAlignmentParameters::cloneFromSelected( const AlgebraicVector& parameters,
						 const AlgebraicSymMatrix& covMatrix ) const
{
  RigidBodyAlignmentParameters* rbap = 
    new RigidBodyAlignmentParameters(alignable(), expandVector( parameters, selector()),
                                     expandSymMatrix(covMatrix, selector()), selector());

  if ( userVariables() ) rbap->setUserVariables(userVariables()->clone());
  rbap->setValid(isValid());

  return rbap;
}



//__________________________________________________________________________________________________
AlgebraicMatrix 
RigidBodyAlignmentParameters::derivatives( const TrajectoryStateOnSurface& tsos, 
					   AlignableDet* alignableDet ) const
{
  return KarimakiAlignmentDerivatives()(tsos);
}


//__________________________________________________________________________________________________
AlgebraicMatrix 
RigidBodyAlignmentParameters::selectedDerivatives( const TrajectoryStateOnSurface& tsos, 
						   AlignableDet* alignableDet ) const
{
  AlgebraicMatrix dev = derivatives( tsos, alignableDet );

  int ncols  = dev.num_col();
  int nrows  = dev.num_row();
  int nsel   = numSelected();

  AlgebraicMatrix seldev( nsel, ncols );

  int ir2=0;
  for ( int irow=0; irow<nrows; ++irow ) {
    if (selector()[irow]) {
      for ( int icol=0; icol<ncols; ++icol ) seldev[ir2][icol] = dev[irow][icol];
      ++ir2;
    }
  }

  return seldev;
}


//__________________________________________________________________________________________________
AlgebraicVector RigidBodyAlignmentParameters::translation(void) const
{ 
  AlgebraicVector shift(3);
  for ( int i=0;i<3;++i ) shift[i]=theParameters[i];

  return shift;
}


//__________________________________________________________________________________________________
AlgebraicVector RigidBodyAlignmentParameters::rotation(void) const
{
  AlgebraicVector rot(3);
  for (int i=0;i<3;++i) rot[i] = theParameters[i+3];

  return rot;
}


//__________________________________________________________________________________________________
AlgebraicVector RigidBodyAlignmentParameters::globalParameters(void) const
{
  AlgebraicVector m_GlobalParameters(N_PARAM, 0);

  AlgebraicVector shift = translation();

  LocalPoint l0   = Local3DPoint( 0.0,  0.0, 0.0 );
  LocalPoint l1   = Local3DPoint(shift[0], shift[1], shift[2]);
  GlobalPoint g0  = theAlignable->surface().toGlobal( l0);
  GlobalPoint g1  = theAlignable->surface().toGlobal( l1);
  GlobalVector dg = g1-g0;

  m_GlobalParameters[0] = dg.x();
  m_GlobalParameters[1] = dg.y();
  m_GlobalParameters[2] = dg.z();

  AlgebraicVector eulerloc = rotation();
  AlignmentTransformations alignmentTransformation;
  Surface::RotationType detrot = theAlignable->surface().rotation();
  AlgebraicVector eulerglob = alignmentTransformation.localToGlobalEulerAngles( eulerloc, detrot );

  m_GlobalParameters[3]=eulerglob[0];
  m_GlobalParameters[4]=eulerglob[1];
  m_GlobalParameters[5]=eulerglob[2];

  return m_GlobalParameters;
}


//__________________________________________________________________________________________________
void RigidBodyAlignmentParameters::print(void) const
{

  std::cout << "Contents of RigidBodyAlignmentParameters:"
            << "\nParameters: " << theParameters
            << "\nCovariance: " << theCovariance << std::endl;
}

