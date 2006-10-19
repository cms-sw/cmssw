/** \file RigidBodyAlignmentParameters.cc
 *
 *  $Date: 2005/07/26 10:13:49 $
 *  $Revision: 1.1 $
 */

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignmentParametrization/interface/KarimakiAlignmentDerivatives.h"
#include "Alignment/CommonAlignmentParametrization/interface/AlignmentTransformations.h"

// This class's header 

#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"

//__________________________________________________________________________________________________
RigidBodyAlignmentParameters::RigidBodyAlignmentParameters(Alignable* alignable, 
							   const AlgebraicVector& parameters, 
							   const AlgebraicSymMatrix& covMatrix) :
  AlignmentParameters( alignable, parameters, covMatrix )
{
 
  if ( parameters.num_row() != N_PARAM )
	edm::LogError("BadParameters") << "Bad parameters size in constructor: "
				       << parameters.num_row() << " != " << N_PARAM;

}

//__________________________________________________________________________________________________
RigidBodyAlignmentParameters::RigidBodyAlignmentParameters(Alignable* alignable, 
							   const AlgebraicVector& parameters, 
							   const AlgebraicSymMatrix& covMatrix,
							   const std::vector<bool>& selection ) :
  AlignmentParameters( alignable, parameters, covMatrix, selection )
{  

  if ( parameters.num_row() != N_PARAM )
	edm::LogError("BadParameters") << "Bad parameters size in constructor: "
				       << parameters.num_row() << " != " << N_PARAM;

}


//__________________________________________________________________________________________________
RigidBodyAlignmentParameters* 
RigidBodyAlignmentParameters::clone( const AlgebraicVector& parameters, 
				     const AlgebraicSymMatrix& covMatrix ) const 
{

  RigidBodyAlignmentParameters* rbap = 
	new RigidBodyAlignmentParameters( alignable(), parameters, covMatrix, selector() );

  if ( userVariables() ) rbap->setUserVariables( userVariables()->clone() );
  rbap->setValid( isValid() );

  return rbap;

}

//__________________________________________________________________________________________________
RigidBodyAlignmentParameters* 
RigidBodyAlignmentParameters::cloneFromSelected( const AlgebraicVector& parameters,
						 const AlgebraicSymMatrix& covMatrix ) const
{

  RigidBodyAlignmentParameters* rbap = 
    new RigidBodyAlignmentParameters(alignable(), expandVector( parameters, selector() ),
									 expandSymMatrix( covMatrix, selector() ),
									 selector() );

  if ( userVariables() ) rbap->setUserVariables(userVariables()->clone());
  rbap->setValid(isValid());

  return rbap;

}



//__________________________________________________________________________________________________
AlgebraicMatrix 
RigidBodyAlignmentParameters::derivatives( const TrajectoryStateOnSurface& tsos, 
					   AlignableDet* alignableDet ) const
{

  AlgebraicMatrix dev = KarimakiAlignmentDerivatives()( tsos );
  return dev;

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
  for ( int irow=0; irow<nrows; ++irow )
    if (selector()[irow]) {
      for ( int icol=0; icol<ncols; ++icol ) seldev[ir2][icol] = dev[irow][icol];
      ++ir2;
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

  AlgebraicVector m_GlobalParameters(6,0);

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

  std::cout << "Contents of RigidBodyAlignmentParameters: " << std::endl;
  std::cout << "Parameters: " << theParameters << std::endl;
  std::cout << "Covariance: " << theCovariance << std::endl;

}

