/** \file RigidBodyAlignmentParameters.cc
 *
 *  Version    : $Revision: 1.14 $
 *  last update: $Date: 2008/09/02 15:08:12 $
 *  by         : $Author: flucke $
 */

#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"
#include "Alignment/CommonAlignmentParametrization/interface/FrameToFrameDerivative.h"
#include "Alignment/CommonAlignmentParametrization/interface/KarimakiAlignmentDerivatives.h"
#include "Alignment/CommonAlignmentParametrization/interface/AlignmentParametersFactory.h"
#include "CondFormats/Alignment/interface/Definitions.h"

// This class's header 
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"

//__________________________________________________________________________________________________
RigidBodyAlignmentParameters::RigidBodyAlignmentParameters(Alignable* ali, bool calcMis) :
  AlignmentParameters(ali, displacementFromAlignable(calcMis ? ali : 0),
		      AlgebraicSymMatrix(N_PARAM, 0))
{
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
RigidBodyAlignmentParameters::derivatives( const TrajectoryStateOnSurface &tsos,
					   const AlignableDetOrUnitPtr &alidet ) const
{
  const Alignable *ali = this->alignable(); // Alignable of these parameters

  if (ali == alidet) { // same alignable => same frame
    return KarimakiAlignmentDerivatives()(tsos);
  } else { // different alignable => transform into correct frame
    const AlgebraicMatrix deriv = KarimakiAlignmentDerivatives()(tsos);
    FrameToFrameDerivative ftfd;
    return ftfd.frameToFrameDerivative(alidet, ali).T() * deriv;
  }
}


//__________________________________________________________________________________________________
AlgebraicMatrix 
RigidBodyAlignmentParameters::selectedDerivatives( const TrajectoryStateOnSurface& tsos, 
						   const AlignableDetOrUnitPtr &alignableDet ) const
{
  const AlgebraicMatrix dev = this->derivatives( tsos, alignableDet );

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
  for ( int i=0;i<3;++i ) shift[i]=theData->parameters()[i];

  return shift;
}


//__________________________________________________________________________________________________
AlgebraicVector RigidBodyAlignmentParameters::rotation(void) const
{
  AlgebraicVector rot(3);
  for (int i=0;i<3;++i) rot[i] = theData->parameters()[i+3];

  return rot;
}

//__________________________________________________________________________________________________
void RigidBodyAlignmentParameters::apply()
{
  Alignable *alignable = this->alignable();
  if (!alignable) {
    throw cms::Exception("BadParameters") 
      << "RigidBodyAlignmentParameters::apply: parameters without alignable";
  }
  
  // Translation in local frame
  AlgebraicVector shift = this->translation(); // fixme: should be LocalVector

  // Translation local->global
  align::LocalVector lv(shift[0], shift[1], shift[2]);
  alignable->move( alignable->surface().toGlobal(lv) );

  // Rotation in local frame
  align::EulerAngles angles = this->rotation();
  // original code:
  //  alignable->rotateInLocalFrame( align::toMatrix(angles) );
  // correct for rounding errors:
  align::RotationType rot = alignable->surface().toGlobal( align::toMatrix(angles) );
  align::rectify(rot);
  alignable->rotateInGlobalFrame(rot);
}

//__________________________________________________________________________________________________
int RigidBodyAlignmentParameters::type() const
{
  return AlignmentParametersFactory::kRigidBody;
}

//__________________________________________________________________________________________________
AlgebraicVector RigidBodyAlignmentParameters::globalParameters(void) const
{
  AlgebraicVector m_GlobalParameters(N_PARAM, 0);

  const AlgebraicVector shift = translation(); // fixme: should return LocalVector

  const align::LocalVector lv(shift[0], shift[1], shift[2]);
  const align::GlobalVector dg = theAlignable->surface().toGlobal(lv);

  m_GlobalParameters[0] = dg.x();
  m_GlobalParameters[1] = dg.y();
  m_GlobalParameters[2] = dg.z();

  const align::EulerAngles eulerglob = theAlignable->surface().toGlobal( rotation() );

  m_GlobalParameters[3]=eulerglob(1);
  m_GlobalParameters[4]=eulerglob(2);
  m_GlobalParameters[5]=eulerglob(3);

  return m_GlobalParameters;
}


//__________________________________________________________________________________________________
void RigidBodyAlignmentParameters::print(void) const
{

  std::cout << "Contents of RigidBodyAlignmentParameters:"
            << "\nParameters: " << theData->parameters()
            << "\nCovariance: " << theData->covariance() << std::endl;
}


//__________________________________________________________________________________________________
AlgebraicVector RigidBodyAlignmentParameters::displacementFromAlignable(const Alignable* ali)
{
  AlgebraicVector displacement(N_PARAM);

  if (ali) {
    const align::RotationType& dR = ali->rotation();
    
    const align::LocalVector shifts( ali->globalRotation() * 
				     ( dR.transposed() * ali->displacement().basicVector() ) );

    const align::EulerAngles angles = align::toAngles( ali->surface().toLocal(dR) );

    displacement[0] = shifts.x();
    displacement[1] = shifts.y();
    displacement[2] = shifts.z();
    displacement[3] = angles(1);
    displacement[4] = angles(2);
    displacement[5] = angles(3);
  }

  return displacement;
}
