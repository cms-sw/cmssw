/** \file BeamSpotAlignmentParameters.cc
 *
 *  Version    : $Revision: 1.1 $
 *  last update: $Date: 2010/09/10 11:16:39 $
 *  by         : $Author: mussgill $
 */

#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"
#include "Alignment/CommonAlignmentParametrization/interface/FrameToFrameDerivative.h"
#include "Alignment/CommonAlignmentParametrization/interface/BeamSpotAlignmentDerivatives.h"
#include "Alignment/CommonAlignmentParametrization/interface/AlignmentParametersFactory.h"
#include "CondFormats/Alignment/interface/Definitions.h"

// This class's header 
#include "Alignment/CommonAlignmentParametrization/interface/BeamSpotAlignmentParameters.h"

//__________________________________________________________________________________________________
BeamSpotAlignmentParameters::BeamSpotAlignmentParameters(Alignable* ali, bool calcMis) :
  AlignmentParameters(ali, displacementFromAlignable(calcMis ? ali : 0),
		      AlgebraicSymMatrix(N_PARAM, 0))
{

}

//__________________________________________________________________________________________________
BeamSpotAlignmentParameters::BeamSpotAlignmentParameters(Alignable* alignable, 
							 const AlgebraicVector& parameters, 
							 const AlgebraicSymMatrix& covMatrix) :
  AlignmentParameters( alignable, parameters, covMatrix )
{  
  if (parameters.num_row() != N_PARAM) {
    throw cms::Exception("BadParameters") << "in BeamSpotAlignmentParameters(): "
                                          << parameters.num_row() << " instead of " << N_PARAM 
                                          << " parameters.";
  }
}

//__________________________________________________________________________________________________
BeamSpotAlignmentParameters::BeamSpotAlignmentParameters(Alignable* alignable, 
							 const AlgebraicVector& parameters, 
							 const AlgebraicSymMatrix& covMatrix,
							 const std::vector<bool>& selection ) :
  AlignmentParameters( alignable, parameters, covMatrix, selection )
{  
  if (parameters.num_row() != N_PARAM) {
    throw cms::Exception("BadParameters") << "in BeamSpotAlignmentParameters(): "
                                          << parameters.num_row() << " instead of " << N_PARAM 
                                          << " parameters.";
  }
}

//__________________________________________________________________________________________________
BeamSpotAlignmentParameters::~BeamSpotAlignmentParameters()
{

} 

//__________________________________________________________________________________________________
BeamSpotAlignmentParameters* 
BeamSpotAlignmentParameters::clone( const AlgebraicVector& parameters, 
				    const AlgebraicSymMatrix& covMatrix ) const 
{
  BeamSpotAlignmentParameters* rbap = 
    new BeamSpotAlignmentParameters( alignable(), parameters, covMatrix, selector());

  if (userVariables()) rbap->setUserVariables(userVariables()->clone());
  rbap->setValid(isValid());

  return rbap;
}

//__________________________________________________________________________________________________
BeamSpotAlignmentParameters* 
BeamSpotAlignmentParameters::cloneFromSelected( const AlgebraicVector& parameters,
						const AlgebraicSymMatrix& covMatrix ) const
{
  BeamSpotAlignmentParameters* rbap = 
    new BeamSpotAlignmentParameters(alignable(), expandVector( parameters, selector()),
				    expandSymMatrix(covMatrix, selector()), selector());

  if ( userVariables() ) rbap->setUserVariables(userVariables()->clone());
  rbap->setValid(isValid());
  
  return rbap;
}

//__________________________________________________________________________________________________
AlgebraicMatrix 
BeamSpotAlignmentParameters::derivatives( const TrajectoryStateOnSurface &tsos,
					  const AlignableDetOrUnitPtr &alidet ) const
{
  const Alignable *ali = this->alignable(); // Alignable of these parameters

  if (ali == alidet) { // same alignable => same frame
    return BeamSpotAlignmentDerivatives()(tsos);
  } else { // different alignable => transform into correct frame
    const AlgebraicMatrix deriv = BeamSpotAlignmentDerivatives()(tsos);
    FrameToFrameDerivative ftfd;
    return ftfd.frameToFrameDerivative(alidet, ali) * deriv;
  }
}

//__________________________________________________________________________________________________
AlgebraicMatrix 
BeamSpotAlignmentParameters::selectedDerivatives( const TrajectoryStateOnSurface& tsos, 
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
AlgebraicVector BeamSpotAlignmentParameters::translation(void) const
{ 
  AlgebraicVector shift(3);
  for ( int i=0;i<2;++i ) shift[i] = theData->parameters()[i];
  shift[2] = 0.0;

  return shift;
}

//__________________________________________________________________________________________________
AlgebraicVector BeamSpotAlignmentParameters::rotation(void) const
{
  AlgebraicVector rot(3);

  double dxdz = theData->parameters()[2];  
  double dydz = theData->parameters()[3];  
  double angleY = std::atan(dxdz);
  double angleX = -std::atan(dydz);

  align::RotationType rotY( std::cos(angleY),  0., -std::sin(angleY), 
			    0.,                1.,  0.,
			    std::sin(angleY),  0.,  std::cos(angleY) );

  align::RotationType rotX( 1.,  0.,                0.,
			    0.,  std::cos(angleX),  std::sin(angleX),
			    0., -std::sin(angleX),  std::cos(angleX) );

  align::EulerAngles angles = align::toAngles(rotY * rotX);

  rot[0] = angles(1);
  rot[1] = angles(2);
  rot[2] = angles(3);

  return rot;
}

//__________________________________________________________________________________________________
void BeamSpotAlignmentParameters::apply()
{
  Alignable *alignable = this->alignable();
  if (!alignable) {
    throw cms::Exception("BadParameters") 
      << "BeamSpotAlignmentParameters::apply: parameters without alignable";
  }
  
  // Translation in local frame
  AlgebraicVector shift = this->translation(); // fixme: should be LocalVector
  
  // Translation local->global
  align::GlobalVector gv(shift[0], shift[1], shift[2]);
  alignable->move( gv );

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
int BeamSpotAlignmentParameters::type() const
{
  return AlignmentParametersFactory::kBeamSpot;
}

//__________________________________________________________________________________________________
AlgebraicVector BeamSpotAlignmentParameters::globalParameters(void) const
{
  AlgebraicVector m_GlobalParameters(N_PARAM, 0);

  const AlgebraicVector shift = translation(); // fixme: should return LocalVector

  const align::GlobalVector dg(shift[0], shift[1], 0);
  
  m_GlobalParameters[0] = dg.x();
  m_GlobalParameters[1] = dg.y();
  
  align::LocalVector lv(0.0, 0.0, 1.0);
  align::GlobalVector gv = theAlignable->surface().toGlobal(lv);
  
  double dxdz = gv.x()/gv.z();
  double dydz = gv.x()/gv.z();
  
  m_GlobalParameters[2] = dxdz;
  m_GlobalParameters[3] = dydz;
  
  return m_GlobalParameters;
}

//__________________________________________________________________________________________________
void BeamSpotAlignmentParameters::print(void) const
{

  std::cout << "Contents of BeamSpotAlignmentParameters:"
            << "\nParameters: " << theData->parameters()
            << "\nCovariance: " << theData->covariance() << std::endl;
}

//__________________________________________________________________________________________________
AlgebraicVector BeamSpotAlignmentParameters::displacementFromAlignable(const Alignable* ali)
{
  AlgebraicVector displacement(N_PARAM);

  if (ali) {
    const align::RotationType& dR = ali->rotation();
    
    const align::LocalVector shifts( ali->globalRotation() * 
				     ( dR.transposed() * ali->displacement().basicVector() ) );
    
    align::GlobalVector gv(0.0, 0.0, 1.0);
    align::LocalVector lv(dR.transposed() * gv.basicVector());
    
    displacement[0] = shifts.x();
    displacement[1] = shifts.y();
    displacement[2] = lv.x()/lv.z();
    displacement[3] = lv.y()/lv.z();
  }

  return displacement;
}
