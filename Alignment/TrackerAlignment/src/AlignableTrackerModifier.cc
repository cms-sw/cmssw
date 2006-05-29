#include "CLHEP/Random/Random.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/Randomize.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/TrackerAlignment/interface/AlignableTrackerModifier.h"


//__________________________________________________________________________________________________
// Return true if given parameter name should be propagated down
const bool AlignableTrackerModifier::isPropagated( const std::string parameterName ) const
{

  if ( parameterName == "gaussian"   || 
	   parameterName == "seed"       ||
	   parameterName == "setError"   ||
	   parameterName == "scaleError"  ) return true;

  return false;

}


//__________________________________________________________________________________________________
/// All known parameters and defaults are defined here! Returns true if modification actually applied.
bool AlignableTrackerModifier::modify( Alignable* alignable, const edm::ParameterSet& pSet )
{

  // Initialize all known parameters (according to ORCA's MisalignmentScenario.cc)
  bool   gaussian   = true;      // Use gaussian distribution (otherwise flat)
  long   seed       = 0;         // Random generator seed (default: ask service)
  bool   setError   = false;     // Apply alignment errors
  double scaleError = 1.;        // Scale to apply to alignment errors
  double phiX       = 0.;        // Rotation angle around X [rad]
  double phiY       = 0.;        // Rotation angle around Y [rad]
  double phiZ       = 0.;        // Rotation angle around Z [rad]
  double localX     = 0.;        // Local rotation angle around X [rad]
  double localY     = 0.;        // Local rotation angle around Y [rad]
  double localZ     = 0.;        // Local rotation angle around Z [rad]
  double dX         = 0.;        // X displacement [cm]
  double dY         = 0.;        // Y displacement [cm]
  double dZ         = 0.;        // Z displacement [cm]
  double twist      = 0.;        // Twist angle [rad]
  double shear      = 0.;        // Shear angle [rad]

  // Reset counter
  m_modified = 0;
  
  // Retrieve parameters
  std::ostringstream error;
  std::vector<std::string> parameterNames = pSet.getParameterNames();
  for ( std::vector<std::string>::iterator iParam = parameterNames.begin(); 
		iParam != parameterNames.end(); iParam++ )
	{
	  if      ( (*iParam) == "gaussian" ) gaussian = pSet.getParameter<bool>( *iParam );
	  else if ( (*iParam) == "setError" ) setError = pSet.getParameter<bool>( *iParam );
	  else if ( (*iParam) == "scaleError" ) scaleError = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "phiX" )     phiX     = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "phiY" )     phiY     = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "phiZ" )     phiZ     = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "localX" )   localX   = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "localY" )   localY   = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "localZ" )   localZ   = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "dX" )       dX       = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "dY" )       dY       = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "dZ" )       dZ       = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "twist" )    twist    = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "shear" )    shear    = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "seed"  )    seed     = static_cast<long>(pSet.getParameter<int>( *iParam ));
	  else if ( pSet.retrieve( *iParam ).typeCode() != 'P' )
		{ // Add unknown parameter to list
		  if ( !error.str().length()) error << "Unknown parameter name(s): ";
		  error << " " << *iParam;
		}
	}

  // Check error
  if ( error.str().length() )
	throw cms::Exception("BadConfig") << error.str();

  // Apply displacements
  if ( fabs(dX) + fabs(dY) + fabs(dZ) > 0 )
	if ( gaussian ) this->randomMove( alignable, dX, dY, dZ, seed );
	else this->randomFlatMove( alignable, dX, dY, dZ, seed );

  // Apply rotations
  if ( fabs(phiX) + fabs(phiY) + fabs(phiZ) > 0 )
	if ( gaussian ) this->randomRotate( alignable, phiX, phiY, phiZ, seed );
	else this->randomFlatRotate( alignable, phiX, phiY, phiZ, seed );

  // Apply local rotations
  if ( fabs(localX) + fabs(localY) + fabs(localZ) > 0 )
	if ( gaussian ) this->randomRotateLocal( alignable, localX, localY, localZ, seed );
	else this->randomFlatRotateLocal( alignable, localX, localY, localZ, seed );


  // Apply twist
  if ( fabs(twist) > 0 )
	edm::LogError("NotImplemented") << "Twist is not implemented yet";

  // Apply shear
  if ( fabs(shear) > 0 )
	edm::LogError("NotImplemented") << "Shear is not implemented yet";

  // Apply error
  if ( setError )
	{
	  // Alignment Position Error for flat distribution: 1 sigma
	  if ( !gaussian ) scaleError *= 0.68;

	  // Error on displacement
	  if ( fabs(dX) + fabs(dY) + fabs(dZ) > 0 )
		this->addAlignmentPositionError( alignable, scaleError*dX, scaleError*dY, scaleError*dZ );

	  // Error on rotations
	  if ( fabs(phiX) + fabs(phiY) + fabs(phiZ) > 0 )
		this->addAlignmentPositionErrorFromRotation( alignable, 
													 scaleError*phiX, scaleError*phiY, 
													 scaleError*phiZ );

	  // Error on local rotations
	  if ( fabs(localX) + fabs(localY) + fabs(localZ) > 0 )
		this->addAlignmentPositionErrorFromLocalRotation( alignable, 
														  scaleError*localX, scaleError*localY, 
														  scaleError*localZ );
	}

  return ( m_modified > 0 );
  
}



//__________________________________________________________________________________________________
/// Random gaussian move. The random number seed is taken from 'seed' if larger than zero.
/// Otherwise, a seed is generated from the RandomNumberGenerator service.
void AlignableTrackerModifier::randomMove( Alignable* alignable, 
										   float sigmaX, float sigmaY,float sigmaZ, long seed )
{

  edm::LogInfo("PrintArgs") << "move randomly with sigma " 
							<< sigmaX << " " << sigmaY << " " << sigmaZ 
							<< std::endl;
  
  DRand48Engine aDRand48Engine;
  if ( seed > 0 ) 
	aDRand48Engine.setSeed( seed, 0 );
  else
	{
	  edm::Service<edm::RandomNumberGenerator> rng;
	  aDRand48Engine.setSeed( rng->mySeed(), 0 );
	}

  RandGauss aGaussObjX( aDRand48Engine, 0., sigmaX );
  RandGauss aGaussObjY( aDRand48Engine, 0., sigmaY );
  RandGauss aGaussObjZ( aDRand48Engine, 0., sigmaZ );

  GlobalVector moveV( aGaussObjX.fire(), aGaussObjY.fire(),
					  aGaussObjZ.fire() );

  alignable->move(moveV);
  m_modified++;

}


//__________________________________________________________________________________________________
/// Random flat move. The random number seed is taken from 'seed' if larger than zero.
/// Otherwise, a seed is generated from the RandomNumberGenerator service.
void AlignableTrackerModifier::randomFlatMove( Alignable* alignable, 
											   float sigmaX, float sigmaY, float sigmaZ, long seed )
{

  edm::LogInfo("PrintArgs") << "flat move randomly with sigma " 
							<< sigmaX << " " << sigmaY << " " << sigmaZ 
							<< std::endl;
  
  DRand48Engine aDRand48Engine;
  if ( seed > 0 )
	aDRand48Engine.setSeed(seed, 0);
  else
	{
	  edm::Service<edm::RandomNumberGenerator> rng;
	  aDRand48Engine.setSeed( rng->mySeed(), 0 );
	}
  
  RandFlat aFlatObjX( aDRand48Engine, sigmaX );
  RandFlat aFlatObjY( aDRand48Engine, sigmaY );
  RandFlat aFlatObjZ( aDRand48Engine, sigmaZ );

  GlobalVector moveV( aFlatObjX.fire(), aFlatObjY.fire(),
					  aFlatObjZ.fire() );
  alignable->move(moveV);
  m_modified++;

}


//__________________________________________________________________________________________________
/// Random gaussian rotation. The random number seed is taken from 'seed' if larger than zero.
/// Otherwise, a seed is generated from the RandomNumberGenerator service.
void AlignableTrackerModifier::randomRotate( Alignable* alignable, 
											 float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ, 
											 long seed )
{
  edm::LogInfo("PrintArgs") << "rotate randomly about GLOBAL x,y,z axis with sigmaPhi " 
							<< sigmaPhiX << " " << sigmaPhiY << " " << sigmaPhiZ 
							<< std::endl;
  
  DRand48Engine aDRand48Engine;
  if ( seed > 0 ) aDRand48Engine.setSeed(seed, 0);
  else
	{
	  edm::Service<edm::RandomNumberGenerator> rng;
	  aDRand48Engine.setSeed( rng->mySeed(), 0 );
	}

  RandGauss aGaussObjX( aDRand48Engine, 0., sigmaPhiX );
  RandGauss aGaussObjY( aDRand48Engine, 0., sigmaPhiY );
  RandGauss aGaussObjZ( aDRand48Engine, 0., sigmaPhiZ ); 
  
  alignable->rotateAroundGlobalX(aGaussObjX.fire());
  alignable->rotateAroundGlobalY(aGaussObjY.fire());
  alignable->rotateAroundGlobalZ(aGaussObjZ.fire());
  m_modified++;

}


//__________________________________________________________________________________________________
/// Random flat rotation. The random number seed is taken from 'seed' if larger than zero.
/// Otherwise, a seed is generated from the RandomNumberGenerator service.
void AlignableTrackerModifier::randomFlatRotate( Alignable* alignable, 
												 float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ, 
												 long seed )
{
  edm::LogInfo("PrintArgs") << "flat rotate randomly about GLOBAL x,y,z axis with sigmaPhi " 
							<< sigmaPhiX << " " << sigmaPhiY << " " << sigmaPhiZ 
							<< std::endl;
  
  DRand48Engine aDRand48Engine;
  if ( seed > 0 ) aDRand48Engine.setSeed(seed, 0);
  else
	{
	  edm::Service<edm::RandomNumberGenerator> rng;
	  aDRand48Engine.setSeed( rng->mySeed(), 0 );
	}

  RandFlat aFlatObjX( aDRand48Engine, 0., sigmaPhiX );
  RandFlat aFlatObjY( aDRand48Engine, 0., sigmaPhiY );
  RandFlat aFlatObjZ( aDRand48Engine, 0., sigmaPhiZ ); 
  
  alignable->rotateAroundGlobalX(aFlatObjX.fire());
  alignable->rotateAroundGlobalY(aFlatObjY.fire());
  alignable->rotateAroundGlobalZ(aFlatObjZ.fire());
  m_modified++;

}


//__________________________________________________________________________________________________
/// Here the rotation Axis is interpreted according to the local coordinate system of the Alignable.
/// First it is rotated around local_x, then the new local_y and then the new local_z.
/// The random number seed is taken from 'seed' if larger than zero.
/// Otherwise, a seed is generated from the RandomNumberGenerator service.
void AlignableTrackerModifier::randomRotateLocal( Alignable* alignable, 
												  float sigmaPhiX, float sigmaPhiY,float sigmaPhiZ, 
												  long seed )
{

  edm::LogInfo("PrintArgs") << "rotate randomly around LOCAL x,y,z axis with sigmaPhi " 
							<< sigmaPhiX << " " << sigmaPhiY << " " << sigmaPhiZ 
							<< std::endl;
  
  DRand48Engine aDRand48Engine;
  if ( seed > 0 ) aDRand48Engine.setSeed(seed, 0);
  else
	{
	  edm::Service<edm::RandomNumberGenerator> rng;
	  aDRand48Engine.setSeed( rng->mySeed(), 0 );
	}

  RandGauss aGaussObjX( aDRand48Engine, 0., sigmaPhiX );
  RandGauss aGaussObjY( aDRand48Engine, 0., sigmaPhiY );
  RandGauss aGaussObjZ( aDRand48Engine, 0., sigmaPhiZ );

  alignable->rotateAroundLocalX(aGaussObjX.fire());
  alignable->rotateAroundLocalY(aGaussObjY.fire());
  alignable->rotateAroundLocalZ(aGaussObjZ.fire());
  m_modified++;

}


//__________________________________________________________________________________________________
/// Here the rotation Axis is interpreted according to the local coordinate system of the Alignable.
/// First it is rotated around local_x, then the new local_y and then the new local_z.
/// The random number seed is taken from 'seed' if larger than zero.
/// Otherwise, a seed is generated from the RandomNumberGenerator service.
void AlignableTrackerModifier::randomFlatRotateLocal( Alignable* alignable, 
													  float sigmaPhiX, float sigmaPhiY, 
													  float sigmaPhiZ, long seed )
{

  edm::LogInfo("PrintArgs") << "flat rotate randomly around LOCAL x,y,z axis with sigmaPhi " 
							<< sigmaPhiX << " " << sigmaPhiY << " " << sigmaPhiZ 
							<< std::endl;
  
  DRand48Engine aDRand48Engine;
  if ( seed > 0 ) aDRand48Engine.setSeed(seed, 0);
  else
	{
	  edm::Service<edm::RandomNumberGenerator> rng;
	  aDRand48Engine.setSeed( rng->mySeed(), 0 );
	}


  RandFlat aFlatObjX( aDRand48Engine, sigmaPhiX );
  RandFlat aFlatObjY( aDRand48Engine, sigmaPhiY );
  RandFlat aFlatObjZ( aDRand48Engine, sigmaPhiZ );
  
  alignable->rotateAroundLocalX(aFlatObjX.fire());
  alignable->rotateAroundLocalY(aFlatObjY.fire());
  alignable->rotateAroundLocalZ(aFlatObjZ.fire());
  m_modified++;

}


//__________________________________________________________________________________________________
void AlignableTrackerModifier::addAlignmentPositionError( Alignable* alignable, 
														  float dx, float dy, float dz )
{

  edm::LogInfo("PrintArgs") << "Adding an AlignmentPositionError of size " 
							<< dx << " "  << dy << " "  << dz << std::endl;

  AlignmentPositionError ape(dx,dy,dz);
  alignable->addAlignmentPositionError( ape );

}



//__________________________________________________________________________________________________
void AlignableTrackerModifier::addAlignmentPositionErrorFromRotation( Alignable* alignable, 
																	  float phiX, float phiY, 
																	  float phiZ )
{

  RotationType rotx( Basic3DVector<float>(1.0, 0.0, 0.0), phiX );
  RotationType roty( Basic3DVector<float>(0.0, 1.0, 0.0), phiY );
  RotationType rotz( Basic3DVector<float>(0.0, 0.0, 1.0), phiZ );
  RotationType rot = rotz * roty * rotx;
  
  this->addAlignmentPositionErrorFromRotation( alignable, rot );

}


//__________________________________________________________________________________________________
void AlignableTrackerModifier::addAlignmentPositionErrorFromLocalRotation( Alignable* alignable, 
																		   float phiX, float phiY, 
																		   float phiZ )
{

  RotationType rotx( Basic3DVector<float>(1.0, 0.0, 0.0), phiX );
  RotationType roty( Basic3DVector<float>(0.0, 1.0, 0.0), phiY );
  RotationType rotz( Basic3DVector<float>(0.0, 0.0, 1.0), phiZ );
  RotationType rot = rotz * roty * rotx;
  
  this->addAlignmentPositionErrorFromLocalRotation( alignable, rot );

}


//__________________________________________________________________________________________________
void AlignableTrackerModifier::addAlignmentPositionErrorFromRotation( Alignable* alignable, 
																	  RotationType& rotation )
{ 

  edm::LogInfo("PrintArgs") << "Adding an AlignmentPositionError from Rotation" << std::endl 
							<< rotation << std::endl;

  alignable->addAlignmentPositionErrorFromRotation( rotation );

}


//__________________________________________________________________________________________________
void AlignableTrackerModifier::addAlignmentPositionErrorFromLocalRotation( Alignable* alignable, 
																		   RotationType& rotation )
{ 
  
  edm::LogInfo("PrintArgs") << "Adding an AlignmentPositionError from Local Rotation" << std::endl 
							<< rotation << std::endl;
  
  alignable->addAlignmentPositionErrorFromLocalRotation( rotation );
  
}

