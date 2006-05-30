#include "CLHEP/Random/Random.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/Randomize.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/MuonAlignment/interface/AlignableMuonModifier.h"


//__________________________________________________________________________________________________
void AlignableMuonModifier::init_( void )
{

  // Initialize all known parameters (according to ORCA's MisalignmentScenario.cc)
  distribution_ = "";   // Switch for distributions ("fixed","flat","gaussian")
  random_       = true;      // Use random distributions
  gaussian_     = true;      // Use gaussian distribution (otherwise flat)
  seed_         = 0;         // Random generator seed (default: ask service)
  setError_     = false;     // Apply alignment errors
  scaleError_   = 1.;        // Scale to apply to alignment errors
  phiX_         = 0.;        // Rotation angle around X [rad]
  phiY_         = 0.;        // Rotation angle around Y [rad]
  phiZ_         = 0.;        // Rotation angle around Z [rad]
  localX_       = 0.;        // Local rotation angle around X [rad]
  localY_       = 0.;        // Local rotation angle around Y [rad]
  localZ_       = 0.;        // Local rotation angle around Z [rad]
  dX_           = 0.;        // X displacement [cm]
  dY_           = 0.;        // Y displacement [cm]
  dZ_           = 0.;        // Z displacement [cm]
  twist_        = 0.;        // Twist angle [rad]
  shear_        = 0.;        // Shear angle [rad]

}

//__________________________________________________________________________________________________
// Return true if given parameter name should be propagated down
const bool AlignableMuonModifier::isPropagated( const std::string parameterName ) const
{

  if ( parameterName == "distribution" || 
	   parameterName == "seed"         ||
	   parameterName == "setError"     ||
	   parameterName == "scaleError"   ) return true;
  
  return false;

}


//__________________________________________________________________________________________________
/// All known parameters and defaults are defined here! Returns true if modification actually applied.
bool AlignableMuonModifier::modify( Alignable* alignable, const edm::ParameterSet& pSet )
{

  // Initialize parameters
  this->init_();

  // Reset counter
  m_modified = 0;
  
  // Retrieve parameters
  std::ostringstream error;
  std::vector<std::string> parameterNames = pSet.getParameterNames();
  for ( std::vector<std::string>::iterator iParam = parameterNames.begin(); 
		iParam != parameterNames.end(); iParam++ )
	{
	  if  ( (*iParam) == "distribution" ) distribution_ = pSet.getParameter<std::string>( *iParam );
	  else if ( (*iParam) == "setError" ) setError_ = pSet.getParameter<bool>( *iParam );
	  else if ( (*iParam) == "scaleError" ) scaleError_ = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "phiX" )     phiX_     = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "phiY" )     phiY_     = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "phiZ" )     phiZ_     = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "localX" )   localX_   = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "localY" )   localY_   = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "localZ" )   localZ_   = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "dX" )       dX_       = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "dY" )       dY_       = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "dZ" )       dZ_       = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "twist" )    twist_    = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "shear" )    shear_    = pSet.getParameter<double>( *iParam );
	  else if ( (*iParam) == "seed"  )    seed_     = static_cast<long>(pSet.getParameter<int>( *iParam ));
	  else if ( pSet.retrieve( *iParam ).typeCode() != 'P' )
		{ // Add unknown parameter to list
		  if ( !error.str().length() ) error << "Unknown parameter name(s): ";
		  error << " " << *iParam;
		}
	}

  // Check error
  if ( error.str().length() )
	throw cms::Exception("BadConfig") << error.str();

  // Decode distribution
  this->setDistribution_( distribution_ );

  // Apply displacements
  if ( fabs(dX_) + fabs(dY_) + fabs(dZ_) > 0 )
	this->moveAlignable( alignable, random_, gaussian_, dX_, dY_, dZ_, seed_ );

  // Apply rotations
  if ( fabs(phiX_) + fabs(phiY_) + fabs(phiZ_) > 0 )
	this->rotateAlignable( alignable, random_, gaussian_, phiX_, phiY_, phiZ_, seed_ );

  // Apply local rotations
  if ( fabs(localX_) + fabs(localY_) + fabs(localZ_) > 0 )
	if ( gaussian_ ) this->randomRotateLocal( alignable, localX_, localY_, localZ_, seed_ );
	else this->randomFlatRotateLocal( alignable, localX_, localY_, localZ_, seed_ );


  // Apply twist
  if ( fabs(twist_) > 0 )
	edm::LogError("NotImplemented") << "Twist is not implemented yet";

  // Apply shear
  if ( fabs(shear_) > 0 )
	edm::LogError("NotImplemented") << "Shear is not implemented yet";

  // Apply error
  if ( setError_ )
	{
	  // Alignment Position Error for flat distribution: 1 sigma
	  if ( !gaussian_ ) scaleError_ *= 0.68;

	  // Error on displacement
	  if ( fabs(dX_) + fabs(dY_) + fabs(dZ_) > 0 )
		this->addAlignmentPositionError( alignable, 
										 scaleError_*dX_, scaleError_*dY_, scaleError_*dZ_ );

	  // Error on rotations
	  if ( fabs(phiX_) + fabs(phiY_) + fabs(phiZ_) > 0 )
		this->addAlignmentPositionErrorFromRotation( alignable, 
													 scaleError_*phiX_, scaleError_*phiY_, 
													 scaleError_*phiZ_ );

	  // Error on local rotations
	  if ( fabs(localX_) + fabs(localY_) + fabs(localZ_) > 0 )
		this->addAlignmentPositionErrorFromLocalRotation( alignable, 
														  scaleError_*localX_, scaleError_*localY_, 
														  scaleError_*localZ_ );
	}

  return ( m_modified > 0 );
  
}


//__________________________________________________________________________________________________
void AlignableMuonModifier::setDistribution_( std::string distr )
{

  if ( distr == "fixed" ) random_ = false;
  else if ( distr == "flat" ) 
	{
	  random_   = true;
	  gaussian_ = false;
	}
  else if ( distr == "gaussian" )
	{
	  random_   = true;
	  gaussian_ = true;
	}

}


//__________________________________________________________________________________________________
/// If 'random' is false, the given movements are strictly applied. Otherwise, a random
/// number is generated according to a gaussian or a flat distribution depending on 'gaussian'.
/// The random number seed is taken from 'seed' if larger than zero.
/// Otherwise, a seed is generated from the RandomNumberGenerator service.
void AlignableMuonModifier::moveAlignable( Alignable* alignable, bool random, bool gaussian,
											  float sigmaX, float sigmaY, float sigmaZ,
											  long seed )
{

  
  std::ostringstream message;

  // Get seed if not given
  if ( seed == 0 && random )
	{
	  edm::Service<edm::RandomNumberGenerator> rng;
	  seed = rng->mySeed();
	}
  
  // Get movement vector according to arguments
  GlobalVector moveV( sigmaX, sigmaY, sigmaZ ); // Default: fixed
  if ( random ) 
	{
	  message << "random ";
	  if (gaussian)
		{
		  moveV = this->gaussianRandomVector_( sigmaX, sigmaY, sigmaZ, seed );
		  message << "gaussian ";
		}
	  else 
		{
		  moveV = flatRandomVector_( sigmaX, sigmaY, sigmaZ, seed );
		  message << "flat ";
		}
	}
  
  message << " move with sigma " << sigmaX << " " << sigmaY << " " << sigmaZ;

  edm::LogInfo("PrintArgs") << message; // Arguments

  edm::LogInfo("PrintMovement") << "applied displacement: " << moveV; // Actual movements
  alignable->move(moveV);
  m_modified++;


}


//__________________________________________________________________________________________________
/// If 'random' is false, the given rotations are strictly applied. Otherwise, a random
/// number is generated according to a gaussian or a flat distribution depending on 'gaussian'.
/// The random number seed is taken from 'seed' if larger than zero.
/// Otherwise, a seed is generated from the RandomNumberGenerator service.
void AlignableMuonModifier::rotateAlignable( Alignable* alignable, bool random, bool gaussian,
												float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ,
												long seed )
{

  
  std::ostringstream message;

  // Get seed if not given
  if ( seed == 0 && random )
	{
	  edm::Service<edm::RandomNumberGenerator> rng;
	  seed = rng->mySeed();
	}
  
  // Get rotation vector according to arguments
  GlobalVector rotV( sigmaPhiX, sigmaPhiY, sigmaPhiZ ); // Default: fixed
  if ( random ) 
	{
	  message << "random ";
	  if (gaussian)
		{
		  rotV = this->gaussianRandomVector_( sigmaPhiX, sigmaPhiY, sigmaPhiZ, seed );
		  message << "gaussian ";
		}
	  else 
		{
		  rotV = flatRandomVector_( sigmaPhiX, sigmaPhiY, sigmaPhiZ, seed );
		  message << "flat ";
		}
	}
  
  message << " global rotation by angles " << sigmaPhiX << " " << sigmaPhiY << " " << sigmaPhiZ;

  edm::LogInfo("PrintArgs") << message; // Arguments

  edm::LogInfo("PrintMovement") << "applied rotation angles: " << rotV; // Actual movements
  alignable->rotateAroundGlobalX( rotV.x() );
  alignable->rotateAroundGlobalY( rotV.y() );
  alignable->rotateAroundGlobalZ( rotV.z() );
  m_modified++;


}

//__________________________________________________________________________________________________
/// If 'random' is false, the given rotations are strictly applied. Otherwise, a random
/// number is generated according to a gaussian or a flat distribution depending on 'gaussian'.
/// The random number seed is taken from 'seed' if larger than zero.
/// Otherwise, a seed is generated from the RandomNumberGenerator service.
void 
AlignableMuonModifier::rotateAlignableLocal( Alignable* alignable, bool random, bool gaussian,
												float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ,
												long seed )
{

  
  std::ostringstream message;

  // Get seed if not given
  if ( seed == 0 && random )
	{
	  edm::Service<edm::RandomNumberGenerator> rng;
	  seed = rng->mySeed();
	}
  
  // Get rotation vector according to arguments
  GlobalVector rotV( sigmaPhiX, sigmaPhiY, sigmaPhiZ ); // Default: fixed
  if ( random ) 
	{
	  message << "random ";
	  if (gaussian)
		{
		  rotV = this->gaussianRandomVector_( sigmaPhiX, sigmaPhiY, sigmaPhiZ, seed );
		  message << "gaussian ";
		}
	  else 
		{
		  rotV = flatRandomVector_( sigmaPhiX, sigmaPhiY, sigmaPhiZ, seed );
		  message << "flat ";
		}
	}
  
  message << " local rotation by angles " << sigmaPhiX << " " << sigmaPhiY << " " << sigmaPhiZ;

  edm::LogInfo("PrintArgs") << message; // Arguments

  edm::LogInfo("PrintMovement") << "applied rotation angles: " << rotV; // Actual movements
  alignable->rotateAroundLocalX( rotV.x() );
  alignable->rotateAroundLocalY( rotV.y() );
  alignable->rotateAroundLocalZ( rotV.z() );
  m_modified++;


}


//__________________________________________________________________________________________________
const GlobalVector 
AlignableMuonModifier::gaussianRandomVector_( float sigmaX, float sigmaY,
												 float sigmaZ, long seed ) const
{

  DRand48Engine aDRand48Engine(seed);

  RandGauss aGaussObjX( aDRand48Engine, 0., sigmaX );
  RandGauss aGaussObjY( aDRand48Engine, 0., sigmaY );
  RandGauss aGaussObjZ( aDRand48Engine, 0., sigmaZ );

  return GlobalVector( aGaussObjX.fire(), aGaussObjY.fire(), aGaussObjZ.fire() );

}


//__________________________________________________________________________________________________
const GlobalVector 
AlignableMuonModifier::flatRandomVector_( const float sigmaX, const float sigmaY, 
											 const float sigmaZ, long seed ) const
{

  DRand48Engine aDRand48Engine(seed);

  RandFlat aFlatObjX( aDRand48Engine, sigmaX );
  RandFlat aFlatObjY( aDRand48Engine, sigmaY );
  RandFlat aFlatObjZ( aDRand48Engine, sigmaZ );

  return GlobalVector( aFlatObjX.fire(), aFlatObjY.fire(), aFlatObjZ.fire() );

}



//__________________________________________________________________________________________________
/// Here the rotation Axis is interpreted according to the local coordinate system of the Alignable.
/// First it is rotated around local_x, then the new local_y and then the new local_z.
/// The random number seed is taken from 'seed' if larger than zero.
/// Otherwise, a seed is generated from the RandomNumberGenerator service.
void AlignableMuonModifier::randomRotateLocal( Alignable* alignable, 
												  float sigmaPhiX, float sigmaPhiY,float sigmaPhiZ, 
												  long seed )
{

  edm::LogInfo("PrintArgs") << "rotate randomly around LOCAL x,y,z axis with sigmaPhi " 
							<< sigmaPhiX << " " << sigmaPhiY << " " << sigmaPhiZ;
  
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

  double phiX = aGaussObjX.fire();
  double phiY = aGaussObjY.fire();
  double phiZ = aGaussObjZ.fire();

  edm::LogInfo("PrintMovement") << "applied angles: " << phiX << "," << phiY << "," << phiZ;

  alignable->rotateAroundLocalX( phiX );
  alignable->rotateAroundLocalY( phiY );
  alignable->rotateAroundLocalZ( phiZ );
  m_modified++;

}


//__________________________________________________________________________________________________
/// Here the rotation Axis is interpreted according to the local coordinate system of the Alignable.
/// First it is rotated around local_x, then the new local_y and then the new local_z.
/// The random number seed is taken from 'seed' if larger than zero.
/// Otherwise, a seed is generated from the RandomNumberGenerator service.
void AlignableMuonModifier::randomFlatRotateLocal( Alignable* alignable, 
													  float sigmaPhiX, float sigmaPhiY, 
													  float sigmaPhiZ, long seed )
{

  edm::LogInfo("PrintArgs") << "flat rotate randomly around LOCAL x,y,z axis with sigmaPhi " 
							<< sigmaPhiX << " " << sigmaPhiY << " " << sigmaPhiZ;
  
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
  
  double phiX = aFlatObjX.fire();
  double phiY = aFlatObjY.fire();
  double phiZ = aFlatObjZ.fire();

  alignable->rotateAroundLocalX( phiX );
  alignable->rotateAroundLocalY( phiY );
  alignable->rotateAroundLocalZ( phiZ );
  m_modified++;

}


//__________________________________________________________________________________________________
void AlignableMuonModifier::addAlignmentPositionError( Alignable* alignable, 
														  float dx, float dy, float dz )
{

  edm::LogInfo("PrintArgs") << "Adding an AlignmentPositionError of size " 
							<< dx << " "  << dy << " "  << dz;

  AlignmentPositionError ape(dx,dy,dz);
  alignable->addAlignmentPositionError( ape );

}



//__________________________________________________________________________________________________
void AlignableMuonModifier::addAlignmentPositionErrorFromRotation( Alignable* alignable, 
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
void AlignableMuonModifier::addAlignmentPositionErrorFromLocalRotation( Alignable* alignable, 
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
void AlignableMuonModifier::addAlignmentPositionErrorFromRotation( Alignable* alignable, 
																	  RotationType& rotation )
{ 

  edm::LogInfo("PrintArgs") << "Adding an AlignmentPositionError from Rotation" << std::endl 
							<< rotation;

  alignable->addAlignmentPositionErrorFromRotation( rotation );

}


//__________________________________________________________________________________________________
void AlignableMuonModifier::addAlignmentPositionErrorFromLocalRotation( Alignable* alignable, 
																		   RotationType& rotation )
{ 
  
  edm::LogInfo("PrintArgs") << "Adding an AlignmentPositionError from Local Rotation" << std::endl 
							<< rotation;
  
  alignable->addAlignmentPositionErrorFromLocalRotation( rotation );
  
}

