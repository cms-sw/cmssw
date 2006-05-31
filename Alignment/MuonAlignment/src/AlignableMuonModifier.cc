#include "CLHEP/Random/Random.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/Randomize.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/MuonAlignment/interface/AlignableMuonModifier.h"


//__________________________________________________________________________________________________
AlignableMuonModifier::AlignableMuonModifier( void )
{

  theDRand48Engine = new DRand48Engine();

}

//__________________________________________________________________________________________________
void AlignableMuonModifier::init_( void )
{

  // Initialize all known parameters (according to ORCA's MisalignmentScenario.cc)
  distribution_ = "";        // Switch for distributions ("fixed","flat","gaussian")
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

  // These are set through 'distribution'
  random_       = true;      // Use random distributions
  gaussian_     = true;      // Use gaussian distribution (otherwise flat)

}

//__________________________________________________________________________________________________
// Return true if given parameter name should be propagated down
const bool AlignableMuonModifier::isPropagated( const std::string parameterName ) const
{

  if ( parameterName == "distribution" || 
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
  this->setDistribution( distribution_ );

  // Apply displacements
  if ( fabs(dX_) + fabs(dY_) + fabs(dZ_) > 0 )
	this->moveAlignable( alignable, random_, gaussian_, dX_, dY_, dZ_ );

  // Apply rotations
  if ( fabs(phiX_) + fabs(phiY_) + fabs(phiZ_) > 0 )
	this->rotateAlignable( alignable, random_, gaussian_, phiX_, phiY_, phiZ_ );

  // Apply local rotations
  if ( fabs(localX_) + fabs(localY_) + fabs(localZ_) > 0 )
	this->rotateAlignableLocal( alignable, random_, gaussian_, localX_, localY_, localZ_ );

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
void AlignableMuonModifier::setDistribution( std::string distr )
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
/// If 'seed' is zero, asks  RandomNumberGenerator service.
void AlignableMuonModifier::setSeed( const long seed )
{

  long m_seed;

  if ( seed > 0 ) m_seed = seed;
  else
	{
	  edm::Service<edm::RandomNumberGenerator> rng;
	  m_seed = rng->mySeed();
	}

  edm::LogInfo("PrintArgs") << "Setting generator seed to " << m_seed;

  theDRand48Engine->setSeed( m_seed );

}

//__________________________________________________________________________________________________
/// If 'random' is false, the given movements are strictly applied. Otherwise, a random
/// number is generated according to a gaussian or a flat distribution depending on 'gaussian'.
void AlignableMuonModifier::moveAlignable( Alignable* alignable, bool random, bool gaussian,
											  float sigmaX, float sigmaY, float sigmaZ )
{

  
  std::ostringstream message;
 
  // Get movement vector according to arguments
  GlobalVector moveV( sigmaX, sigmaY, sigmaZ ); // Default: fixed
  if ( random ) 
	{
	  message << "random ";
	  if (gaussian)
		{
		  moveV = this->gaussianRandomVector_( sigmaX, sigmaY, sigmaZ );
		  message << "gaussian ";
		}
	  else 
		{
		  moveV = flatRandomVector_( sigmaX, sigmaY, sigmaZ );
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
void AlignableMuonModifier::rotateAlignable( Alignable* alignable, bool random, bool gaussian,
												float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ )
{

  
  std::ostringstream message;

  // Get rotation vector according to arguments
  GlobalVector rotV( sigmaPhiX, sigmaPhiY, sigmaPhiZ ); // Default: fixed
  if ( random ) 
	{
	  message << "random ";
	  if (gaussian)
		{
		  rotV = this->gaussianRandomVector_( sigmaPhiX, sigmaPhiY, sigmaPhiZ );
		  message << "gaussian ";
		}
	  else 
		{
		  rotV = flatRandomVector_( sigmaPhiX, sigmaPhiY, sigmaPhiZ );
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
void 
AlignableMuonModifier::rotateAlignableLocal( Alignable* alignable, bool random, bool gaussian,
												float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ )
{

  
  std::ostringstream message;

  // Get rotation vector according to arguments
  GlobalVector rotV( sigmaPhiX, sigmaPhiY, sigmaPhiZ ); // Default: fixed
  if ( random ) 
	{
	  message << "random ";
	  if (gaussian)
		{
		  rotV = this->gaussianRandomVector_( sigmaPhiX, sigmaPhiY, sigmaPhiZ );
		  message << "gaussian ";
		}
	  else 
		{
		  rotV = flatRandomVector_( sigmaPhiX, sigmaPhiY, sigmaPhiZ );
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
AlignableMuonModifier::gaussianRandomVector_( float sigmaX, float sigmaY, float sigmaZ ) const
{

  // Pass by reference, otherwise pointer is deleted!
  RandGauss aGaussObjX( *theDRand48Engine, 0., sigmaX );
  RandGauss aGaussObjY( *theDRand48Engine, 0., sigmaY );
  RandGauss aGaussObjZ( *theDRand48Engine, 0., sigmaZ );

  return GlobalVector( aGaussObjX.fire(), aGaussObjY.fire(), aGaussObjZ.fire() );

}


//__________________________________________________________________________________________________
const GlobalVector 
AlignableMuonModifier::flatRandomVector_( const float sigmaX, const float sigmaY, 
											 const float sigmaZ ) const
{

  RandFlat aFlatObjX( *theDRand48Engine, -sigmaX, sigmaX );
  RandFlat aFlatObjY( *theDRand48Engine, -sigmaY, sigmaY );
  RandFlat aFlatObjZ( *theDRand48Engine, -sigmaZ, sigmaZ );

  return GlobalVector( aFlatObjX.fire(), aFlatObjY.fire(), aFlatObjZ.fire() );

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

