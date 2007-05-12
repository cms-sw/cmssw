#ifndef Alignment_TrackerAlignment_AlignableModifier_H
#define Alignment_TrackerAlignment_AlignableModifier_H

#include <iostream>
#include <vector>
#include <string>

#include "CLHEP/Random/DRand48Engine.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"

/// AlignableModifier is a helper class to modify the Alignables.
///
/// Configuration parameters are defined in this class.

class AlignableModifier 
{

public:
  
  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;

  /// Constructor
  AlignableModifier();

  /// Destructor
  ~AlignableModifier();

  /// Modify given set of alignables according to parameters
  bool modify( Alignable* alignable, const edm::ParameterSet& pSet );

  /// Check if given parameter should be propagated
  const bool isPropagated( const std::string parameterName ) const;

  /// Move alignable in global space according to parameters
  void moveAlignable( Alignable* alignable, bool random, bool gaussian,
					  float sigmaX, float sigmaY, float sigmaZ );

  /// Move alignable in local space according to parameters
  void moveAlignableLocal( Alignable* alignable, bool random, bool gaussian,
 						   float sigmaX, float sigmaY, float sigmaZ );

  /// Rotate alignable in global space according to parameters
  void rotateAlignable( Alignable* alignable, bool random, bool gaussian,
						float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ );

  /// Rotate alignable in local space according to parameters
  void rotateAlignableLocal( Alignable* alignable, bool random, bool gaussian,
							 float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ );
  
  /// Add the AlignmentPositionError (in global frame) to Alignable
  void addAlignmentPositionError( Alignable* alignable, 
								  float dx, float dy, float dz );

  /// Add the AlignmentPositionError (in local frame) to Alignable
  void addAlignmentPositionErrorLocal( Alignable* alignable, 
									   float dx, float dy, float dz );

  /// Add alignment position error resulting from rotation in global frame
  void addAlignmentPositionErrorFromRotation( Alignable* alignable, 
											  float phiX, float phiY, float phiZ ); 

  /// Add alignment position error resulting from rotation in local frame
  void addAlignmentPositionErrorFromLocalRotation( Alignable* alignable, 
												   float phiX, float phiY, float phiZ ); 

  /// Add alignment position error resulting from rotation in global frame
  void addAlignmentPositionErrorFromRotation( Alignable* alignable, 
											  RotationType& rotation ); 

  /// Add alignment position error resulting from rotation in local frame
  void addAlignmentPositionErrorFromLocalRotation( Alignable* alignable, 
												   RotationType& rotation ); 

  /// Decodes string and sets distribution accordingly ('fixed', 'flat' or 'gaussian').
  void setDistribution( std::string distr );

  /// Resets the generator seed according to the argument.
  void setSeed( long seed );

private:

  /// Unique random number generator
  DRand48Engine* theDRand48Engine;

  /// Initialisation of all parameters
  void init_(); 

  /// Return a vector of random numbers (gaussian distribution)
  const std::vector<float> gaussianRandomVector_( float sigmaX, float sigmaY, float sigmaZ ) const;
  /// Return a vector of random numbers (flat distribution)
  const std::vector<float> flatRandomVector_( float sigmaX, float sigmaY, float sigmaZ ) const;

  int m_modified; // Indicates if a modification was performed

  // All parameters (see AlignableModifier::init() for definitions)
  std::string distribution_;
  bool   random_, gaussian_, setError_;
  bool   setRotations_,setTranslations_;
  long   seed_;
  double scaleError_,scale_;
  double phiX_, phiY_, phiZ_;
  double phiXlocal_, phiYlocal_, phiZlocal_;
  double dX_, dY_, dZ_;
  double dXlocal_, dYlocal_, dZlocal_;
  double twist_, shear_;

};


#endif //AlignableModifier_H












