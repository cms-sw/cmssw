#ifndef Alignment_TrackerAlignment_AlignableTrackerModifier_H
#define Alignment_TrackerAlignment_AlignableTrackerModifier_H

#include <iostream>
#include <vector>
#include <string>

#include "CLHEP/Random/DRand48Engine.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"

/// AlignableTrackerModifier is a helper class to modify the Alignables.
///
/// Configuration parameters are defined in this class.

class AlignableTrackerModifier 
{

public:
  
  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;

  /// Constructor
  AlignableTrackerModifier();

  /// Destructor
  ~AlignableTrackerModifier() {};

  /// Modify given set of alignables according to parameters
  bool modify( Alignable* alignable, const edm::ParameterSet& pSet );

  /// Check if given parameter should be propagated
  const bool isPropagated( const std::string parameterName ) const;

  /// Move alignable in global space according to parameters
  void moveAlignable( Alignable* alignable, bool random, bool gaussian,
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
  const GlobalVector gaussianRandomVector_( float sigmaX, float sigmaY, float sigmaZ ) const;
  /// Return a vector of random numbers (flat distribution)
  const GlobalVector flatRandomVector_( float sigmaX, float sigmaY, float sigmaZ ) const;

  int m_modified; // Indicates if a modification was performed

  // All parameters (see AlignableTrackerModifier::init() for definitions)
  std::string distribution_;
  bool   random_, gaussian_, setError_;
  long   seed_;
  double scaleError_;
  double phiX_, phiY_, phiZ_;
  double localX_, localY_, localZ_;
  double dX_, dY_, dZ_;
  double twist_, shear_;

};


#endif //AlignableTrackerModifier_H












