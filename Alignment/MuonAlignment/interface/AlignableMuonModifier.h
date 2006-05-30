#ifndef Alignment_MuonAlignment_AlignableMuonModifier_H
#define Alignment_MuonAlignment_AlignableMuonModifier_H

#include <iostream>
#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"

/// AlignableMuonModifier is a helper class to modify the Alignables.
///
/// Configuration parameters are defined in this class.

class AlignableMuonModifier 
{

public:
  
  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;

  /// Constructor
  AlignableMuonModifier() {};

  /// Destructor
  ~AlignableMuonModifier() {};

  /// Modify given set of alignables according to parameters
  bool modify( Alignable* alignable, const edm::ParameterSet& pSet );

  /// Check if given parameter should be propagated
  const bool isPropagated( const std::string parameterName ) const;

  /// Move alignable in global space according to parameters
  void moveAlignable( Alignable* alignable, bool random, bool gaussian,
					  float sigmaX, float sigmaY, float sigmaZ,
					  long seed );

  /// Rotate alignable in global space according to parameters
  void rotateAlignable( Alignable* alignable, bool random, bool gaussian,
						float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ,
						long seed );

  /// Rotate alignable in local space according to parameters
  void rotateAlignableLocal( Alignable* alignable, bool random, bool gaussian,
							 float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ,
							 long seed );
  
  /// Random gaussian rotation of an Alignable in local frame
  void randomRotateLocal( Alignable* alignable, 
						  float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ,
						  long seed );
  
  /// Random flat rotation of an Alignable in local frame
  void randomFlatRotateLocal( Alignable* alignable, 
							  float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ,
							  long seed );

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

private:

  /// Initialisation of all parameters
  void init_(); 
  /// Decodes string and sets distribution accordingly
  void setDistribution_( std::string distr );
  /// Return a vector of random numbers (gaussian distribution)
  const GlobalVector gaussianRandomVector_( const float sigmaX, const float sigmaY, 
											 const float sigmaZ, long seed ) const;
  /// Return a vector of random numbers (flat distribution)
  const GlobalVector flatRandomVector_( const float sigmaX, const float sigmaY, 
										 const float sigmaZ, long seed ) const;

  int m_modified; // Indicates if a modification was performed

  // All parameters (see AlignableMuonModifier::init() for definitions)
  std::string distribution_;
  bool   random_, gaussian_, setError_;
  long   seed_;
  double scaleError_;
  double phiX_, phiY_, phiZ_;
  double localX_, localY_, localZ_;
  double dX_, dY_, dZ_;
  double twist_, shear_;

};


#endif //AlignableMuonModifier_H












