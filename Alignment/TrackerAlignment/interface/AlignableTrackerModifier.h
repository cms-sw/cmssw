#ifndef Alignment_TrackerAlignment_AlignableTrackerModifier_H
#define Alignment_TrackerAlignment_AlignableTrackerModifier_H

#include <iostream>
#include <vector>
#include <string>

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
  AlignableTrackerModifier() {};

  /// Destructor
  ~AlignableTrackerModifier() {};

  /// Modify given set of alignables according to parameters
  bool modify( Alignable* alignable, const edm::ParameterSet& pSet );

  /// Check if given parameter should be propagated
  const bool isPropagated( const std::string parameterName ) const;
  
  /// Random gaussian move in global space of a collection of Alignables
  void randomMove( Alignable* alignable, 
				   float sigmaX, float sigmaY, float sigmaZ,
				   long seed );
  
  /// Random flat move in global space of a collection of Alignables
  void randomFlatMove( Alignable* alignable, 
					   float sigmaX, float sigmaY, float sigmaZ,
					   long seed );
  
  /// Random gaussian rotation of a collection of Alignables
  void randomRotate( Alignable* alignable, 
					 float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ,
					 long seed );
  
  /// Random gaussian rotation of a collection of Alignables in local frame
  void randomRotateLocal( Alignable* alignable, 
						  float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ,
						  long seed );
  
  /// Random flat rotation of a collection of Alignables
  void randomFlatRotate( Alignable* alignable, 
						 float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ,
						 long seed );
  
  /// Random flat rotation of a collection of Alignables in local frame
  void randomFlatRotateLocal( Alignable* alignable, 
							  float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ,
							  long seed );

  /// Add the AlignmentPositionError (in global frame) to all elements of vector
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

  int m_modified; // Indicates if a modification was performed

};


#endif //AlignableTrackerModifier_H












