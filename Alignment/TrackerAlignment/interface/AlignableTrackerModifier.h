#ifndef Alignment_TrackerAlignment_AlignableTrackerModifier_H
#define Alignment_TrackerAlignment_AlignableTrackerModifier_H

#include <iostream>
#include <vector>

#include "Alignment/CommonAlignment/interface/Alignable.h"

/// AlignableTrackerModifier is a helper class to modify the Alignables

class AlignableTrackerModifier 
{

public:
  
  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;

  /// Constructor
  AlignableTrackerModifier();

  /// Destructor
  ~AlignableTrackerModifier();
  
  /// Random gaussian move in global space of a collection of Alignables
  void randomMove( std::vector<Alignable*> comp, 
				   float sigmaX, float sigmaY, float sigmaZ,
				   bool setSeed, long seed );

  /// Random flat move in global space of a collection of Alignables
  void randomFlatMove( std::vector<Alignable*> comp, 
					   float sigmaX, float sigmaY, float sigmaZ,
					   bool setSeed, long seed );

  /// Random gaussian movement of all components of a collection of Alignables in local frame
  void randomMoveComponentsLocal( std::vector<Alignable*> comp, 
								  float sigmaX, float sigmaY, float sigmaZ,
								  bool setSeed, long seed );

  /// Random gaussian rotation of a collection of Alignables
  void randomRotate( std::vector<Alignable*> comp, 
					 float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ,
					 bool setSeed, long seed );

  /// Random gaussian rotation of a collection of Alignables in local frame
  void randomRotateLocal( std::vector<Alignable*> comp, 
						  float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ,
						  bool setSeed, long seed );

  /// Random flat rotation of a collection of Alignables in local frame
  void randomFlatRotateLocal( std::vector<Alignable*> comp, 
							  float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ,
							  bool setSeed, long seed );

  /// Random gaussian rotation of all components of a collection of Alignables
  void randomRotateComponentsLocal( std::vector<Alignable*> comp, 
									float sigmaPhiX, float sigmaPhiY, float sigmaPhiZ,
									bool setSeed, long seed );

  /// Add the AlignmentPositionError (in global frame) to all elements of vector
  void addAlignmentPositionError( std::vector<Alignable*> comp, 
								  float dx, float dy, float dz );

  /// Add the AlignmentPositionError (in local frame) to all elements of vector
  void addAlignmentPositionErrorLocal( std::vector<Alignable*> comp, 
									   float dx, float dy, float dz );
  
  /// Add alignment position error resulting from rotation in global frame
  void addAlignmentPositionErrorFromRotation( std::vector<Alignable*> comp, 
											  RotationType& rotation ); 

  /// Add alignment position error resulting from rotation in local frame
  void addAlignmentPositionErrorFromLocalRotation( std::vector<Alignable*> comp, 
												   RotationType& rotation ); 

};


#endif //AlignableTrackerModifier_H












