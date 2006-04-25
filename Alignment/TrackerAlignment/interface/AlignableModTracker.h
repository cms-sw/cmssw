#ifndef AlignableModTracker_H
#define AlignableModTracker_H

//#include "TrackerReco/TkAlignment/interface/AlignableDoIt.h"
#include "TrackerReco/TkAlignment/interface/AlignableRod.h"
#include "TrackerReco/TkAlignment/interface/AlignableBarrelLayer.h"
#include "TrackerReco/TkAlignment/interface/AlignableHalfBarrel.h"
#include "TrackerReco/TkAlignment/interface/AlignablePxHalfBarrel.h"
//#include "TrackerReco/TkAlignment/interface/AlignablePxHalfBarrelLayer.h"
#include "TrackerReco/TkAlignment/interface/AlignablePetal.h"
#include "TrackerReco/TkAlignment/interface/AlignableEndcapLayer.h"
#include "TrackerReco/TkAlignment/interface/AlignableEndcap.h"
#include "TrackerReco/TkAlignment/interface/AlignableTracker.h"
#include "TrackerReco/TkAlignment/interface/AlignableTID.h"
#include "TrackerReco/TkAlignment/interface/AlignableTIDLayer.h"
#include "TrackerReco/TkAlignment/interface/AlignableTIDRing.h"

#include "CommonDet/DetLayout/interface/DetLayer.h"
#include "CommonDet/BasicDet/interface/DetType.h"
#include "CommonDet/BasicDet/interface/DetUnit.h"

#include "CommonDet/BasicDet/interface/Readout.h"
#include "Utilities/Notification/interface/TimingReport.h"


#include "Utilities/UI/interface/SimpleConfigurable.h"


#include <sstream>

/* AlignableModTracker is a class which should help you modifying the Alignable
 * Tracker.
 */

class AlignableModTracker {

public:
  
  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;

  AlignableModTracker();
  ~AlignableModTracker(){
    cout <<"==> finishing the Tracker Components Modification <==" <<endl;
  }
  
  /** random move in global space of a particular collecion of Alignables
   * e.g. Dets, Rods, Petals, Barrel or whatever
   */
  void randomMove(vector<Alignable*> comp, float sigmaX,
		  float sigmaY, float sigmaZ,
		  bool setSeed, long seed);

  /* random move in global space according to a flat distribution */
  void randomFlatMove(vector<Alignable*> comp, float sigmaX,
		  float sigmaY, float sigmaZ,
		  bool setSeed, long seed);

  /** random movement of all the COMPONENTS of the particular collection of
   *  Alignables (which should better be AlignableComposites) withing the
   *  host sturcture. X,Y,Z axis are interpreted as local coordinates in 
   *  the Composite. WATCHOUT. If the vector you supply here is a vector of
   *  rod/wedges, the DETs on them will be moved "locally"
   */
  void randomMoveComponentsLocal(vector<Alignable*> comp, 
				 float sigmaX,
				 float sigmaY, 
				 float sigmaZ,
				 bool setSeed,
				 long seed);

  /** random rotation of a particular collecion of Alignables
   * e.g. Dets, Rods, Petals, Barrel or whatever Here the rotation Axis is 
   *  interpreted according to the global coordinate system
   */
  void randomRotate(vector<Alignable*> comp, 
		    float sigmaPhiX,
		    float sigmaPhiY, float sigmaPhiZ,
		    bool setSeed, long seed);

  /** random rotation of a particular collecion of Alignables
   * e.g. Dets, Rods, Petals, Barrel or whatever Here the rotation Axis is 
   *  interpreted according to the local coordinate system of the Alignable*.
   * First it is rotated around local_x, then the new local_y and then the 
   * new local_z
   */
  void randomRotateLocal(vector<Alignable*> comp, 
			 float sigmaPhiX,
			 float sigmaPhiY, float sigmaPhiZ,
			 bool setSeed, long seed);

  /** random rotation using a flat distribution **/
  void randomFlatRotateLocal(vector<Alignable*> comp, 
			     float sigmaPhiX,
			     float sigmaPhiY, float sigmaPhiZ,
			     bool setSeed, long seed);

  /** random rotation of all the COMPONENTS of the particular collection of
   *  Alignables (which should better be AlignableComposites) withing the
   *  host sturcture. Here the rotation Axis is interpreted according to the 
   *  local  coordinate system of the COMPONENTS.
   */
  void randomRotateComponentsLocal(vector<Alignable*> 
				   comp, 
				   float sigmaPhiX,
				   float sigmaPhiY, 
				   float sigmaPhiZ,
				   bool setSeed, long seed);

  /** add the AlignmentPositionError to all elements in vector<Alignable*>
   *  where the dx,dy,dz are interpreted as global coordinates
   */
  void addAlignmentPositionError( vector<Alignable*> comp, 
				  float dx, float dy, 
				  float dz);

  /** add the AlignmentPositionError to all elements in vector<Alignable*>
   *  where the dx,dy,dz are interpreted as local coordinates coordinates
   *  of the Alignable* which are handed over
   */
  void addAlignmentPositionErrorLocal( vector<Alignable*> comp, 
				       float dx, float dy, 
				       float dz);
  
  void addAlignmentPositionErrorFromRotation(vector<Alignable*> comp, 
					     RotationType& rotation); 

  void addAlignmentPositionErrorFromLocalRotation(vector<Alignable*> comp, 
						  RotationType& rotation); 
  /*
  void randomRotate(vector<Alignable*> comp, float sigmaX,
					 float sigmaY, float sigmaZ);

  void randomRotateComponentsLocal(vector<Alignable*> comp,
 							float sigmaX,
							float sigmaY, 
							float sigmaZ);
  */
  
  void printTrackerSummary(AlignableTracker* theAlignableTracker);
  void printOutOfTrackerModule(vector<Alignable*>);
 private:
  
  AlignableTracker* theAlignableTracker;

};


#endif //AlignableModTracker_H












