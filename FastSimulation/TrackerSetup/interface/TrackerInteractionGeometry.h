#ifndef FastSimulation_TrackerSetup_TrackerInteractionGeometry_H
#define FastSimulation_TrackerSetup_TrackerInteractionGeometry_H
// v0  who ? when ? 
// 11 Dec 2003 Florian Beaudette. Removed the surfaces corresponding to ECAL 
//             This will carried out by the FamosTrajectoryManager
// 12 Oct 2006 Patrick Janot. Removed hardcoded active geometry & rings
//                            Removed RecHit smearing parameterization

//FAMOS Headers
#include "FastSimulation/TrackerSetup/interface/TrackerLayer.h"
#include "FastSimulation/TrackerSetup/interface/TrackerRing.h"

#include <list>
#include <map>

class MediumProperties;
class GeometricSearchTracker;

class TrackerInteractionGeometry
{

 public:

  TrackerInteractionGeometry(const GeometricSearchTracker* geomSearchTracker);

  ~TrackerInteractionGeometry();

  /// Returns the first pointer in the cylinder list
  inline std::list<TrackerLayer>::iterator cylinderBegin() 
    { return _theCylinders.begin(); }

  /// Returns the last pointer in the cylinder list
  inline std::list<TrackerLayer>::iterator cylinderEnd() 
    { return _theCylinders.end(); }

  /// Returns the number of cylinders in the Tracker
  inline const int nCylinders() const 
    { return static_cast<const int>(_theCylinders.size()); }

 private:

  /// The list of tracker (sensistive or not) layers
  std::list<TrackerLayer> _theCylinders;

  /// The following list gives the thicknesses of the various layers.
  /// The beam pipe
  MediumProperties *_theMPBeamPipe;
  /// The barrel pixel layers
  MediumProperties *_theMPPixelBarrel;
  /// The endcap pixel layers
  MediumProperties *_theMPPixelEndcap;
  /// A series of cables/walls to reproduce the full sim
  MediumProperties *_theMPPixelOutside1;
  MediumProperties *_theMPPixelOutside2;
  MediumProperties *_theMPPixelOutside3;
  MediumProperties *_theMPPixelOutside4;
  MediumProperties *_theMPPixelOutside;
  MediumProperties *_theMPPixelOutside5;
  MediumProperties *_theMPPixelOutside6;
  /// The tracker inner barrel layer 1
  MediumProperties *_theMPTIB1;
  /// The tracker inner barrel layer 2
  MediumProperties *_theMPTIB2;
  /// The tracker inner barrel layer 3
  MediumProperties *_theMPTIB3;
  /// The tracker inner barrel layer 4
  MediumProperties *_theMPTIB4;
  /// The tracker outer barrel layer 1
  MediumProperties *_theMPTOB1;
  /// The tracker outer barrel layer 2
  MediumProperties *_theMPTOB2;
  /// The tracker outer barrel layer 3
  MediumProperties *_theMPTOB3;
  /// The tracker outer barrel layer 4
  MediumProperties *_theMPTOB4;
  /// The tracker outer barrel layer 5
  MediumProperties *_theMPTOB5;
  /// The tracker outer barrel layer 6
  MediumProperties *_theMPTOB6;
  /// The Tracker EndCap layers
  MediumProperties *_theMPEndcap;
  /// The tracker inner disks
  MediumProperties *_theMPInner;
  /// Some material in front of the tracker outer barrel (cylinder) 
  MediumProperties *_theMPTOBBInside;
  /// Some material around the tracker inner barrel (disk) 
  MediumProperties *_theMPTIBEOutside;
  /// Some material around the tracker outer barrel (disk) 
  MediumProperties *_theMPTOBEOutside;
  /// Some material around the tracker inner disks (disk) 
  MediumProperties *_theMPTIDEOutside;
  /// Cables around the tracker (one barrel, two disks)
  MediumProperties *_theMPBarrelOutside;
  MediumProperties *_theMPEndcapOutside;
  MediumProperties *_theMPEndcapOutside2;

};
#endif
