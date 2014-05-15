#ifndef FastSimulation_TrackerSetup_TrackerInteractionGeometry_H
#define FastSimulation_TrackerSetup_TrackerInteractionGeometry_H
// v0  who ? when ? 
// 11 Dec 2003 Florian Beaudette. Removed the surfaces corresponding to ECAL 
//             This will carried out by the FamosTrajectoryManager
// 12 Oct 2006 Patrick Janot. Removed hardcoded active geometry & rings
//                            Removed RecHit smearing parameterization
// 16 Nov 2007 Patrick Janot. Make the whole thing configurable 

//FAMOS Headers
#include "FastSimulation/TrackerSetup/interface/TrackerLayer.h"

#include <list>
#include <vector>

class MediumProperties;
class GeometricSearchTracker;

namespace edm { 
  class ParameterSet;
}

class TrackerInteractionGeometry
{

 public:

  // put phase 1 layers after stanadCMS to avoid changing other code (fudge factors)
  enum FirstCylinders { PXB=0,PXD=3,TIB=5,TID=9,TOB=12,TEC=18,PXEXTRA=27 };

  /// Constructor : get the configurable parameters
  TrackerInteractionGeometry(const edm::ParameterSet& trackerMaterial,
			     const GeometricSearchTracker* geomSearchTracker);

  /// Destructor
  ~TrackerInteractionGeometry();

  /// Initialize the interaction geometry
  /// void initialize(const GeometricSearchTracker* geomSearchTracker);

  /// Returns the first pointer in the cylinder list
  inline std::list<TrackerLayer>::const_iterator cylinderBegin() const
    { return _theCylinders.begin(); }

  /// Returns the last pointer in the cylinder list
  inline std::list<TrackerLayer>::const_iterator cylinderEnd() const
    { return _theCylinders.end(); }

  /// Returns the number of cylinders in the Tracker
  inline const int nCylinders() const 
    { return static_cast<const int>(_theCylinders.size()); }

 private:

  // Fudge factors to apply to each layer material (private use only)
  std::vector<double> fudgeFactors(unsigned layerNr); 
  std::vector<double> minDim(unsigned layerNr);
  std::vector<double> maxDim(unsigned layerNr);
 
 private:

  /// The list of tracker (sensistive or not) layers
  std::list<TrackerLayer> _theCylinders;

  /// Thickness of all layers
  /// Version of the description
  unsigned int version;
  /// Beam Pipe
  std::vector<double> beamPipeThickness;
  /// Pixel Barrel Layers 1-3
  std::vector<double> pxbThickness;
  /// Pixel Barrel services at the end of layers 1-3
  std::vector<double> pxb1CablesThickness;
  std::vector<double> pxb2CablesThickness;
  std::vector<double> pxb3CablesThickness;
  /// Pixel Barrel outside cables
  std::vector<double> pxbOutCables1Thickness;
  std::vector<double> pxbOutCables2Thickness;
  /// Pixel Disks 1-2
  std::vector<double> pxdThickness;
  /// Pixel Endcap outside cables
  std::vector<double> pxdOutCables1Thickness;
  std::vector<double> pxdOutCables2Thickness;
  /// Tracker Inner barrel layers 1-4
  std::vector<double> tibLayer1Thickness;
  std::vector<double> tibLayer2Thickness;
  std::vector<double> tibLayer3Thickness;
  std::vector<double> tibLayer4Thickness;
  /// TIB outside services (endcap)
  std::vector<double> tibOutCables1Thickness;
  std::vector<double> tibOutCables2Thickness;
  /// Tracker Inner disks layers 1-3
  std::vector<double> tidLayer1Thickness;
  std::vector<double> tidLayer2Thickness;
  std::vector<double> tidLayer3Thickness;
  /// TID outside wall (endcap)
  std::vector<double> tidOutsideThickness;
  /// TOB inside wall (barrel)
  std::vector<double> tobInsideThickness;
  /// Tracker Outer barrel layers 1-6
  std::vector<double> tobLayer1Thickness;
  std::vector<double> tobLayer2Thickness;
  std::vector<double> tobLayer3Thickness;
  std::vector<double> tobLayer4Thickness;
  std::vector<double> tobLayer5Thickness;
  std::vector<double> tobLayer6Thickness;
  // TOB services (endcap)
  std::vector<double> tobOutsideThickness;
  // Tracker EndCap disks layers 1-9
  std::vector<double> tecLayerThickness;
  // TOB outside wall (barrel)
  std::vector<double> barrelCablesThickness;
  // TEC outside wall (endcap)
  std::vector<double> endcapCables1Thickness;
  std::vector<double> endcapCables2Thickness;

  /// Position of dead material layers (cables, services, etc.)
  /// Beam pipe
  std::vector<double> beamPipeRadius;
  std::vector<double> beamPipeLength;
  /// Cables and Services at the end of PIXB1,2,3 ("disk")
  std::vector<double> pxb1CablesInnerRadius;
  std::vector<double> pxb2CablesInnerRadius;
  std::vector<double> pxb3CablesInnerRadius;
  /// Pixel Barrel Outside walls and cables
  std::vector<double> pxbOutCables1InnerRadius;
  std::vector<double> pxbOutCables1OuterRadius;
  std::vector<double> pxbOutCables1ZPosition;
  std::vector<double> pxbOutCables2InnerRadius;
  std::vector<double> pxbOutCables2OuterRadius;
  std::vector<double> pxbOutCables2ZPosition;
  /// Pixel Outside walls and cables (barrel and endcaps)
  std::vector<double> pixelOutCablesRadius;
  std::vector<double> pixelOutCablesLength;
  std::vector<double> pixelOutCablesInnerRadius;
  std::vector<double> pixelOutCablesOuterRadius;
  std::vector<double> pixelOutCablesZPosition;
  /// Tracker Inner Barrel Outside Cables and walls (endcap) 
  std::vector<double> tibOutCables1InnerRadius;
  std::vector<double> tibOutCables1OuterRadius;
  std::vector<double> tibOutCables1ZPosition;
  std::vector<double> tibOutCables2InnerRadius;
  std::vector<double> tibOutCables2OuterRadius;
  std::vector<double> tibOutCables2ZPosition;
  /// Tracker outer barrel Inside wall (barrel)
  std::vector<double> tobInCablesRadius;
  std::vector<double> tobInCablesLength;
  /// Tracker Inner Disks Outside Cables and walls
  std::vector<double> tidOutCablesInnerRadius;
  std::vector<double> tidOutCablesZPosition;
  /// Tracker Outer Barrel Outside Cables and walls (barrel and endcaps)
  std::vector<double> tobOutCablesInnerRadius;
  std::vector<double> tobOutCablesOuterRadius;
  std::vector<double> tobOutCablesZPosition;
  std::vector<double> tobOutCablesRadius;
  std::vector<double> tobOutCablesLength;
  /// Tracker Endcaps Outside Cables and walls
  std::vector<double> tecOutCables1InnerRadius;
  std::vector<double> tecOutCables1OuterRadius;
  std::vector<double> tecOutCables1ZPosition;
  std::vector<double> tecOutCables2InnerRadius;
  std::vector<double> tecOutCables2OuterRadius;
  std::vector<double> tecOutCables2ZPosition;

  // Fudge factors for layer inhomogeneities
  std::vector<unsigned int> fudgeLayer;
  std::vector<double> fudgeMin;
  std::vector<double> fudgeMax;
  std::vector<double> fudgeFactor;

  /// The following list gives the thicknesses of the various layers.
  std::vector<MediumProperties*> _mediumProperties;

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
  MediumProperties *_theMPInner1;
  MediumProperties *_theMPInner2;
  MediumProperties *_theMPInner3;
  /// Some material in front of the tracker outer barrel (cylinder) 
  MediumProperties *_theMPTOBBInside;
  /// Some material around the tracker inner barrel (disk) 
  MediumProperties *_theMPTIBEOutside1;
  MediumProperties *_theMPTIBEOutside2;
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
