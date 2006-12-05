using namespace std;

//Framework Headers
#include "FWCore/Utilities/interface/Exception.h"

//CMSSW Headers
#include "Geometry/Surface/interface/Surface.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "Geometry/Surface/interface/SimpleCylinderBounds.h"
#include "Geometry/Surface/interface/BoundDisk.h"
#include "Geometry/Surface/interface/SimpleDiskBounds.h"

// Tracker/Tracking Headers
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "TrackingTools/PatternTools/interface/MediumProperties.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

//FAMOS Headers
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometry.h"

#include<iostream>

TrackerInteractionGeometry::TrackerInteractionGeometry(const GeometricSearchTracker* theGeomSearchTracker)
{
 
  // Check that the Reco Tracker Geometry has been loaded
  if ( !theGeomSearchTracker ) 
    throw cms::Exception("FastSimulation/TrackerInteractionGeometry") 
      << "The pointer to the GeometricSearchTracker was not set"; 

  // The vector of Barrel Tracker Layers 
  std::vector< BarrelDetLayer*> barrelLayers = 
    theGeomSearchTracker->barrelLayers();

  // The vector of Forward Tracker Layers (positive z)
  std::vector< ForwardDetLayer*>  posForwardLayers = 
    theGeomSearchTracker->posForwardLayers();

  // Local pointers
  BoundCylinder* theCylinder;
  BoundDisk* theDisk;

  // Fraction of radiation length : had oc values to account 
  // for detectors, cables, support, ...
  // Note : the second argument is not used in FAMOS
  // Note : the first argument is tuned to reproduce the CMSIM material
  //        in terms or radiation length.

  // The Beam pipe
  _theMPBeamPipe = new MediumProperties(0.0038,0.0001);  
  // The pixel barrel layers
  _theMPPixelBarrel = new MediumProperties(0.023,0.0001);  
  // The pixel endcap disks
  _theMPPixelEndcap = new MediumProperties(0.040,0.0001);  
  // The various cables thicknesses for each layer / disks
  _theMPPixelOutside1 = new MediumProperties(0.050,0.0001);  
  _theMPPixelOutside2 = new MediumProperties(0.040,0.0001);  
  _theMPPixelOutside3 = new MediumProperties(0.030,0.0001);  
  _theMPPixelOutside4 = new MediumProperties(0.040,0.0001);  
  _theMPPixelOutside  = new MediumProperties(0.025,0.0001);  
  _theMPPixelOutside5 = new MediumProperties(0.023,0.0001);  
  _theMPPixelOutside6 = new MediumProperties(0.085,0.0001);  
  // The tracker inner barrel layers
  _theMPTIB1 = new MediumProperties(0.060,0.0001);  
  _theMPTIB2 = new MediumProperties(0.047,0.0001);  
  _theMPTIB3 = new MediumProperties(0.032,0.0001);  
  _theMPTIB4 = new MediumProperties(0.030,0.0001);  
  // The tracker outer barrel layers
  _theMPTOB1 = new MediumProperties(0.044,0.0001);  
  _theMPTOB2 = new MediumProperties(0.044,0.0001);  
  _theMPTOB3 = new MediumProperties(0.033,0.0001);  
  _theMPTOB4 = new MediumProperties(0.033,0.0001);  
  _theMPTOB5 = new MediumProperties(0.033,0.0001);  
  _theMPTOB6 = new MediumProperties(0.033,0.0001);  
  // The tracker inner disks
  _theMPInner = new MediumProperties(0.050,0.0001);  
  // The tracker endcap disks
  _theMPEndcap = new MediumProperties(0.041,0.0001);  
  // Various cable thicknesses 
  _theMPTOBBInside = new MediumProperties(0.014,0.0001);  
  _theMPTIBEOutside = new MediumProperties(0.040,0.0001);  
  _theMPTIDEOutside = new MediumProperties(0.070,0.0001);  
  _theMPTOBEOutside = new MediumProperties(0.090,0.0001);  
  _theMPBarrelOutside = new MediumProperties(0.088,0.0001);  
  _theMPEndcapOutside = new MediumProperties(0.260,0.0001);  
  _theMPEndcapOutside2 = new MediumProperties(0.080,0.0001);  

  // Create the nest of cylinders
  const Surface::PositionType thePosition(0.,0.,0.);
  const Surface::RotationType theRotation(1.,0.,0.,0.,1.,0.,0.,0.,1.);
  // Beam Pipe
  //  const SimpleCylinderBounds  PIPE( 0.997,   1.003,  -300., 300.);
  const SimpleCylinderBounds  PIPE( 2.497,   2.503,  -26.4, 26.4);

  // Take the active layer position from the Tracker Reco Geometry
  // Pixel barrel
  std::vector< BarrelDetLayer*>::const_iterator bl = barrelLayers.begin();
  double pixelLength = (**bl).specificSurface().bounds().length()/2.;
  // First pixel barrel layer: r=4.41058, l=53.38
  const SimpleCylinderBounds  PIXB1( 
        (**bl).specificSurface().radius()-0.0150, 
        (**bl).specificSurface().radius()+0.0150,
	-pixelLength,
  	 pixelLength);
  // "Cables" 
  const SimpleDiskBounds PIXBOut1(4.0,5.2,-0.5,0.5);
  const Surface::PositionType PPIXBOut1(0.0,0.0,pixelLength+0.001);

  // Second pixel barrel layer: r=7.30732, l=53.38
  ++bl;
  const SimpleCylinderBounds  PIXB2( 
        (**bl).specificSurface().radius()-0.0150, 
        (**bl).specificSurface().radius()+0.0150, 
       -pixelLength-0.002,  
       +pixelLength+0.002);
  // "Cables"
  double maxRadius = (**bl).specificSurface().radius()+0.02;
  const SimpleDiskBounds PIXBOut2(6.5,maxRadius,-0.5,0.5);
  const Surface::PositionType PPIXBOut2(0.0,0.0,pixelLength+0.003);

  // More cables
  ++bl;
  maxRadius = (**bl).specificSurface().radius()-0.02;
  const SimpleDiskBounds PIXBOut3(9.0,maxRadius,-0.5,0.5);
  const Surface::PositionType PPIXBOut3(0.0,0.0,pixelLength+0.004);
  // Third pixel barrel layer: r=10.1726, l=53.38
  const SimpleCylinderBounds  PIXB3( 
        (**bl).specificSurface().radius()-0.0150, 
        (**bl).specificSurface().radius()+0.0150, 
       -pixelLength-0.005,  
       +pixelLength+0.005);

  // Pixel Barrel Outside walls and cables
  const SimpleDiskBounds PIXBOut4(12.5,14.9,-0.5,0.5);
  const Surface::PositionType PPIXBOut4(0.0,0.0,27.999);

  const SimpleDiskBounds PIXBOut(3.0,15.0,-0.5,0.5);
  const Surface::PositionType PPIXBOut(0.0,0.0,28.0);

  const SimpleCylinderBounds  PIXBOut5( 17.0, 17.2, -64.8, 64.8);

  const SimpleDiskBounds PIXBOut6(3.0,17.3,-0.5,0.5);
  const Surface::PositionType PPIXBOut6(0.0,0.0,64.9);


  // Tracker Inner Barrel : thin detectors (300 microns)
  // First TIB layer: r=25.6786, l=130.04
  ++bl;
  const SimpleCylinderBounds  TIB1( 
        (**bl).specificSurface().radius()-0.0150, 
        (**bl).specificSurface().radius()+0.0150, 
       -(**bl).specificSurface().bounds().length()/2.,  
       +(**bl).specificSurface().bounds().length()/2.);
  // Second TIB layer: r=34.0341, l=131.999
  ++bl;
  const SimpleCylinderBounds  TIB2( 
        (**bl).specificSurface().radius()-0.0150, 
        (**bl).specificSurface().radius()+0.0150, 
       -(**bl).specificSurface().bounds().length()/2.,  
       +(**bl).specificSurface().bounds().length()/2.);
  // Third TIB layer: r=41.9599, l=131.628  !!!! Needs to be larger than TIB2
  // This is so because TIB2 (and TIB1) have tilted modules.
  ++bl;
  const SimpleCylinderBounds  TIB3( 
        (**bl).specificSurface().radius()-0.0150, 
        (**bl).specificSurface().radius()+0.0150, 
       -(**bl).specificSurface().bounds().length()/2.-0.6,  
       +(**bl).specificSurface().bounds().length()/2.+0.6);
  // Fourth TIB layer: r=49.8924, l=132.78
  ++bl;
  const SimpleCylinderBounds  TIB4( 
        (**bl).specificSurface().radius()-0.0150, 
        (**bl).specificSurface().radius()+0.0150, 
       -(**bl).specificSurface().bounds().length()/2.-0.1,  
       +(**bl).specificSurface().bounds().length()/2.+0.1);

  // Inner Barrel Cylinder & Ends : Cables and walls
  const SimpleDiskBounds TIBEOut(22.5,50.4,-0.5,0.5);
  const Surface::PositionType PTIBEOut(0.0,0.0,71.5);

  const SimpleDiskBounds TIBEOut2(35.5,50.401,-0.5,0.5);
  const Surface::PositionType PTIBEOut2(0.0,0.0,71.501);

  // Inner Tracker / Outer Barrel Wall
  const SimpleCylinderBounds  TOBCIn ( 54.0, 55.0,-108.2,108.2);

  // First TOB layer: r=60.7671, l=216.576
  ++bl;
  const SimpleCylinderBounds  TOB1( 
        (**bl).specificSurface().radius()-0.0150, 
        (**bl).specificSurface().radius()+0.0150, 
       -(**bl).specificSurface().bounds().length()/2.,  
       +(**bl).specificSurface().bounds().length()/2.);
  // Second TOB layer: r=69.3966, l=216.576
  ++bl;
  const SimpleCylinderBounds  TOB2( 
        (**bl).specificSurface().radius()-0.0150, 
        (**bl).specificSurface().radius()+0.0150, 
       -(**bl).specificSurface().bounds().length()/2.,  
       +(**bl).specificSurface().bounds().length()/2.);
  // Third TOB layer: r=78.0686, l=216.576 !!! Needs to be larger than TOB2
  // This is so because TOB2 (and TOB1) have tilted modules.
  ++bl;
  const SimpleCylinderBounds  TOB3( 
        (**bl).specificSurface().radius()-0.0150, 
        (**bl).specificSurface().radius()+0.0150, 
       -(**bl).specificSurface().bounds().length()/2.-0.5,  
       +(**bl).specificSurface().bounds().length()/2.+0.5);
  // Fourth TOB layer: r=86.8618, l=216.576
  ++bl;
  const SimpleCylinderBounds  TOB4( 
        (**bl).specificSurface().radius()-0.0150, 
        (**bl).specificSurface().radius()+0.0150, 
       -(**bl).specificSurface().bounds().length()/2.-0.5,  
       +(**bl).specificSurface().bounds().length()/2.+0.5);
  // Fifth TOB layer: r=96.5557, l=216.576
  ++bl;
  const SimpleCylinderBounds  TOB5( 
        (**bl).specificSurface().radius()-0.0150, 
        (**bl).specificSurface().radius()+0.0150, 
       -(**bl).specificSurface().bounds().length()/2.-0.5,  
       +(**bl).specificSurface().bounds().length()/2.+0.5);
  // Sixth TOB layer: r=108.05, l=216.576
  ++bl;
  const SimpleCylinderBounds  TOB6( 
        (**bl).specificSurface().radius()-0.0150, 
        (**bl).specificSurface().radius()+0.0150, 
       -(**bl).specificSurface().bounds().length()/2.-0.5,  
       +(**bl).specificSurface().bounds().length()/2.+0.5);

  const SimpleDiskBounds TOBEOut(55.0,109.5,-0.5,0.5);
  const Surface::PositionType PTOBEOut(0.0,0.0,110.0);

  const Surface::RotationType theRotation2(1.,0.,0.,0.,1.,0.,0.,0.,1.);

  // Outside : Barrel
  const SimpleCylinderBounds  TBOut ( 119.0, 120.0,-299.9,299.9);

  // And now the disks...
  std::vector< ForwardDetLayer*>::const_iterator fl = posForwardLayers.begin();

  // Pixel disks 
  // First Pixel disk: Z pos 35.5 radii 5.42078, 16.0756
  const SimpleDiskBounds PIXD1(
	  (**fl).specificSurface().innerRadius(), 
	  (**fl).specificSurface().outerRadius(),
	  -0.015,0.015);
  const Surface::PositionType PPIXD1(0.0,0.0,(**fl).surface().position().z()); 
  // Second Pixel disk: Z pos 48.5 radii 5.42078, 16.0756
  ++fl;
  const SimpleDiskBounds PIXD2(
	  (**fl).specificSurface().innerRadius(), 
	  (**fl).specificSurface().outerRadius(),
	  -0.015,0.015);
  const Surface::PositionType PPIXD2(0.0,0.0,(**fl).surface().position().z()); 

  // Tracker Inner disks (add 3 cm for the outer radius to simulate cables, 
  // and remove 1cm to inner radius to allow for some extrapolation margin)
  // First TID : Z pos 78.445 radii 23.14, 50.4337
  ++fl;
  const SimpleDiskBounds TID1(
	  (**fl).specificSurface().innerRadius()-1.0, 
	  (**fl).specificSurface().outerRadius()+3.5,
	  -0.015,0.015);
  const Surface::PositionType PTID1(0.,0.,(**fl).surface().position().z()); 
  // Second TID : Z pos 90.445 radii 23.14, 50.4337
  ++fl;
  const SimpleDiskBounds TID2(
	  (**fl).specificSurface().innerRadius()-1.0, 
	  (**fl).specificSurface().outerRadius()+3.5,
	  -0.015,0.015);
  const Surface::PositionType PTID2(0.,0.,(**fl).surface().position().z()); 
  // Third TID : Z pos 105.445 radii 23.14, 50.4337
  ++fl;
  const SimpleDiskBounds TID3(
	  (**fl).specificSurface().innerRadius()-1.0, 
	  (**fl).specificSurface().outerRadius()+3.5,
	  -0.015,0.015);
  const Surface::PositionType PTID3(0.,0.,(**fl).surface().position().z()); 

  // TID Wall and cables
  const SimpleDiskBounds TIDEOut(32.0,54.004,-0.5,0.5);
  const Surface::PositionType PTIDEOut(0.0,0.0,108.0);


  // Tracker Endcaps : Add 11 cm to outer radius to correct for a bug, remove
  // 5cm to the inner radius (TEC7,8,9) to correct for a simular bug, and
  // remove other 2cm to inner radius to allow for some extrapolation margin
  // First TEC: Z pos 131.892 radii 23.3749, 99.1967
  ++fl;
  const SimpleDiskBounds TEC1(
	  (**fl).specificSurface().innerRadius()-2.0, 
	  (**fl).specificSurface().outerRadius()+2.0,
	  -0.015,0.015);
  const Surface::PositionType PTEC1(0.,0,(**fl).surface().position().z()); 
  // Second TEC: Z pos 145.892 radii 23.3749, 99.1967
  ++fl;
  const SimpleDiskBounds TEC2(
	  (**fl).specificSurface().innerRadius()-2.0, 
	  (**fl).specificSurface().outerRadius()+2.0,
	  -0.015,0.015);
  const Surface::PositionType PTEC2(0.,0.,(**fl).surface().position().z());
  // Third TEC: Z pos 159.892 radii 23.3749, 99.1967
  ++fl;
  const SimpleDiskBounds TEC3(
	  (**fl).specificSurface().innerRadius()-2.0, 
	  (**fl).specificSurface().outerRadius()+2.0,
	  -0.015,0.015);
  const Surface::PositionType PTEC3(0.,0.,(**fl).surface().position().z());
  // Fourth TEC: Z pos 173.892 radii 32.1263, 99.1967
  ++fl;
  const SimpleDiskBounds TEC4(
	  (**fl).specificSurface().innerRadius()-2.0, 
	  (**fl).specificSurface().outerRadius()+2.0,
	  -0.015,0.015);
  const Surface::PositionType PTEC4(0.,0.,(**fl).surface().position().z());
  // Fifth TEC: Z pos 187.892 radii 32.1263, 99.1967
  ++fl;
  const SimpleDiskBounds TEC5(
	  (**fl).specificSurface().innerRadius()-2.0, 
	  (**fl).specificSurface().outerRadius()+2.0,
	  -0.015,0.015);
  const Surface::PositionType PTEC5(0.,0.,(**fl).surface().position().z());
  // Sixth TEC: Z pos 205.392 radii 32.1263, 99.1967
  ++fl;
  const SimpleDiskBounds TEC6(
	  (**fl).specificSurface().innerRadius()-2.0, 
	  (**fl).specificSurface().outerRadius()+2.0,
	  -0.015,0.015);
  const Surface::PositionType PTEC6(0.,0.,(**fl).surface().position().z());
  // Seventh TEC: Z pos 224.121 radii 44.7432, 99.1967
  ++fl;
  const SimpleDiskBounds TEC7(
	  (**fl).specificSurface().innerRadius()-2.0, 
	  (**fl).specificSurface().outerRadius()+2.0,
	  -0.015,0.015);
  const Surface::PositionType PTEC7(0.,0.,(**fl).surface().position().z());
  // Eighth TEC: Z pos 244.621 radii 44.7432, 99.1967
  ++fl;
  const SimpleDiskBounds TEC8(
	  (**fl).specificSurface().innerRadius()-2.0, 
	  (**fl).specificSurface().outerRadius()+2.0,
	  -0.015,0.015);
  const Surface::PositionType PTEC8(0.,0.,(**fl).surface().position().z());
  // Nineth TEC: Z pos 266.121 radii 56.1781, 99.1967
  ++fl;
  const SimpleDiskBounds TEC9(
	  (**fl).specificSurface().innerRadius()-2.0, 
	  (**fl).specificSurface().outerRadius()+2.0,
	  -0.015,0.015);
  const Surface::PositionType PTEC9(0.,0.,(**fl).surface().position().z());

  // Outside : Endcap
  const SimpleDiskBounds TEOut(6.0,120.001,-0.5,0.5);
  const Surface::PositionType PTEOut(0.0,0.0,300.0);

  const SimpleDiskBounds TEOut2(70.0,120.001,-0.5,0.5);
  const Surface::PositionType PTEOut2(0.0,0.0,300.0);

  // The ordering of disks and cylinders is essential here
  // (from inside to outside)
  // Do not change it thoughtlessly.


  // Beam Pipe

  theCylinder = new BoundCylinder(thePosition,theRotation,PIPE);
  theCylinder->setMediumProperties(_theMPBeamPipe);
  _theCylinders.push_back(TrackerLayer(theCylinder,false));

  // Pixels 

  theCylinder = new BoundCylinder(thePosition,theRotation,PIXB1);
  theCylinder->setMediumProperties(_theMPPixelBarrel);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,1));

  theDisk = new BoundDisk(PPIXBOut1,theRotation2,PIXBOut1);
  theDisk->setMediumProperties(_theMPPixelOutside1);
  _theCylinders.push_back(TrackerLayer(theDisk,true));

  theCylinder = new BoundCylinder(thePosition,theRotation,PIXB2);
  theCylinder->setMediumProperties(_theMPPixelBarrel);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,2));

  theDisk = new BoundDisk(PPIXBOut2,theRotation2,PIXBOut2);
  theDisk->setMediumProperties(_theMPPixelOutside2);
  _theCylinders.push_back(TrackerLayer(theDisk,true));

  theDisk = new BoundDisk(PPIXBOut3,theRotation2,PIXBOut3);
  theDisk->setMediumProperties(_theMPPixelOutside3);
  _theCylinders.push_back(TrackerLayer(theDisk,true));

  theCylinder = new BoundCylinder(thePosition,theRotation,PIXB3);
  theCylinder->setMediumProperties(_theMPPixelBarrel);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,3));

  theDisk = new BoundDisk(PPIXBOut4,theRotation2,PIXBOut4);
  theDisk->setMediumProperties(_theMPPixelOutside4);
  _theCylinders.push_back(TrackerLayer(theDisk,true));

  theDisk = new BoundDisk(PPIXBOut,theRotation2,PIXBOut);
  theDisk->setMediumProperties(_theMPPixelOutside);
  _theCylinders.push_back(TrackerLayer(theDisk,true));

  theDisk = new BoundDisk(PPIXD1,theRotation2,PIXD1);
  theDisk->setMediumProperties(_theMPPixelEndcap);
  _theCylinders.push_back(TrackerLayer(theDisk,true,4));

  theDisk = new BoundDisk(PPIXD2,theRotation2,PIXD2);
  theDisk->setMediumProperties(_theMPPixelEndcap);
  _theCylinders.push_back(TrackerLayer(theDisk,true,5));

  theCylinder = new BoundCylinder(thePosition,theRotation,PIXBOut5);
  theCylinder->setMediumProperties(_theMPPixelOutside5);
  _theCylinders.push_back(TrackerLayer(theCylinder,false));

  theDisk = new BoundDisk(PPIXBOut6,theRotation2,PIXBOut6);
  theDisk->setMediumProperties(_theMPPixelOutside6);
  _theCylinders.push_back(TrackerLayer(theDisk,true));


  // Inner Barrel 

  theCylinder = new BoundCylinder(thePosition,theRotation,TIB1);
  theCylinder->setMediumProperties(_theMPTIB1);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,6));


  theCylinder = new BoundCylinder(thePosition,theRotation,TIB2);
  theCylinder->setMediumProperties(_theMPTIB2);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,7));

  theCylinder = new BoundCylinder(thePosition,theRotation,TIB3);
  theCylinder->setMediumProperties(_theMPTIB3);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,8));

  theCylinder = new BoundCylinder(thePosition,theRotation,TIB4);
  theCylinder->setMediumProperties(_theMPTIB4);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,9));


  theDisk = new BoundDisk(PTIBEOut,theRotation2,TIBEOut);
  theDisk->setMediumProperties(_theMPTIBEOutside);
  _theCylinders.push_back(TrackerLayer(theDisk,true));

  theDisk = new BoundDisk(PTIBEOut2,theRotation2,TIBEOut2);
  theDisk->setMediumProperties(_theMPTIBEOutside);
  _theCylinders.push_back(TrackerLayer(theDisk,true));

  
  // Inner Endcaps

  theDisk = new BoundDisk(PTID1,theRotation2,TID1);
  theDisk->setMediumProperties(_theMPInner);
  _theCylinders.push_back(TrackerLayer(theDisk,10,1,3));

  theDisk = new BoundDisk(PTID2,theRotation2,TID2);
  theDisk->setMediumProperties(_theMPInner);
  _theCylinders.push_back(TrackerLayer(theDisk,11,1,3));

  theDisk = new BoundDisk(PTID3,theRotation2,TID3);
  theDisk->setMediumProperties(_theMPInner);
  _theCylinders.push_back(TrackerLayer(theDisk,12,1,3));

  theDisk = new BoundDisk(PTIDEOut,theRotation2,TIDEOut);
  theDisk->setMediumProperties(_theMPTIDEOutside);
  _theCylinders.push_back(TrackerLayer(theDisk,true));


  // Outer Barrel 

  theCylinder = new BoundCylinder(thePosition,theRotation,TOBCIn);
  theCylinder->setMediumProperties(_theMPTOBBInside);
  _theCylinders.push_back(TrackerLayer(theCylinder,false));

  theCylinder = new BoundCylinder(thePosition,theRotation,TOB1);
  theCylinder->setMediumProperties(_theMPTOB1);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,13));

  theCylinder = new BoundCylinder(thePosition,theRotation,TOB2);
  theCylinder->setMediumProperties(_theMPTOB2);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,14));

  theCylinder = new BoundCylinder(thePosition,theRotation,TOB3);
  theCylinder->setMediumProperties(_theMPTOB3);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,15));

  theCylinder = new BoundCylinder(thePosition,theRotation,TOB4);
  theCylinder->setMediumProperties(_theMPTOB4);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,16));

  theCylinder = new BoundCylinder(thePosition,theRotation,TOB5);
  theCylinder->setMediumProperties(_theMPTOB5);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,17));

  theCylinder = new BoundCylinder(thePosition,theRotation,TOB6);
  theCylinder->setMediumProperties(_theMPTOB6);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,18));

  theDisk = new BoundDisk(PTOBEOut,theRotation2,TOBEOut);
  theDisk->setMediumProperties(_theMPTOBEOutside);
  _theCylinders.push_back(TrackerLayer(theDisk,true));


  // Outer Endcaps
 
  theDisk = new BoundDisk(PTEC1,theRotation2,TEC1);
  theDisk->setMediumProperties(_theMPEndcap);
  _theCylinders.push_back(TrackerLayer(theDisk,19,1,7));

  theDisk = new BoundDisk(PTEC2,theRotation2,TEC2);
  theDisk->setMediumProperties(_theMPEndcap);
  _theCylinders.push_back(TrackerLayer(theDisk,20,1,7));

  theDisk = new BoundDisk(PTEC3,theRotation2,TEC3);
  theDisk->setMediumProperties(_theMPEndcap);
  _theCylinders.push_back(TrackerLayer(theDisk,21,1,7));

  theDisk = new BoundDisk(PTEC4,theRotation2,TEC4);
  theDisk->setMediumProperties(_theMPEndcap);
  _theCylinders.push_back(TrackerLayer(theDisk,22,2,7));

  theDisk = new BoundDisk(PTEC5,theRotation2,TEC5);
  theDisk->setMediumProperties(_theMPEndcap);
  _theCylinders.push_back(TrackerLayer(theDisk,23,2,7));

  theDisk = new BoundDisk(PTEC6,theRotation2,TEC6);
  theDisk->setMediumProperties(_theMPEndcap);
  _theCylinders.push_back(TrackerLayer(theDisk,24,2,7));

  theDisk = new BoundDisk(PTEC7,theRotation2,TEC7);
  theDisk->setMediumProperties(_theMPEndcap);
  _theCylinders.push_back(TrackerLayer(theDisk,25,3,7));

  theDisk = new BoundDisk(PTEC8,theRotation2,TEC8);
  theDisk->setMediumProperties(_theMPEndcap);
  _theCylinders.push_back(TrackerLayer(theDisk,26,3,7));

  theDisk = new BoundDisk(PTEC9,theRotation2,TEC9);
  theDisk->setMediumProperties(_theMPEndcap);
  _theCylinders.push_back(TrackerLayer(theDisk,27,4,7));


  // Tracker Outside

  theCylinder = new BoundCylinder(thePosition,theRotation,TBOut);
  theCylinder->setMediumProperties(_theMPBarrelOutside);
  _theCylinders.push_back(TrackerLayer(theCylinder,false));

  theDisk = new BoundDisk(PTEOut,theRotation2,TEOut);
  theDisk->setMediumProperties(_theMPEndcapOutside);
  _theCylinders.push_back(TrackerLayer(theDisk,true));

  theDisk = new BoundDisk(PTEOut2,theRotation2,TEOut2);
  theDisk->setMediumProperties(_theMPEndcapOutside2);
  _theCylinders.push_back(TrackerLayer(theDisk,true));

  // Check overall compatibility of cylinder dimensions
  // (must be nested cylinders)
  // Throw an exception if the test fails
  double zin, rin;
  double zout, rout;
  unsigned nCyl=0;
  std::list<TrackerLayer>::iterator cyliterOut=cylinderBegin();
  // Inner cylinder dimensions
  if ( cyliterOut->forward() ) {
    zin = cyliterOut->disk()->position().z();
    rin = cyliterOut->disk()->outerRadius();
  } else {
    zin = cyliterOut->cylinder()->bounds().length()/2.;
    rin = cyliterOut->cylinder()->bounds().width()/2.;
  }
  // Go to the next cylinder
  ++cyliterOut;
  // And loop over all cylinders
  while ( cyliterOut != cylinderEnd() ) {
    // Outer cylinder dimensions
    if ( cyliterOut->forward() ) {
      zout = cyliterOut->disk()->position().z();
      rout = cyliterOut->disk()->outerRadius();
    } else {
      zout = cyliterOut->cylinder()->bounds().length()/2.;
      rout = cyliterOut->cylinder()->bounds().width()/2.;
    }

    nCyl++;
    if ( zout < zin || rout < rin ) { 
      throw cms::Exception("FastSimulation/TrackerInteractionGeometry ") 
	<< " WARNING with cylinder number " << nCyl 
	<< " (Active Layer Number = " <<  cyliterOut->layerNumber() 
	<< " Forward ? " <<  cyliterOut->forward() << " ) "
	<< " has dimensions smaller than previous cylinder : " << endl
	<< " zout/zin = " << zout << " " << zin << endl
	<< " rout/rin = " << rout << " " << rin << endl;
    } else {
      //      cout << " Cylinder number " << nCyl 
      //	   << " (Active Layer Number = " <<  cyliterOut->layerNumber() 
      //	   << " Forward ? " <<  cyliterOut->forward() << " ) "
      //	<< " has dimensions of : " 
      //	<< " zout = " << zout << "; " 
      //	<< " rout = " << rout << endl;
    }
    // Go to the next cylinder
    cyliterOut++;
    // Inner cylinder becomes outer cylinder
    zin = zout;
    rin = rout;
    // End test
  } 
    
}

TrackerInteractionGeometry::~TrackerInteractionGeometry()
{
  _theCylinders.clear();
  //  _theRings.clear();

  // The Beam pipe
  delete _theMPBeamPipe;
  // The pixel barrel layers
  delete _theMPPixelBarrel;
  // The pixel endcap disks
  delete _theMPPixelEndcap;
  // The various cables thicnesses for each layer / disks
  delete _theMPPixelOutside1;
  delete _theMPPixelOutside2;
  delete _theMPPixelOutside3;
  delete _theMPPixelOutside4;
  delete _theMPPixelOutside;
  delete _theMPPixelOutside5;
  delete _theMPPixelOutside6;
  // The tracker inner barrel layers
  delete _theMPTIB1;
  delete _theMPTIB2;
  delete _theMPTIB3;
  delete _theMPTIB4;
  // The tracker outer barrel layers
  delete _theMPTOB1;
  delete _theMPTOB2;
  delete _theMPTOB3;
  delete _theMPTOB4;
  delete _theMPTOB5;
  delete _theMPTOB6;
  // The tracker inner disks
  delete _theMPInner;
  // The tracker endcap disks
  delete _theMPEndcap;
  // Various cable thicknesses 
  delete _theMPTOBBInside;
  delete _theMPTIBEOutside;
  delete _theMPTIDEOutside;
  delete _theMPTOBEOutside;
  delete _theMPBarrelOutside;
  delete _theMPEndcapOutside;
  delete _theMPEndcapOutside2;
}
