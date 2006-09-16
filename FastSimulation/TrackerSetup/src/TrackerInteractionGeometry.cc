using namespace std;

//CMSSW Headers
#include "Geometry/Surface/interface/Surface.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "Geometry/Surface/interface/SimpleCylinderBounds.h"
#include "Geometry/Surface/interface/BoundDisk.h"
#include "Geometry/Surface/interface/SimpleDiskBounds.h"
#include "TrackingTools/PatternTools/interface/MediumProperties.h"

//FAMOS Headers
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometry.h"

#include<iostream>

TrackerInteractionGeometry::TrackerInteractionGeometry()
{
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

  // The Pixel error parametrization comes from a file
  //  _thePixelErrorParametrization = new PixelErrorParametrization();

  // Read the layer resolution and efficiency
  double f = 1./sqrt(12.);
  double c = 1./cos(0.05);
  double s = 1./sin(0.05);

  // The following list gives the local x/y hit resolutions and 
  // hit efficiencies for single muons, in the various tracker 
  // layers and ring
  double TIB1x = 0.00195*c;
  double TIB1y = 0.00195*s;
  double TIB1e = 0.99;
  double TIB2x = 0.00191*c;
  double TIB2y = 0.00191*s;
  double TIB2e = 0.99;
  double TIB3x = 0.00325;
  double TIB3y = 12.*f;
  double TIB3e = 0.99;
  double TIB4x = 0.00323;
  double TIB4y = 12.*f;
  double TIB4e = 0.99;
  // Outer Barrel
  double TOB1x = 0.00461*c;
  double TOB1y = 0.00461*s;
  double TOB1e = 0.99;
  double TOB2x = 0.00458*c;
  double TOB2y = 0.00458*s;
  double TOB2e = 0.99;
  double TOB3x = 0.00488;
  double TOB3y = 19.*f;
  double TOB3e = 0.99;
  double TOB4x = 0.00491;
  double TOB4y = 19.*f;
  double TOB4e = 0.99;
  double TOB5x = 0.00293;
  double TOB5y = 19.*f;
  double TOB5e = 0.99;
  double TOB6x = 0.00299;
  double TOB6y = 19.*f;
  double TOB6e = 0.99;
  // Mini Endcap and forward Rings
  double Ring1r = 23.31;
  double Ring1R = 32.21;
  double Ring1x = 0.00262*c;
  double Ring1y = 0.00262*s;
  double Ring1e = 0.99;
  double Ring2r = 32.22;
  double Ring2R = 40.02;
  double Ring2x = 0.00354*c;
  double Ring2y = 0.00354*s;
  double Ring2e = 0.99;
  double Ring3r = 40.0;
  double Ring3R = 50.3;
  double Ring3x = 0.00391;
  double Ring3y = 11.*f;
  double Ring3e = 0.99;
  double Ring4r = 50.34;
  double Ring4R = 61.04;
  double Ring4x = 0.00346;
  double Ring4y = 12.*f;
  double Ring4e = 0.99;
  double Ring5r = 61.05;
  double Ring5R = 74.05;
  double Ring5x = 0.00378*c;
  double Ring5y = 0.00378*s;
  double Ring5e = 0.99;
  double Ring6r = 74.06;
  double Ring6R = 90.06;
  double Ring6x = 0.00508;
  double Ring6y = 19.*f;
  double Ring6e = 0.99;
  double Ring7r = 90.0;
  double Ring7R = 109.0;
  double Ring7x = 0.00422;
  double Ring7y = 20.5*f;
  double Ring7e = 0.99;


  // Create the nest of cylinders
  const Surface::PositionType thePosition(0.,0.,0.);
  const Surface::RotationType theRotation(1.,0.,0.,0.,1.,0.,0.,0.,1.);
  // Beam Pipe
  //  const SimpleCylinderBounds  PIPE( 0.997,   1.003,  -300., 300.);
  const SimpleCylinderBounds  PIPE( 2.497,   2.503,  -26.4, 26.4);

  // Pixel barrel : thin detectors (300 microns)
  const SimpleCylinderBounds  PIXB1( 4.36,   4.39,  -26.5,26.5);
  const SimpleCylinderBounds  PIXB2( 7.2705, 7.3005,-26.502,26.502);
  const SimpleCylinderBounds  PIXB3(10.142, 10.172, -26.504,26.505);

  // Pixel Barrel Outside
  // ***  CHECK if this material exists *** 
  // (Something like that is need to reproduce the Pixel Material...)
  const SimpleDiskBounds PIXBOut1(4.0,5.2,-0.5,0.5);
  const Surface::PositionType PPIXBOut1(0.0,0.0,26.501);
  const SimpleDiskBounds PIXBOut2(6.5,7.31,-0.5,0.5);
  const Surface::PositionType PPIXBOut2(0.0,0.0,26.503);
  const SimpleDiskBounds PIXBOut3(9.0,10.1,-0.5,0.5);
  const Surface::PositionType PPIXBOut3(0.0,0.0,26.504);
  const SimpleDiskBounds PIXBOut4(12.5,14.9,-0.5,0.5);
  const Surface::PositionType PPIXBOut4(0.0,0.0,27.999);
  const SimpleDiskBounds PIXBOut(3.0,15.0,-0.5,0.5);
  const Surface::PositionType PPIXBOut(0.0,0.0,28.0);

  const SimpleCylinderBounds  PIXBOut5( 17.0, 17.2, -64.8, 64.8);

  const SimpleDiskBounds PIXBOut6(3.0,17.3,-0.5,0.5);
  const Surface::PositionType PPIXBOut6(0.0,0.0,64.9);


  // Tracker Inner Barrel : thin detectors (300 microns)
  const SimpleCylinderBounds  TIB1( 25.3305, 25.6305, -70.0, 70.0);
  const SimpleCylinderBounds  TIB2( 33.979, 34.279,   -70.0, 70.0);
  const SimpleCylinderBounds  TIB3( 41.7397, 42.0397, -70.0, 70.0);
  const SimpleCylinderBounds  TIB4( 49.7112, 50.0112, -70.0, 70.0);

  // Inner Barrel Cylinder & Ends

  const SimpleDiskBounds TIBEOut(22.5,53.0,-0.5,0.5);
  const Surface::PositionType PTIBEOut(0.0,0.0,71.5);
  const SimpleDiskBounds TIBEOut2(35.5,53.001,-0.5,0.5);
  const Surface::PositionType PTIBEOut2(0.0,0.0,71.501);

  // Tracker Outer Barrel : thick detectors (500 microns)
  const SimpleCylinderBounds  TOB1( 60.7671, 61.2671, -109.0,109.0);
  const SimpleCylinderBounds  TOB2( 69.3966, 69.8966, -109.0,109.0);
  const SimpleCylinderBounds  TOB3( 78.0686, 78.5686, -109.0,109.0);
  const SimpleCylinderBounds  TOB4( 86.55, 86.95,-109.0,109.0);
  const SimpleCylinderBounds  TOB5( 96.25, 96.75,-109.0,109.0);
  const SimpleCylinderBounds  TOB6(107.75,108.25,-109.0,109.0);

  // Outer Barrel Cylinder & Ends
  const SimpleCylinderBounds  TOBCIn ( 54.0, 55.0,-109.0,109.0);
  const SimpleCylinderBounds  TOBCOut(108.5,109.5,-109.0,109.0);

  const SimpleDiskBounds TOBEOut(55.0,109.5,-0.5,0.5);
  const Surface::PositionType PTOBEOut(0.0,0.0,110.0);

  const Surface::RotationType theRotation2(1.,0.,0.,0.,1.,0.,0.,0.,1.);

  // Outside : Barrel
  const SimpleCylinderBounds  TBOut ( 119.0, 120.0,-299.9,299.9);

  // Pixel disks : thin detectors (300 microns)
  const SimpleDiskBounds PIXD1(6.0,15.001,-0.015,0.015);
  const Surface::PositionType PPIXD1(0.0,0.0,35.50); 
  const SimpleDiskBounds PIXD2(6.0,15.002,-0.015,0.015);
  const Surface::PositionType PPIXD2(0.0,0.0,48.50); 
  // Tracker Inner disks : thin detectors (300 microns)
  const SimpleDiskBounds TID1(23.3,54.001,-0.15,0.15);
  const Surface::PositionType PTID1(0.0,0.0,78.445); 
  const SimpleDiskBounds TID2(23.3,54.002,-0.15,0.15);
  const Surface::PositionType PTID2(0.0,0.0,90.445); 
  const SimpleDiskBounds TID3(23.3,54.003,-0.15,0.15);
  const Surface::PositionType PTID3(0.0,0.0,105.445); 

  const SimpleDiskBounds TIDEOut(32.0,54.004,-0.5,0.5);
  const Surface::PositionType PTIDEOut(0.0,0.0,108.4);


  // Tracker Endcaps : thick detectors (500 microns)
  // *** Warning*** the inner three rings are made
  // of thin detectors.... not reproduced here. 
  const SimpleDiskBounds TEC1(23.3,109.501,-0.25,0.25);
  const Surface::PositionType PTEC1(0.0,0.0,131.892);
  const SimpleDiskBounds TEC2(23.3,109.502,-0.25,0.25);
  const Surface::PositionType PTEC2(0.0,0.0,145.892);
  const SimpleDiskBounds TEC3(23.3,109.503,-0.25,0.25);
  const Surface::PositionType PTEC3(0.0,0.0,158.892);
  const SimpleDiskBounds TEC4(23.3,109.504,-0.25,0.25);
  const Surface::PositionType PTEC4(0.0,0.0,173.892);
  const SimpleDiskBounds TEC5(28.0,109.505,-0.25,0.25);
  const Surface::PositionType PTEC5(0.0,0.0,187.892);
  const SimpleDiskBounds TEC6(28.0,109.506,-0.25,0.25);
  const Surface::PositionType PTEC6(0.0,0.0,205.392);
  const SimpleDiskBounds TEC7(28.0,109.507,-0.25,0.25);
  const Surface::PositionType PTEC7(0.0,0.0,224.121);
  const SimpleDiskBounds TEC8(28.0,109.508,-0.25,0.25);
  const Surface::PositionType PTEC8(0.0,0.0,244.621);
  const SimpleDiskBounds TEC9(28.0,109.509,-0.25,0.25);
  const Surface::PositionType PTEC9(0.0,0.0,266.121);

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
  _theCylinders.push_back(TrackerLayer(theCylinder,false,6,TIB1x,TIB1y,TIB1e));


  theCylinder = new BoundCylinder(thePosition,theRotation,TIB2);
  theCylinder->setMediumProperties(_theMPTIB2);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,7,TIB2x,TIB2y,TIB2e));

  theCylinder = new BoundCylinder(thePosition,theRotation,TIB3);
  theCylinder->setMediumProperties(_theMPTIB3);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,8,TIB3x,TIB3y,TIB3e));

  theCylinder = new BoundCylinder(thePosition,theRotation,TIB4);
  theCylinder->setMediumProperties(_theMPTIB4);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,9,TIB4x,TIB4y,TIB4e));


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
  _theCylinders.push_back(TrackerLayer(theCylinder,false,13,TOB1x,TOB1y,TOB1e));

  theCylinder = new BoundCylinder(thePosition,theRotation,TOB2);
  theCylinder->setMediumProperties(_theMPTOB2);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,14,TOB2x,TOB2y,TOB2e));

  theCylinder = new BoundCylinder(thePosition,theRotation,TOB3);
  theCylinder->setMediumProperties(_theMPTOB3);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,15,TOB3x,TOB3y,TOB3e));

  theCylinder = new BoundCylinder(thePosition,theRotation,TOB4);
  theCylinder->setMediumProperties(_theMPTOB4);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,16,TOB4x,TOB4y,TOB4e));

  theCylinder = new BoundCylinder(thePosition,theRotation,TOB5);
  theCylinder->setMediumProperties(_theMPTOB5);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,17,TOB5x,TOB5y,TOB5e));

  theCylinder = new BoundCylinder(thePosition,theRotation,TOB6);
  theCylinder->setMediumProperties(_theMPTOB6);
  _theCylinders.push_back(TrackerLayer(theCylinder,false,18,TOB6x,TOB6y,TOB6e));

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


  // .. finally create the rings with their characteristics
  _theRings.insert(pair<unsigned,TrackerRing>
		   (1,TrackerRing(Ring1r,Ring1R,Ring1x,Ring1y,Ring1e) ) );
  _theRings.insert(pair<unsigned,TrackerRing>
		   (2,TrackerRing(Ring2r,Ring2R,Ring2x,Ring2y,Ring2e) ) );
  _theRings.insert(pair<unsigned,TrackerRing>
		   (3,TrackerRing(Ring3r,Ring3R,Ring3x,Ring3y,Ring3e) ) );
  _theRings.insert(pair<unsigned,TrackerRing>
		   (4,TrackerRing(Ring4r,Ring4R,Ring4x,Ring4y,Ring4e) ) );
  _theRings.insert(pair<unsigned,TrackerRing>
		   (5,TrackerRing(Ring5r,Ring5R,Ring5x,Ring5y,Ring5e) ) );
  _theRings.insert(pair<unsigned,TrackerRing>
		   (6,TrackerRing(Ring6r,Ring6R,Ring6x,Ring6y,Ring6e) ) );
  _theRings.insert(pair<unsigned,TrackerRing>
		   (7,TrackerRing(Ring7r,Ring7R,Ring7x,Ring7y,Ring7e) ) );

}

TrackerInteractionGeometry::~TrackerInteractionGeometry()
{
  _theCylinders.clear();
  _theRings.clear();

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

unsigned 
TrackerInteractionGeometry::theRingNr(double radius, unsigned first, unsigned last) const {

  if ( radius < theRing(first).innerRadius() || 
       radius > theRing(last).outerRadius() ) return 0;

  unsigned myRingNr;
  for ( myRingNr=first; myRingNr<=last; ++myRingNr ) {

    TrackerRing myRing = theRing(myRingNr);
    if ( radius >=  myRing.innerRadius() && 
	 radius <=  myRing.outerRadius() ) break;

  }

  return myRingNr;

}

