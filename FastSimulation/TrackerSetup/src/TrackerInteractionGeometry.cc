//Framework Headers
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//CMSSW Headers
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"

// Tracker/Tracking Headers
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

//FAMOS Headers
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometry.h"

#include<iostream>

TrackerInteractionGeometry::TrackerInteractionGeometry(const edm::ParameterSet& trackerMaterial,
						       const GeometricSearchTracker* theGeomSearchTracker)
{

  use_hardcoded = trackerMaterial.getParameter<bool >("use_hardcoded_geometry"); 

  if(!use_hardcoded){
    std::vector<double> disk_thickness = trackerMaterial.getParameter<std::vector<double> >("disk_thickness");
    std::vector<double> disk_inner_radius = trackerMaterial.getParameter<std::vector<double> >("disk_inner_radius");
    std::vector<double> disk_outer_radius = trackerMaterial.getParameter<std::vector<double> >("disk_outer_radius");
    std::vector<double> disk_z = trackerMaterial.getParameter<std::vector<double> >("disk_z");
    
    assert(disk_inner_radius.size() == disk_outer_radius.size() && disk_inner_radius.size() == disk_z.size() && disk_inner_radius.size() ==  disk_thickness.size());
    std::cout << "number of disk layers = " << disk_z.size() << std::endl;
    
    const Surface::RotationType theRotation2(1.,0.,0.,0.,1.,0.,0.,0.,1.);
        
    std::vector<double> barrel_thickness = trackerMaterial.getParameter<std::vector<double> >("barrel_thickness");
    std::vector<double> barrel_radius = trackerMaterial.getParameter<std::vector<double> >("barrel_radius");
    std::vector<double> barrel_length = trackerMaterial.getParameter<std::vector<double> >("barrel_length");
    
    assert(barrel_length.size() == barrel_radius.size() && barrel_length.size() ==  barrel_thickness.size());
    std::cout << "number of barrel layers = " << barrel_length.size() << std::endl;
    
    const Surface::PositionType thePosition(0.,0.,0.);
    const Surface::RotationType theRotation(1.,0.,0.,0.,1.,0.,0.,0.,1.);
    
    for(unsigned int i = 0, j = 0; i < barrel_length.size() || j < disk_z.size(); ){
      
      bool add_disk = false;
      if(i < barrel_length.size() && j < disk_z.size()){
	if(disk_outer_radius[j] < barrel_radius[i])
	  add_disk = true;
	else
	  add_disk = false;
      }
      else if (i < barrel_length.size() && !(j < disk_z.size()))
	add_disk = false;
      else if (!(i < barrel_length.size()) && j < disk_z.size())
	add_disk = true;
      else
	assert(0);
      
      if(add_disk){
	_mediumProperties.push_back(new MediumProperties(disk_thickness[j],0.0001));  
	
	const SimpleDiskBounds diskBounds(disk_inner_radius[j],disk_outer_radius[j],-0.0150,+0.0150);
	const Surface::PositionType positionType(0.,0.,disk_z[j]);
	
	unsigned layerNr = i+j;
	BoundDisk* theDisk = new BoundDisk(positionType,theRotation2,diskBounds);
	theDisk->setMediumProperties(*_mediumProperties[_mediumProperties.size() -1 ]);
	if ( theDisk->mediumProperties().radLen() > 0. ) 
	  _theCylinders.push_back(TrackerLayer(theDisk,true,layerNr,
					       std::vector<double>(),std::vector<double>(),
					       std::vector<double>()));
	else
	  delete theDisk;
	
	j++;
	
      }
      else 
	{
	  
	  // Create the nest of cylinders
	  
	  const SimpleCylinderBounds  cylBounds(  barrel_radius[i]-0.0150, barrel_radius[i]+0.0150, -barrel_length[i]/2, +barrel_length[i]/2);
	  
	  _mediumProperties.push_back(new MediumProperties(barrel_thickness[i],0.0001));  
	  
	  unsigned layerNr = i+j;
	  BoundCylinder* theCylinder = new BoundCylinder(thePosition,theRotation,cylBounds);
	  theCylinder->setMediumProperties(*_mediumProperties[_mediumProperties.size() -1 ]);
	  if ( theCylinder->mediumProperties().radLen() > 0. ) 
	    _theCylinders.push_back(TrackerLayer(theCylinder,false,layerNr,
						 std::vector<double>(),std::vector<double>(),
						 std::vector<double>()));

	  else
	    delete theCylinder;
	  
	  i++;
	}
    }
  }
  else {
    // Fraction of radiation length : had oc values to account 
    // for detectors, cables, support, ...
    // Note : the second argument is not used in FAMOS
    // Note : the first argument is tuned to reproduce the CMSIM material
    //        in terms or radiation length.
    
    
    // Thickness of all layers
    // Version of the material description
    version = trackerMaterial.getParameter<unsigned int>("TrackerMaterialVersion");
    // Beam Pipe
    beamPipeThickness = trackerMaterial.getParameter<std::vector<double> >("BeamPipeThickness");
    // Pixel Barrel Layers 1-3
    pxbThickness = trackerMaterial.getParameter<std::vector<double> >("PXBThickness");
    // Pixel Barrel services at the end of layers 1-3
    pxb1CablesThickness = trackerMaterial.getParameter<std::vector<double> >("PXB1CablesThickness");
    pxb2CablesThickness = trackerMaterial.getParameter<std::vector<double> >("PXB2CablesThickness");
    pxb3CablesThickness = trackerMaterial.getParameter<std::vector<double> >("PXB3CablesThickness");
    // Pixel Barrel outside cables
    pxbOutCables1Thickness = trackerMaterial.getParameter<std::vector<double> >("PXBOutCables1Thickness");
    pxbOutCables2Thickness = trackerMaterial.getParameter<std::vector<double> >("PXBOutCables2Thickness");
    // Pixel Disks 1-2
    pxdThickness = trackerMaterial.getParameter<std::vector<double> >("PXDThickness");
    // Pixel Endcap outside cables
    pxdOutCables1Thickness = trackerMaterial.getParameter<std::vector<double> >("PXDOutCables1Thickness");
    pxdOutCables2Thickness = trackerMaterial.getParameter<std::vector<double> >("PXDOutCables2Thickness");
    // Tracker Inner barrel layers 1-4
    tibLayer1Thickness = trackerMaterial.getParameter<std::vector<double> >("TIBLayer1Thickness");
    tibLayer2Thickness = trackerMaterial.getParameter<std::vector<double> >("TIBLayer2Thickness");
    tibLayer3Thickness = trackerMaterial.getParameter<std::vector<double> >("TIBLayer3Thickness");
    tibLayer4Thickness = trackerMaterial.getParameter<std::vector<double> >("TIBLayer4Thickness");
    // TIB outside services (endcap)
    tibOutCables1Thickness = trackerMaterial.getParameter<std::vector<double> >("TIBOutCables1Thickness");
    tibOutCables2Thickness = trackerMaterial.getParameter<std::vector<double> >("TIBOutCables2Thickness");
    // Tracker Inner disks layers 1-3
    tidLayer1Thickness = trackerMaterial.getParameter<std::vector<double> >("TIDLayer1Thickness");
    tidLayer2Thickness = trackerMaterial.getParameter<std::vector<double> >("TIDLayer2Thickness");
    tidLayer3Thickness = trackerMaterial.getParameter<std::vector<double> >("TIDLayer3Thickness");
    // TID outside wall (endcap)
    tidOutsideThickness = trackerMaterial.getParameter<std::vector<double> >("TIDOutsideThickness");
    // TOB inside wall (barrel)
    tobInsideThickness = trackerMaterial.getParameter<std::vector<double> >("TOBInsideThickness");
    // Tracker Outer barrel layers 1-6
    tobLayer1Thickness = trackerMaterial.getParameter<std::vector<double> >("TOBLayer1Thickness");
    tobLayer2Thickness = trackerMaterial.getParameter<std::vector<double> >("TOBLayer2Thickness");
    tobLayer3Thickness = trackerMaterial.getParameter<std::vector<double> >("TOBLayer3Thickness");
    tobLayer4Thickness = trackerMaterial.getParameter<std::vector<double> >("TOBLayer4Thickness");
    tobLayer5Thickness = trackerMaterial.getParameter<std::vector<double> >("TOBLayer5Thickness");
    tobLayer6Thickness = trackerMaterial.getParameter<std::vector<double> >("TOBLayer6Thickness");
    // TOB services (endcap)
    tobOutsideThickness = trackerMaterial.getParameter<std::vector<double> >("TOBOutsideThickness");
    // Tracker EndCap disks layers 1-9
    tecLayerThickness = trackerMaterial.getParameter<std::vector<double> >("TECLayerThickness");
    // TOB outside wall (barrel)
    barrelCablesThickness = trackerMaterial.getParameter<std::vector<double> >("BarrelCablesThickness");
    // TEC outside wall (endcap)
    endcapCables1Thickness = trackerMaterial.getParameter<std::vector<double> >("EndcapCables1Thickness");
    endcapCables2Thickness = trackerMaterial.getParameter<std::vector<double> >("EndcapCables2Thickness");
    
    // Position of dead material layers (cables, services, etc.)
    // Beam pipe
    beamPipeRadius = trackerMaterial.getParameter<std::vector<double> >("BeamPipeRadius");
    beamPipeLength = trackerMaterial.getParameter<std::vector<double> >("BeamPipeLength");
    // Cables and Services at the end of PIXB1,2,3 ("disk")
    pxb1CablesInnerRadius = trackerMaterial.getParameter<std::vector<double> >("PXB1CablesInnerRadius");
    pxb2CablesInnerRadius = trackerMaterial.getParameter<std::vector<double> >("PXB2CablesInnerRadius");
    pxb3CablesInnerRadius = trackerMaterial.getParameter<std::vector<double> >("PXB3CablesInnerRadius");
    // Pixel Barrel Outside walls and cables
    pxbOutCables1InnerRadius = trackerMaterial.getParameter<std::vector<double> >("PXBOutCables1InnerRadius");
    pxbOutCables1OuterRadius = trackerMaterial.getParameter<std::vector<double> >("PXBOutCables1OuterRadius");
    pxbOutCables1ZPosition = trackerMaterial.getParameter<std::vector<double> >("PXBOutCables1ZPosition");
    pxbOutCables2InnerRadius = trackerMaterial.getParameter<std::vector<double> >("PXBOutCables2InnerRadius");
    pxbOutCables2OuterRadius = trackerMaterial.getParameter<std::vector<double> >("PXBOutCables2OuterRadius");
    pxbOutCables2ZPosition = trackerMaterial.getParameter<std::vector<double> >("PXBOutCables2ZPosition");
    // Pixel Outside walls and cables (barrel and endcaps)
    pixelOutCablesRadius = trackerMaterial.getParameter<std::vector<double> >("PixelOutCablesRadius");
    pixelOutCablesLength = trackerMaterial.getParameter<std::vector<double> >("PixelOutCablesLength");
    pixelOutCablesInnerRadius = trackerMaterial.getParameter<std::vector<double> >("PixelOutCablesInnerRadius");
    pixelOutCablesOuterRadius = trackerMaterial.getParameter<std::vector<double> >("PixelOutCablesOuterRadius");
    pixelOutCablesZPosition = trackerMaterial.getParameter<std::vector<double> >("PixelOutCablesZPosition");
    // Tracker Inner Barrel Outside Cables and walls (endcap) 
    tibOutCables1InnerRadius = trackerMaterial.getParameter<std::vector<double> >("TIBOutCables1InnerRadius");
    tibOutCables1OuterRadius = trackerMaterial.getParameter<std::vector<double> >("TIBOutCables1OuterRadius");
    tibOutCables1ZPosition = trackerMaterial.getParameter<std::vector<double> >("TIBOutCables1ZPosition");
    tibOutCables2InnerRadius = trackerMaterial.getParameter<std::vector<double> >("TIBOutCables2InnerRadius");
    tibOutCables2OuterRadius = trackerMaterial.getParameter<std::vector<double> >("TIBOutCables2OuterRadius");
    tibOutCables2ZPosition = trackerMaterial.getParameter<std::vector<double> >("TIBOutCables2ZPosition");
    // Tracker outer barrel Inside wall (barrel)
    tobInCablesRadius = trackerMaterial.getParameter<std::vector<double> >("TOBInCablesRadius");
    tobInCablesLength = trackerMaterial.getParameter<std::vector<double> >("TOBInCablesLength");
    // Tracker Inner Disks Outside Cables and walls
    tidOutCablesInnerRadius = trackerMaterial.getParameter<std::vector<double> >("TIDOutCablesInnerRadius");
    tidOutCablesZPosition = trackerMaterial.getParameter<std::vector<double> >("TIDOutCablesZPosition");
    // Tracker Outer Barrel Outside Cables and walls (barrel and endcaps)
    tobOutCablesInnerRadius = trackerMaterial.getParameter<std::vector<double> >("TOBOutCablesInnerRadius");
    tobOutCablesOuterRadius = trackerMaterial.getParameter<std::vector<double> >("TOBOutCablesOuterRadius");
    tobOutCablesZPosition = trackerMaterial.getParameter<std::vector<double> >("TOBOutCablesZPosition");
    tobOutCablesRadius = trackerMaterial.getParameter<std::vector<double> >("TOBOutCablesRadius");
    tobOutCablesLength = trackerMaterial.getParameter<std::vector<double> >("TOBOutCablesLength");
    // Tracker Endcaps Outside Cables and walls
    tecOutCables1InnerRadius = trackerMaterial.getParameter<std::vector<double> >("TECOutCables1InnerRadius");
    tecOutCables1OuterRadius = trackerMaterial.getParameter<std::vector<double> >("TECOutCables1OuterRadius");
    tecOutCables1ZPosition = trackerMaterial.getParameter<std::vector<double> >("TECOutCables1ZPosition");
    tecOutCables2InnerRadius = trackerMaterial.getParameter<std::vector<double> >("TECOutCables2InnerRadius");
    tecOutCables2OuterRadius = trackerMaterial.getParameter<std::vector<double> >("TECOutCables2OuterRadius");
    tecOutCables2ZPosition = trackerMaterial.getParameter<std::vector<double> >("TECOutCables2ZPosition");
    
    // Fudge factors for tracker layer material inhomogeneities
    fudgeLayer = trackerMaterial.getParameter<std::vector<unsigned int> >("FudgeLayer");
    fudgeMin = trackerMaterial.getParameter<std::vector<double> >("FudgeMin");
    fudgeMax = trackerMaterial.getParameter<std::vector<double> >("FudgeMax");
    fudgeFactor = trackerMaterial.getParameter<std::vector<double> >("FudgeFactor");
    
    // The previous std::vector must have the same size!
    if ( fudgeLayer.size() != fudgeMin.size() ||  
	 fudgeLayer.size() != fudgeMax.size() ||  
	 fudgeLayer.size() != fudgeFactor.size() ) {
      throw cms::Exception("FastSimulation/TrackerInteractionGeometry ") 
	<< " WARNING with fudge factors !  You have " << fudgeLayer.size() 
	<< " layers, but " 
	<< fudgeMin.size() << " min values, "
	<< fudgeMax.size() << " max values and "
	<< fudgeFactor.size() << " fudge factor values!"
	<< std::endl
	<< "Please make enter the same number of inputs " 
	<< "in FastSimulation/TrackerInteractionGeometry/data/TrackerMaterial.cfi"
	<< std::endl;
    }
    
    // The Beam pipe
    _theMPBeamPipe = new MediumProperties(beamPipeThickness[version],0.0001);  
    // The pixel barrel layers
    _theMPPixelBarrel = new MediumProperties(pxbThickness[version],0.0001);  
    // Pixel Barrel services at the end of layers 1-3
    _theMPPixelOutside1 = new MediumProperties(pxb1CablesThickness[version],0.0001);  
    _theMPPixelOutside2 = new MediumProperties(pxb2CablesThickness[version],0.0001);  
    _theMPPixelOutside3 = new MediumProperties(pxb3CablesThickness[version],0.0001);  
    // Pixel Barrel outside cables
    _theMPPixelOutside4 = new MediumProperties(pxbOutCables1Thickness[version],0.0001);  
    _theMPPixelOutside  = new MediumProperties(pxbOutCables2Thickness[version],0.0001);  
    // The pixel endcap disks
    _theMPPixelEndcap = new MediumProperties(pxdThickness[version],0.0001);  
    // Pixel Endcap outside cables
    _theMPPixelOutside5 = new MediumProperties(pxdOutCables1Thickness[version],0.0001);  
    _theMPPixelOutside6 = new MediumProperties(pxdOutCables2Thickness[version],0.0001);  
    // The tracker inner barrel layers 1-4
    _theMPTIB1 = new MediumProperties(tibLayer1Thickness[version],0.0001);  
    _theMPTIB2 = new MediumProperties(tibLayer2Thickness[version],0.0001);  
    _theMPTIB3 = new MediumProperties(tibLayer3Thickness[version],0.0001);  
    _theMPTIB4 = new MediumProperties(tibLayer4Thickness[version],0.0001);  
    // TIB outside services (endcap)
    _theMPTIBEOutside1 = new MediumProperties(tibOutCables1Thickness[version],0.0001);  
    _theMPTIBEOutside2 = new MediumProperties(tibOutCables2Thickness[version],0.0001);  
    // The tracker inner disks 1-3
    _theMPInner1 = new MediumProperties(tidLayer1Thickness[version],0.0001);  
    _theMPInner2 = new MediumProperties(tidLayer2Thickness[version],0.0001);  
    _theMPInner3 = new MediumProperties(tidLayer3Thickness[version],0.0001);  
    // TID outside wall (endcap)
    _theMPTIDEOutside = new MediumProperties(tidOutsideThickness[version],0.0001);  
    // TOB inside wall (barrel)
    _theMPTOBBInside = new MediumProperties(tobInsideThickness[version],0.0001);  
    // The tracker outer barrel layers 1-6
    _theMPTOB1 = new MediumProperties(tobLayer1Thickness[version],0.0001);  
    _theMPTOB2 = new MediumProperties(tobLayer2Thickness[version],0.0001);  
    _theMPTOB3 = new MediumProperties(tobLayer3Thickness[version],0.0001);  
    _theMPTOB4 = new MediumProperties(tobLayer4Thickness[version],0.0001);  
    _theMPTOB5 = new MediumProperties(tobLayer5Thickness[version],0.0001);  
    _theMPTOB6 = new MediumProperties(tobLayer6Thickness[version],0.0001);  
    // TOB services (endcap)
    _theMPTOBEOutside = new MediumProperties(tobOutsideThickness[version],0.0001);  
    // The tracker endcap disks 1-9
    _theMPEndcap = new MediumProperties(tecLayerThickness[version],0.0001);  
    // TOB outside wall (barrel)
    _theMPBarrelOutside = new MediumProperties(barrelCablesThickness[version],0.0001);  
    // TEC outside wall (endcap)
    _theMPEndcapOutside = new MediumProperties(endcapCables1Thickness[version],0.0001);  
    _theMPEndcapOutside2 = new MediumProperties(endcapCables2Thickness[version],0.0001);  
    
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
    
    // Create the nest of cylinders
    const Surface::PositionType thePosition(0.,0.,0.);
    const Surface::RotationType theRotation(1.,0.,0.,0.,1.,0.,0.,0.,1.);
    // Beam Pipe
    //  const SimpleCylinderBounds  PIPE( 0.997,   1.003,  -300., 300.);
    const SimpleCylinderBounds  PIPE( beamPipeRadius[version]-0.003, beamPipeRadius[version]+0.003,  
				      -beamPipeLength[version],       beamPipeLength[version]);
    
    // Take the active layer position from the Tracker Reco Geometry
    // Pixel barrel
    std::vector< BarrelDetLayer*>::const_iterator bl = barrelLayers.begin();
    double maxLength = (**bl).specificSurface().bounds().length()/2.+1.7;
    double maxRadius = (**bl).specificSurface().radius()+0.01;
    // First pixel barrel layer: r=4.41058, l=53.38
    const SimpleCylinderBounds  PIXB1( maxRadius-0.005, maxRadius+0.005, -maxLength, +maxLength);
    // "Cables" 
    const SimpleDiskBounds PIXBOut1(pxb1CablesInnerRadius[version],maxRadius+0.01,-0.5,0.5);
    const Surface::PositionType PPIXBOut1(0.0,0.0,maxLength);
    
    // Second pixel barrel layer: r=7.30732, l=53.38
    ++bl;
    maxLength = std::max( (**bl).specificSurface().bounds().length()/2.+1.7, maxLength+0.000 );
    maxRadius = (**bl).specificSurface().radius();
    const SimpleCylinderBounds  PIXB2( maxRadius-0.005, maxRadius+0.005, -maxLength, +maxLength);
    
    // "Cables"
    const SimpleDiskBounds PIXBOut2(pxb2CablesInnerRadius[version],maxRadius+0.005,-0.5,0.5);
    const Surface::PositionType PPIXBOut2(0.0,0.0,maxLength);
    
    // More cables
    ++bl;
    maxRadius = (**bl).specificSurface().radius();
    const SimpleDiskBounds PIXBOut3(pxb3CablesInnerRadius[version],maxRadius,-0.5,0.5);
    const Surface::PositionType PPIXBOut3(0.0,0.0,maxLength);
    
    // Third pixel barrel layer: r=10.1726, l=53.38
    maxLength = std::max( (**bl).specificSurface().bounds().length()/2.+1.7, maxLength+0.000 );
    const SimpleCylinderBounds  PIXB3( maxRadius-0.005, maxRadius+0.005, -maxLength, +maxLength);
    
    // Pixel Barrel Outside walls and cables
    const SimpleDiskBounds PIXBOut4( pxbOutCables1InnerRadius[version],pxbOutCables1OuterRadius[version],-0.5,0.5);
    const Surface::PositionType PPIXBOut4(0.0,0.0,pxbOutCables1ZPosition[version]);
    
    const SimpleDiskBounds PIXBOut(pxbOutCables2InnerRadius[version],pxbOutCables2OuterRadius[version],-0.5,0.5);
    const Surface::PositionType PPIXBOut(0.0,0.0,pxbOutCables2ZPosition[version]);
    
    const SimpleCylinderBounds  PIXBOut5( pixelOutCablesRadius[version]-0.1, pixelOutCablesRadius[version]+0.1, 
					  -pixelOutCablesLength[version],     pixelOutCablesLength[version]);
    
    const SimpleDiskBounds PIXBOut6(pixelOutCablesInnerRadius[version],pixelOutCablesOuterRadius[version],-0.5,0.5);
    const Surface::PositionType PPIXBOut6(0.0,0.0,pixelOutCablesZPosition[version]);
    
    
    // Tracker Inner Barrel : thin detectors (300 microns)
    // First TIB layer: r=25.6786, l=130.04
    ++bl;
    maxRadius = (**bl).specificSurface().radius();
    maxLength = (**bl).specificSurface().bounds().length()/2.;
    const SimpleCylinderBounds  TIB1( maxRadius-0.0150, maxRadius+0.0150, -maxLength, +maxLength);
    // Second TIB layer: r=34.0341, l=131.999
    ++bl;
    maxRadius = (**bl).specificSurface().radius();
    maxLength = std::max( (**bl).specificSurface().bounds().length()/2., maxLength+0.000 );
    const SimpleCylinderBounds  TIB2( maxRadius-0.0150, maxRadius+0.0150, -maxLength, +maxLength);
    // Third TIB layer: r=41.9599, l=131.628  !!!! Needs to be larger than TIB2
    ++bl;
    maxRadius = (**bl).specificSurface().radius();
    maxLength = std::max( (**bl).specificSurface().bounds().length()/2., maxLength+0.000 );
    const SimpleCylinderBounds  TIB3( maxRadius-0.0150, maxRadius+0.0150, -maxLength, +maxLength);
    // Fourth TIB layer: r=49.8924, l=132.78
    ++bl;
    maxRadius = (**bl).specificSurface().radius();
    maxLength = std::max( (**bl).specificSurface().bounds().length()/2., maxLength+0.000 );
    const SimpleCylinderBounds  TIB4( maxRadius-0.0150, maxRadius+0.0150, -maxLength, +maxLength);
    
    // Inner Barrel Cylinder & Ends : Cables and walls
    const SimpleDiskBounds TIBEOut(tibOutCables1InnerRadius[version],tibOutCables1OuterRadius[version],-0.05,0.05);
    const Surface::PositionType PTIBEOut(0.0,0.0,tibOutCables1ZPosition[version]);
    
    const SimpleDiskBounds TIBEOut2(tibOutCables2InnerRadius[version],tibOutCables2OuterRadius[version],-0.05,0.05);
    const Surface::PositionType PTIBEOut2(0.0,0.0,tibOutCables2ZPosition[version]);
    
    // Inner Tracker / Outer Barrel Wall
    const SimpleCylinderBounds  TOBCIn ( tobInCablesRadius[version]-0.5, tobInCablesRadius[version]+0.5,
					 -tobInCablesLength[version],     tobInCablesLength[version]);
    
    // First TOB layer: r=60.7671, l=216.576
    ++bl;
    maxRadius = (**bl).specificSurface().radius();
    maxLength = (**bl).specificSurface().bounds().length()/2.+0.0;
    const SimpleCylinderBounds  TOB1( maxRadius-0.0150, maxRadius+0.0150, -maxLength, +maxLength);
    // Second TOB layer: r=69.3966, l=216.576
    ++bl;
    maxRadius = (**bl).specificSurface().radius();
    maxLength = std::max( (**bl).specificSurface().bounds().length()/2.+0.0, maxLength+0.000 );
    const SimpleCylinderBounds  TOB2( maxRadius-0.0150, maxRadius+0.0150, -maxLength, +maxLength);
    // Third TOB layer: r=78.0686, l=216.576
    ++bl;
    maxRadius = (**bl).specificSurface().radius();
    maxLength = std::max( (**bl).specificSurface().bounds().length()/2.+0.0, maxLength+0.000 );
    const SimpleCylinderBounds  TOB3( maxRadius-0.0150, maxRadius+0.0150, -maxLength, +maxLength);
    // Fourth TOB layer: r=86.8618, l=216.576
    ++bl;
    maxRadius = (**bl).specificSurface().radius();
    maxLength = std::max( (**bl).specificSurface().bounds().length()/2.+0.0, maxLength+0.000 );
    const SimpleCylinderBounds  TOB4( maxRadius-0.0150, maxRadius+0.0150, -maxLength, +maxLength);
    // Fifth TOB layer: r=96.5557, l=216.576
    ++bl;
    maxRadius = (**bl).specificSurface().radius();
    maxLength = std::max( (**bl).specificSurface().bounds().length()/2.+0.0, maxLength+0.000 );
    const SimpleCylinderBounds  TOB5( maxRadius-0.0150, maxRadius+0.0150, -maxLength, +maxLength);
    // Sixth TOB layer: r=108.05, l=216.576
    ++bl;
    maxRadius = (**bl).specificSurface().radius();
    maxLength = std::max( (**bl).specificSurface().bounds().length()/2.+0.0, maxLength+0.000 );
    const SimpleCylinderBounds  TOB6( maxRadius-0.0150, maxRadius+0.0150, -maxLength, +maxLength);
    
    const SimpleDiskBounds TOBEOut(tobOutCablesInnerRadius[version],tobOutCablesOuterRadius[version],-0.5,0.5);
    const Surface::PositionType PTOBEOut(0.0,0.0,tobOutCablesZPosition[version]);
    
    const Surface::RotationType theRotation2(1.,0.,0.,0.,1.,0.,0.,0.,1.);
    
    // Outside : Barrel
    const SimpleCylinderBounds  TBOut ( tobOutCablesRadius[version]-0.5, tobOutCablesRadius[version]+0.5,
					-tobOutCablesLength[version],     tobOutCablesLength[version]);
    
    // And now the disks...
    std::vector< ForwardDetLayer*>::const_iterator fl = posForwardLayers.begin();
    
    // Pixel disks 
    // First Pixel disk: Z pos 35.5 radii 5.42078, 16.0756
    double innerRadius = (**fl).specificSurface().innerRadius()-1.0;
    double outerRadius = (**fl).specificSurface().outerRadius()+2.0;
    const SimpleDiskBounds PIXD1(innerRadius, outerRadius,-0.0150,+0.0150);
    const Surface::PositionType PPIXD1(0.0,0.0,(**fl).surface().position().z()); 
    // Second Pixel disk: Z pos 48.5 radii 5.42078, 16.0756
    ++fl;
    innerRadius = (**fl).specificSurface().innerRadius()-1.0;
    outerRadius = std::max( (**fl).specificSurface().outerRadius()+2.0, outerRadius+0.000 );
    const SimpleDiskBounds PIXD2(innerRadius, outerRadius,-0.0150,+0.0150);
    const Surface::PositionType PPIXD2(0.0,0.0,(**fl).surface().position().z()); 
    
    // Tracker Inner disks (add 3 cm for the outer radius to simulate cables, 
    // and remove 1cm to inner radius to allow for some extrapolation margin)
    // First TID : Z pos 78.445 radii 23.14, 50.4337
    ++fl;
    innerRadius = (**fl).specificSurface().innerRadius()-0.5;
    outerRadius = (**fl).specificSurface().outerRadius()+3.5;
    const SimpleDiskBounds TID1(innerRadius,outerRadius,-0.0150,+0.0150);
    const Surface::PositionType PTID1(0.,0.,(**fl).surface().position().z()); 
    // Second TID : Z pos 90.445 radii 23.14, 50.4337
    ++fl;
    innerRadius = (**fl).specificSurface().innerRadius()-0.5;
    outerRadius = std::max( (**fl).specificSurface().outerRadius()+3.5, outerRadius+0.000);
    const SimpleDiskBounds TID2(innerRadius,outerRadius,-0.0150,+0.0150);
    const Surface::PositionType PTID2(0.,0.,(**fl).surface().position().z()); 
    // Third TID : Z pos 105.445 radii 23.14, 50.4337
    ++fl;
    innerRadius = (**fl).specificSurface().innerRadius()-0.5;
    outerRadius = std::max( (**fl).specificSurface().outerRadius()+3.5, outerRadius+0.000);
    const SimpleDiskBounds TID3(innerRadius,outerRadius,-0.0150,+0.0150);
    const Surface::PositionType PTID3(0.,0.,(**fl).surface().position().z()); 
    
    // TID Wall and cables
    const SimpleDiskBounds TIDEOut(tidOutCablesInnerRadius[version],outerRadius+1.0,-0.5,0.5);
    const Surface::PositionType PTIDEOut(0.0,0.0,tidOutCablesZPosition[version]);
    
    
    // Tracker Endcaps : Add 11 cm to outer radius to correct for a bug, remove
    // 5cm to the inner radius (TEC7,8,9) to correct for a simular bug, and
    // remove other 2cm to inner radius to allow for some extrapolation margin
    // First TEC: Z pos 131.892 radii 23.3749, 99.1967
    ++fl;
    innerRadius = (**fl).specificSurface().innerRadius()-1.5;
    outerRadius = (**fl).specificSurface().outerRadius()+2.0;
    const SimpleDiskBounds TEC1(innerRadius,outerRadius,-0.0150,+0.0150);
    const Surface::PositionType PTEC1(0.,0,(**fl).surface().position().z()); 
    // Second TEC: Z pos 145.892 radii 23.3749, 99.1967
    ++fl;
    innerRadius = (**fl).specificSurface().innerRadius()-1.5;
    outerRadius = std::max( (**fl).specificSurface().outerRadius()+2.0, outerRadius+0.000 );
    const SimpleDiskBounds TEC2(innerRadius,outerRadius,-0.0150,+0.0150);
    const Surface::PositionType PTEC2(0.,0.,(**fl).surface().position().z());
    // Third TEC: Z pos 159.892 radii 23.3749, 99.1967
    ++fl;
    innerRadius = (**fl).specificSurface().innerRadius()-1.5;
    outerRadius = std::max( (**fl).specificSurface().outerRadius()+2.0, outerRadius+0.000 );
    const SimpleDiskBounds TEC3(innerRadius,outerRadius,-0.0150,+0.0150);
    const Surface::PositionType PTEC3(0.,0.,(**fl).surface().position().z());
    // Fourth TEC: Z pos 173.892 radii 32.1263, 99.1967
    ++fl;
    innerRadius = (**fl).specificSurface().innerRadius()-2.5;
    outerRadius = std::max( (**fl).specificSurface().outerRadius()+2.0, outerRadius+0.000 );
    const SimpleDiskBounds TEC4(innerRadius,outerRadius,-0.0150,+0.0150);
    const Surface::PositionType PTEC4(0.,0.,(**fl).surface().position().z());
    // Fifth TEC: Z pos 187.892 radii 32.1263, 99.1967
    ++fl;
    innerRadius = (**fl).specificSurface().innerRadius()-2.5;
    outerRadius = std::max( (**fl).specificSurface().outerRadius()+2.0, outerRadius+0.000 );
    const SimpleDiskBounds TEC5(innerRadius,outerRadius,-0.0150,+0.0150);
    const Surface::PositionType PTEC5(0.,0.,(**fl).surface().position().z());
    // Sixth TEC: Z pos 205.392 radii 32.1263, 99.1967
    ++fl;
    innerRadius = (**fl).specificSurface().innerRadius()-2.5;
    outerRadius = std::max( (**fl).specificSurface().outerRadius()+2.0, outerRadius+0.000 );
    const SimpleDiskBounds TEC6(innerRadius,outerRadius,-0.0150,+0.0150);
    const Surface::PositionType PTEC6(0.,0.,(**fl).surface().position().z());
    // Seventh TEC: Z pos 224.121 radii 44.7432, 99.1967
    ++fl;
    innerRadius = (**fl).specificSurface().innerRadius()-9.5;
    outerRadius = std::max( (**fl).specificSurface().outerRadius()+2.0, outerRadius+0.000 );
    const SimpleDiskBounds TEC7(innerRadius,outerRadius,-0.0150,+0.0150);
    const Surface::PositionType PTEC7(0.,0.,(**fl).surface().position().z());
    // Eighth TEC: Z pos 244.621 radii 44.7432, 99.1967
    ++fl;
    innerRadius = (**fl).specificSurface().innerRadius()-9.5;
    outerRadius = std::max( (**fl).specificSurface().outerRadius()+2.0, outerRadius+0.000 );
    const SimpleDiskBounds TEC8(innerRadius,outerRadius,-0.0150,+0.0150);
    const Surface::PositionType PTEC8(0.,0.,(**fl).surface().position().z());
    // Nineth TEC: Z pos 266.121 radii 56.1781, 99.1967
    ++fl;
    innerRadius = (**fl).specificSurface().innerRadius()-20.5;
    outerRadius = std::max( (**fl).specificSurface().outerRadius()+2.0, outerRadius+0.000 );
    const SimpleDiskBounds TEC9(innerRadius,outerRadius,-0.0150,+0.0150);
    const Surface::PositionType PTEC9(0.,0.,(**fl).surface().position().z());
    
    // Outside : Endcap
    const SimpleDiskBounds TEOut(tecOutCables1InnerRadius[version],tecOutCables1OuterRadius[version],-0.5,0.5);
    const Surface::PositionType PTEOut(0.0,0.0,tecOutCables1ZPosition[version]);
    
    const SimpleDiskBounds TEOut2(tecOutCables2InnerRadius[version],tecOutCables2OuterRadius[version],-0.5,0.5);
    const Surface::PositionType PTEOut2(0.0,0.0,tecOutCables2ZPosition[version]);
    
    // The ordering of disks and cylinders is essential here
    // (from inside to outside)
    // Do not change it thoughtlessly.
    
    
    // Beam Pipe
    
    unsigned layerNr = 100;
    theCylinder = new BoundCylinder(thePosition,theRotation,PIPE);
    theCylinder->setMediumProperties(*_theMPBeamPipe);
    if ( theCylinder->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theCylinder,false,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theCylinder;
    
    // Pixels 
    
    layerNr = TrackerInteractionGeometry::PXB+1;
    theCylinder = new BoundCylinder(thePosition,theRotation,PIXB1);
    theCylinder->setMediumProperties(*_theMPPixelBarrel);
    if ( theCylinder->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theCylinder,false,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theCylinder;
    
    layerNr = 101;
    theDisk = new BoundDisk(PPIXBOut1,theRotation2,PIXBOut1);
    theDisk->setMediumProperties(*_theMPPixelOutside1);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,true,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    layerNr = TrackerInteractionGeometry::PXB+2;
    theCylinder = new BoundCylinder(thePosition,theRotation,PIXB2);
    theCylinder->setMediumProperties(*_theMPPixelBarrel);
    if ( theCylinder->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theCylinder,false,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theCylinder;
    
    layerNr = 102;
    theDisk = new BoundDisk(PPIXBOut2,theRotation2,PIXBOut2);
    theDisk->setMediumProperties(*_theMPPixelOutside2);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,true,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    layerNr = 103;
    theDisk = new BoundDisk(PPIXBOut3,theRotation2,PIXBOut3);
    theDisk->setMediumProperties(*_theMPPixelOutside3);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,true,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    layerNr = TrackerInteractionGeometry::PXB+3;
    theCylinder = new BoundCylinder(thePosition,theRotation,PIXB3);
    theCylinder->setMediumProperties(*_theMPPixelBarrel);
    if ( theCylinder->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theCylinder,false,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theCylinder;
    
    layerNr = 104;
    theDisk = new BoundDisk(PPIXBOut4,theRotation2,PIXBOut4);
    theDisk->setMediumProperties(*_theMPPixelOutside4);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,true,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    layerNr = 105;
    theDisk = new BoundDisk(PPIXBOut,theRotation2,PIXBOut);
    theDisk->setMediumProperties(*_theMPPixelOutside);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,true,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    layerNr = TrackerInteractionGeometry::PXD+1;
    theDisk = new BoundDisk(PPIXD1,theRotation2,PIXD1);
    theDisk->setMediumProperties(*_theMPPixelEndcap);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,true,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    layerNr = TrackerInteractionGeometry::PXD+2;
    theDisk = new BoundDisk(PPIXD2,theRotation2,PIXD2);
    theDisk->setMediumProperties(*_theMPPixelEndcap);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,true,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    layerNr = 106;
    theCylinder = new BoundCylinder(thePosition,theRotation,PIXBOut5);
    theCylinder->setMediumProperties(*_theMPPixelOutside5);
    if ( theCylinder->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theCylinder,false,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theCylinder;
    
    layerNr = 107;
    theDisk = new BoundDisk(PPIXBOut6,theRotation2,PIXBOut6);
    theDisk->setMediumProperties(*_theMPPixelOutside6);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,true,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    // Inner Barrel 
    
    layerNr = TrackerInteractionGeometry::TIB+1;
    theCylinder = new BoundCylinder(thePosition,theRotation,TIB1);
    theCylinder->setMediumProperties(*_theMPTIB1);
    if ( theCylinder->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theCylinder,false,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theCylinder;
    
    layerNr = TrackerInteractionGeometry::TIB+2;
    theCylinder = new BoundCylinder(thePosition,theRotation,TIB2);
    theCylinder->setMediumProperties(*_theMPTIB2);
    if ( theCylinder->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theCylinder,false,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theCylinder;
    
    layerNr = TrackerInteractionGeometry::TIB+3;
    theCylinder = new BoundCylinder(thePosition,theRotation,TIB3);
    theCylinder->setMediumProperties(*_theMPTIB3);
    if ( theCylinder->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theCylinder,false,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theCylinder;
    
    layerNr = TrackerInteractionGeometry::TIB+4;
    theCylinder = new BoundCylinder(thePosition,theRotation,TIB4);
    theCylinder->setMediumProperties(*_theMPTIB4);
    if ( theCylinder->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theCylinder,false,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theCylinder;
    
    layerNr = 108;
    theDisk = new BoundDisk(PTIBEOut,theRotation2,TIBEOut);
    theDisk->setMediumProperties(*_theMPTIBEOutside1);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,true,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    layerNr = 109;
    theDisk = new BoundDisk(PTIBEOut2,theRotation2,TIBEOut2);
    theDisk->setMediumProperties(*_theMPTIBEOutside2);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,true,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    // Inner Endcaps
    
    layerNr = TrackerInteractionGeometry::TID+1;
    theDisk = new BoundDisk(PTID1,theRotation2,TID1);
    theDisk->setMediumProperties(*_theMPInner1);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    layerNr = TrackerInteractionGeometry::TID+2;
    theDisk = new BoundDisk(PTID2,theRotation2,TID2);
    theDisk->setMediumProperties(*_theMPInner2);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    
    else
      delete theDisk;
    
    layerNr = TrackerInteractionGeometry::TID+3;
    theDisk = new BoundDisk(PTID3,theRotation2,TID3);
    theDisk->setMediumProperties(*_theMPInner3);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,12,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    layerNr = 110;
    theDisk = new BoundDisk(PTIDEOut,theRotation2,TIDEOut);
    theDisk->setMediumProperties(*_theMPTIDEOutside);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,true,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    
    // Outer Barrel 
    
    layerNr = 111;
    theCylinder = new BoundCylinder(thePosition,theRotation,TOBCIn);
    theCylinder->setMediumProperties(*_theMPTOBBInside);
    if ( theCylinder->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theCylinder,false,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theCylinder;
    
    layerNr = TrackerInteractionGeometry::TOB+1;
    theCylinder = new BoundCylinder(thePosition,theRotation,TOB1);
    theCylinder->setMediumProperties(*_theMPTOB1);
    if ( theCylinder->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theCylinder,false,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theCylinder;
    
    layerNr = TrackerInteractionGeometry::TOB+2;
    theCylinder = new BoundCylinder(thePosition,theRotation,TOB2);
    theCylinder->setMediumProperties(*_theMPTOB2);
    if ( theCylinder->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theCylinder,false,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theCylinder;
    
    layerNr = TrackerInteractionGeometry::TOB+3;
    theCylinder = new BoundCylinder(thePosition,theRotation,TOB3);
    theCylinder->setMediumProperties(*_theMPTOB3);
    if ( theCylinder->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theCylinder,false,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theCylinder;
    
    layerNr = TrackerInteractionGeometry::TOB+4;
    theCylinder = new BoundCylinder(thePosition,theRotation,TOB4);
    theCylinder->setMediumProperties(*_theMPTOB4);
    if ( theCylinder->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theCylinder,false,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theCylinder;
    
    layerNr = TrackerInteractionGeometry::TOB+5;
    theCylinder = new BoundCylinder(thePosition,theRotation,TOB5);
    theCylinder->setMediumProperties(*_theMPTOB5);
    if ( theCylinder->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theCylinder,false,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theCylinder;
    
    layerNr = TrackerInteractionGeometry::TOB+6;
    theCylinder = new BoundCylinder(thePosition,theRotation,TOB6);
    theCylinder->setMediumProperties(*_theMPTOB6);
    if ( theCylinder->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theCylinder,false,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theCylinder;
    
    layerNr = 112;
    theDisk = new BoundDisk(PTOBEOut,theRotation2,TOBEOut);
    theDisk->setMediumProperties(*_theMPTOBEOutside);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,true,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    // Outer Endcaps
    
    layerNr = TrackerInteractionGeometry::TEC+1;
    theDisk = new BoundDisk(PTEC1,theRotation2,TEC1);
    theDisk->setMediumProperties(*_theMPEndcap);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    layerNr = TrackerInteractionGeometry::TEC+2;
    theDisk = new BoundDisk(PTEC2,theRotation2,TEC2);
    theDisk->setMediumProperties(*_theMPEndcap);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    layerNr = TrackerInteractionGeometry::TEC+3;
    theDisk = new BoundDisk(PTEC3,theRotation2,TEC3);
    theDisk->setMediumProperties(*_theMPEndcap);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    layerNr = TrackerInteractionGeometry::TEC+4;
    theDisk = new BoundDisk(PTEC4,theRotation2,TEC4);
    theDisk->setMediumProperties(*_theMPEndcap);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    layerNr = TrackerInteractionGeometry::TEC+5;
    theDisk = new BoundDisk(PTEC5,theRotation2,TEC5);
    theDisk->setMediumProperties(*_theMPEndcap);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    layerNr = TrackerInteractionGeometry::TEC+6;
    theDisk = new BoundDisk(PTEC6,theRotation2,TEC6);
    theDisk->setMediumProperties(*_theMPEndcap);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    layerNr = TrackerInteractionGeometry::TEC+7;
    theDisk = new BoundDisk(PTEC7,theRotation2,TEC7);
    theDisk->setMediumProperties(*_theMPEndcap);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    layerNr = TrackerInteractionGeometry::TEC+8;
    theDisk = new BoundDisk(PTEC8,theRotation2,TEC8);
    theDisk->setMediumProperties(*_theMPEndcap);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    layerNr = TrackerInteractionGeometry::TEC+9;
    theDisk = new BoundDisk(PTEC9,theRotation2,TEC9);
    theDisk->setMediumProperties(*_theMPEndcap);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    
    // Tracker Outside
    
    layerNr = 113;
    theCylinder = new BoundCylinder(thePosition,theRotation,TBOut);
    theCylinder->setMediumProperties(*_theMPBarrelOutside);
    if ( theCylinder->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theCylinder,false,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theCylinder;
    
    layerNr = 114;
    theDisk = new BoundDisk(PTEOut,theRotation2,TEOut);
    theDisk->setMediumProperties(*_theMPEndcapOutside);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,true,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
    layerNr = 115;
    theDisk = new BoundDisk(PTEOut2,theRotation2,TEOut2);
    theDisk->setMediumProperties(*_theMPEndcapOutside2);
    if ( theDisk->mediumProperties().radLen() > 0. ) 
      _theCylinders.push_back(TrackerLayer(theDisk,true,layerNr,
					   minDim(layerNr),maxDim(layerNr),
					   fudgeFactors(layerNr)));
    else
      delete theDisk;
    
  }
  

  // Check overall compatibility of cylinder dimensions
  // (must be nested cylinders)
  // Throw an exception if the test fails
  double zin, rin;
  double zout, rout;
  unsigned nCyl=0;
  std::list<TrackerLayer>::const_iterator cyliterOut=cylinderBegin();
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
	<< " has dimensions smaller than previous cylinder : " << std::endl
	<< " zout/zin = " << zout << " " << zin << std::endl
	<< " rout/rin = " << rout << " " << rin << std::endl;
    } else {
      /*
      std::cout << " Cylinder number " << nCyl 
		<< " (Active Layer Number = " <<  cyliterOut->layerNumber() 
		<< " Forward ? " <<  cyliterOut->forward() << " ) "
		<< " has dimensions of : " 
		<< " zout = " << zout << "; " 
		<< " rout = " << rout << std::endl;
      */
    }

    // Go to the next cylinder
    cyliterOut++;
    // Inner cylinder becomes outer cylinder
    zin = zout;
    rin = rout;
    // End test
  } 
    
}

std::vector<double>
TrackerInteractionGeometry::minDim(unsigned layerNr) { 
  std::vector<double> min;
  for ( unsigned iLayer=0; iLayer<fudgeFactor.size(); ++iLayer ) {   
    if ( layerNr != fudgeLayer[iLayer] ) continue;
    min.push_back(fudgeMin[iLayer]);
  }
  return min;
}

std::vector<double>
TrackerInteractionGeometry::maxDim(unsigned layerNr) { 
  std::vector<double> max;
  for ( unsigned iLayer=0; iLayer<fudgeFactor.size(); ++iLayer ) {   
    if ( layerNr != fudgeLayer[iLayer] ) continue;
    max.push_back(fudgeMax[iLayer]);
  }
  return max;
}

std::vector<double>
TrackerInteractionGeometry::fudgeFactors(unsigned layerNr) { 
  std::vector<double> fudge;
  for ( unsigned iLayer=0; iLayer<fudgeFactor.size(); ++iLayer ) {   
    if ( layerNr != fudgeLayer[iLayer] ) continue;
    fudge.push_back(fudgeFactor[iLayer]);
  }
  return fudge;
}

TrackerInteractionGeometry::~TrackerInteractionGeometry()
{
  _theCylinders.clear();
  //  _theRings.clear();

  if(use_hardcoded){
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
    delete _theMPInner1;
    delete _theMPInner2;
    delete _theMPInner3;
    // The tracker endcap disks
    delete _theMPEndcap;
    // Various cable thicknesses 
    delete _theMPTOBBInside;
    delete _theMPTIBEOutside1;
    delete _theMPTIBEOutside2;
    delete _theMPTIDEOutside;
    delete _theMPTOBEOutside;
    delete _theMPBarrelOutside;
    delete _theMPEndcapOutside;
    delete _theMPEndcapOutside2;
    
  }
  else {
    for(unsigned int i = 0; i < _mediumProperties.size(); i++){
      delete _mediumProperties[i];
    }
  }

}
