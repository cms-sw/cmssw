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

    std::vector<double> disk_thickness;
    std::vector<double> disk_inner_radius;
    std::vector<double> disk_outer_radius;
    std::vector<double> disk_z;
    
    std::vector<double> barrel_thickness;
    std::vector<double> barrel_radius;
    std::vector<double> barrel_length;
    
    for(unsigned int i = 0; i < barrelLayers.size(); i++){
      std::cout << "defining barrellayer: " << i << std::endl;
      std::cout << "   length,radius= " <<
	barrelLayers[i]->specificSurface().bounds().length() << " " <<
	barrelLayers[i]->specificSurface().radius() << " " << std::endl;
      
      barrel_length.push_back( barrelLayers[i]->specificSurface().bounds().length());
      barrel_radius.push_back( barrelLayers[i]->specificSurface().radius());
      barrel_thickness.push_back(0.05);
      
    }
    
    for(unsigned int i = 0; i < posForwardLayers.size(); i++){
      
      std::cout << "defining forwardlayer: " << i << std::endl;
      std::cout << "  z,inner R, outer R = " <<     
	posForwardLayers[i]->surface().position().z() << " " <<
	posForwardLayers[i]->specificSurface().innerRadius() << " " <<
	posForwardLayers[i]->specificSurface().outerRadius() << std::endl;
      
      disk_z.push_back(posForwardLayers[i]->surface().position().z());
      disk_inner_radius.push_back(posForwardLayers[i]->specificSurface().innerRadius());
      disk_outer_radius.push_back(posForwardLayers[i]->specificSurface().outerRadius());
      disk_thickness.push_back(0.05);
      
    }

    //sort the forward layers according to z
    for(unsigned int i = 0; i < disk_z.size(); i++){
      
      float min_z = -1;
      int min_z_index = -1;

      //find the minimum out of the remaining elements (elements i to disk_z.size() - 1)
      for(unsigned int j = i; j < disk_z.size(); j++){
	if(j == i){
	  min_z = disk_z[j];
	  min_z_index = j;
	}
	else if (disk_z[j] < min_z){
	  min_z = disk_z[j];
	  min_z_index = j;
	}
      }

      //swap the element at min_z_index with the one at index i
      float temp_z = disk_z[i];
      float temp_inner_radius = disk_inner_radius[i];
      float temp_outer_radius = disk_outer_radius[i];
      float temp_thickness = disk_thickness[i];

      disk_z[i] = disk_z[min_z_index];
      disk_inner_radius[i] = disk_inner_radius[min_z_index];
      disk_outer_radius[i] = disk_outer_radius[min_z_index];
      disk_thickness[i] = disk_thickness[min_z_index];

      disk_z[min_z_index] = temp_z;
      disk_inner_radius[min_z_index] = temp_inner_radius;
      disk_outer_radius[min_z_index] = temp_outer_radius;
      disk_thickness[min_z_index] = temp_thickness;
    }

    //sort the barrel layers according to z
    for(unsigned int i = 0; i < barrel_radius.size(); i++){
      
      float min_r = -1;
      int min_r_index = -1;

      //find the minimum out of the remaining elements (elements i to barrel_radius.size() - 1)
      for(unsigned int j = i; j < barrel_radius.size(); j++){
	if(j == i){
	  min_r = barrel_radius[j];
	  min_r_index = j;
	}
	else if (barrel_radius[j] < min_r){
	  min_r = barrel_radius[j];
	  min_r_index = j;
	}
      }

      //swap the element at min_r_index with the one at index i
      float temp_r = barrel_radius[i];
      float temp_barrel_length = barrel_length[i];
      float temp_barrel_thickness = barrel_thickness[i];
  
      barrel_radius[i] = barrel_radius[min_r_index];
      barrel_length[i] = barrel_length[min_r_index];
      barrel_thickness[i] = barrel_thickness[min_r_index];
  
      barrel_radius[min_r_index] = temp_r;
      barrel_length[min_r_index] = temp_barrel_length;
      barrel_thickness[min_r_index] = temp_barrel_thickness;
    }

    
    const Surface::RotationType theRotation2(1.,0.,0.,0.,1.,0.,0.,0.,1.);
                
    const Surface::PositionType thePosition(0.,0.,0.);
    const Surface::RotationType theRotation(1.,0.,0.,0.,1.,0.,0.,0.,1.);
    
    float max_Z=0.;
    float max_R=0.;

    for(unsigned int i = 0, j = 0; i < barrel_length.size() || j < disk_z.size(); ){
      std::cout << "i,j = " << i << " " << j << std::endl;

      bool add_disk = false;
      if(i < barrel_length.size() && j < disk_z.size()){
	//	if(disk_outer_radius[j] < barrel_radius[i])
	if(disk_z[j] < barrel_length[i]/2.)
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

      std::cout << "add_disk= " << add_disk << std::endl;
      
      if(add_disk){
	_mediumProperties.push_back(new MediumProperties(disk_thickness[j],0.0001));  

	if (disk_outer_radius[j]<max_R)
	  {
	    // need to to this in order to have nested cylinders.... geometry will be weird bu rely on matching with reco geometry...
	    max_R=max_R+0.1;
	  }
	else
	  max_R=disk_outer_radius[j];
      
	const SimpleDiskBounds diskBounds(disk_inner_radius[j],max_R,-0.0150,+0.0150);
	const Surface::PositionType positionType(0.,0.,disk_z[j]);

	unsigned layerNr = i+j;
	BoundDisk* theDisk = new BoundDisk(positionType,theRotation2,diskBounds);
	theDisk->setMediumProperties(_mediumProperties[_mediumProperties.size() -1 ]);
	if ( theDisk->mediumProperties()->radLen() > 0. ) 
	  {
	    _theCylinders.push_back(TrackerLayer(theDisk,true,layerNr,
					       std::vector<double>(),std::vector<double>(),
					       std::vector<double>()));
	    std::cout << "disk added" << std::endl;
	    if (disk_z[j]>max_Z) max_Z=disk_z[j];
	    
	  }
	else
	  delete theDisk;
	j++;
      }
      else 
	{
	  
	  // Create the nest of cylinders
	  
	  //const SimpleCylinderBounds  cylBounds(  barrel_radius[i]-0.0150, barrel_radius[i]+0.0150, -barrel_length[i]/2, +barrel_length[i]/2);
	  
	  if (barrel_length[i]<2.*max_Z)
	    {
	      // need to to this in order to have nested cylinders.... geometry will be weird bu rely on matching with reco geometry...
	      max_Z=max_Z+0.1;
	    }
	  else
	    max_Z=barrel_length[i]/2.;
	  
	  const SimpleCylinderBounds  cylBounds(  barrel_radius[i], barrel_radius[i], -max_Z, max_Z);
	  
	  _mediumProperties.push_back(new MediumProperties(barrel_thickness[i],0.0001));  
	  
	  unsigned layerNr = i+j;
	  BoundCylinder* theCylinder = new BoundCylinder(thePosition,theRotation,cylBounds);
	  theCylinder->setMediumProperties(_mediumProperties[_mediumProperties.size() -1 ]);
	  if ( theCylinder->mediumProperties()->radLen() > 0. ) 
	    {
	    _theCylinders.push_back(TrackerLayer(theCylinder,false,layerNr,
						 std::vector<double>(),std::vector<double>(),
						 std::vector<double>()));
	    std::cout << "cylinder added" << std::endl;

	    if (barrel_radius[i]>max_R) max_R=barrel_radius[i];
	    }
	  else
	    delete theCylinder;
	  
	  i++;
	}
    }


  // Check overall compatibility of cylinder dimensions
  // (must be nested cylinders)
  // Throw an exception if the test fails
  double zin, rin;
  double zout, rout;
  unsigned nCyl=0;
  std::list<TrackerLayer>::const_iterator cyliterOut=cylinderBegin();
  std::cout << "Number of defined cylinders: " << nCylinders() << std::endl;

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
      //      throw cms::Exception("FastSimulation/TrackerInteractionGeometry ") 
	std::cout 
	<< " WARNING with cylinder number " << nCyl 
	<< " (Active Layer Number = " <<  cyliterOut->layerNumber() 
	<< " Forward ? " <<  cyliterOut->forward() << " ) "
	<< " has dimensions smaller than previous cylinder : " << std::endl
	<< " zout/zin = " << zout << " " << zin << std::endl
	<< " rout/rin = " << rout << " " << rin << std::endl;
    } else {
      std::cout << " Cylinder number " << nCyl 
		<< " (Active Layer Number = " <<  cyliterOut->layerNumber() 
		<< " Forward ? " <<  cyliterOut->forward() << " ) "
		<< " has dimensions of : " 
		<< " zout = " << zout << "; " 
		<< " rout = " << rout << std::endl;
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
