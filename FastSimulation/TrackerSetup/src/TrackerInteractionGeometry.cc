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
    LogDebug("FastSimGeom") << "defining barrellayer: " << i ;
    LogDebug("FastSimGeom") << "   length,radius= " <<
      barrelLayers[i]->specificSurface().bounds().length() << " " <<
      barrelLayers[i]->specificSurface().radius() << " " ;
    
    barrel_length.push_back( barrelLayers[i]->specificSurface().bounds().length());
    barrel_radius.push_back( barrelLayers[i]->specificSurface().radius());
    barrel_thickness.push_back(0.05);
    
  }
  
  for(unsigned int i = 0; i < posForwardLayers.size(); i++){
    
    LogDebug("FastSimGeom") << "defining forwardlayer: " << i ;
    LogDebug("FastSimGeom") << "  z,inner R, outer R = " <<     
      posForwardLayers[i]->surface().position().z() << " " <<
      posForwardLayers[i]->specificSurface().innerRadius() << " " <<
      posForwardLayers[i]->specificSurface().outerRadius() ;
    
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
    LogDebug("FastSimGeom") << "i,j = " << i << " " << j ;
    
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
    
    LogDebug("FastSimGeom") << "add_disk= " << add_disk ;
    
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
      theDisk->setMediumProperties(*_mediumProperties[_mediumProperties.size() -1 ]);
	if ( theDisk->mediumProperties().radLen() > 0. ) 
	  {
	    _theCylinders.push_back(TrackerLayer(theDisk,true,layerNr,
						 std::vector<double>(),std::vector<double>(),
						 std::vector<double>()));
	    LogDebug("FastSimGeom") << "disk added" ;
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
	theCylinder->setMediumProperties(*_mediumProperties[_mediumProperties.size() -1 ]);
	if ( theCylinder->mediumProperties().radLen() > 0. ) 
	  {
	    _theCylinders.push_back(TrackerLayer(theCylinder,false,layerNr,
						 std::vector<double>(),std::vector<double>(),
						 std::vector<double>()));
	    LogDebug("FastSimGeom") << "cylinder added" ;

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
  LogDebug("FastSimGeom") << "Number of defined cylinders: " << nCylinders() ;

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
	//	std::cout 
	<< " WARNING with cylinder number " << nCyl 
	<< " (Active Layer Number = " <<  cyliterOut->layerNumber() 
	<< " Forward ? " <<  cyliterOut->forward() << " ) "
	<< " has dimensions smaller than previous cylinder : " << std::endl
	<< " zout/zin = " << zout << " " << zin << std::endl
	<< " rout/rin = " << rout << " " << rin << std::endl;
    } else {
      LogDebug("FastSimGeom") << " Cylinder number " << nCyl 
		<< " (Active Layer Number = " <<  cyliterOut->layerNumber() 
		<< " Forward ? " <<  cyliterOut->forward() << " ) "
		<< " has dimensions of : " 
		<< " zout = " << zout << "; " 
		<< " rout = " << rout ;
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

  for(unsigned int i = 0; i < _mediumProperties.size(); i++){
    delete _mediumProperties[i];
  }

}
