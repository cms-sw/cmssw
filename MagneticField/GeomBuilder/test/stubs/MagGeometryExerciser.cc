//#include "Utilities/Configuration/interface/Architecture.h"

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2005/12/12 18:36:24 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/GeomBuilder/test/stubs/MagGeometryExerciser.h"
#include "MagneticField/VolumeBasedEngine/interface/MagGeometry.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "Utilities/Timing/interface/TimingReport.h"

#include <CLHEP/Random/RandFlat.h>
#include <algorithm>

using namespace std;

MagGeometryExerciser::MagGeometryExerciser(MagGeometry * g) : theGeometry(g) {
  const vector<MagVolume6Faces*>& theBVolumes = theGeometry->barrelVolumes();
  const vector<MagVolume6Faces*>& theEVolumes = theGeometry->endcapVolumes();

  volumes = theBVolumes;
  volumes.insert(volumes.end(), theEVolumes.begin(), theEVolumes.end());
}



MagGeometryExerciser::~MagGeometryExerciser(){}


//----------------------------------------------------------------------
// Check if findVolume succeeds for random points.
// Note: findVolumeTolerance in pset to change the tolerance.
void MagGeometryExerciser::testFindVolume(int ntry){  
  
  bool zSymmetric = true; // findVolume just sees the "real" volumes,
                          // so if the half at z<0 is built it will fail for z>0!
  bool barrelOnly = false; // Only in the barrel
  
  const float minR = 0.;
  const float maxR = 1000.;
  const float minPhi = -Geom::pi();
  const float maxPhi = Geom::pi();
  float minZ = -1600;
  float maxZ = 1600;

  if (barrelOnly) {
    minZ = -662.;
    maxZ = 662.;  
  }
  
  if (zSymmetric) maxZ=0.;

  cout << "-----------------------------------------------------" << endl
       << " findVolume(random) test" << endl;
  
  // Test  known overlaps/gaps
  if (true) {
    cout << "Known points:" << endl;
    testFindVolume(GlobalPoint(0,0,0));
  }

  cout << "Random points:" << endl;
  for (int i = 0; i<ntry; ++i) {
    float R = RandFlat::shoot(minR,maxR);
    float Z = RandFlat::shoot(minZ,maxZ);
    float phi = RandFlat::shoot(minPhi,maxPhi);

    GlobalPoint gp(GlobalPoint::Cylindrical(R,phi,Z));
    
    if (barrelOnly && !(theGeometry->inBarrel(gp))) continue; // Barrel

    testFindVolume(gp);
    
  }
}

//----------------------------------------------------------------------
// Check if findVolume succeeds for the given point.
void MagGeometryExerciser::testFindVolume(const GlobalPoint & gp){

  bool reportSuccess = false; // printouts for succeeding calls

  // Note: uses the default tolerance.
  MagVolume6Faces* vol = (MagVolume6Faces*) theGeometry->findVolume(gp);
  
  if (reportSuccess || vol==0) {
    cout << gp << " "
	 << (vol !=0 ? vol->name : "ERROR no volume found! ")
	 << endl;
  }

  // If it fails, try with a linear search
  if (vol==0) {
    float tolerance = 0.03;
    vol =  (MagVolume6Faces*) theGeometry->findVolume1(gp,tolerance);
    cout << "Was in volume: "
	 << (vol !=0 ? vol->name : "none")
	 << " (tolerance = " << tolerance << ")"
	 << endl;
  }
}



//----------------------------------------------------------------------
// Check that a set of points is inside() one and only one volume.
void MagGeometryExerciser::testInside(int ntry) {

  cout << "-----------------------------------------------------" << endl
       << " inside(random) test" << endl;


  // Test random points: they should be found inside() one and only one volume.
    
  bool zSymmetric = true;
  bool barrelOnly = false;
  bool test82 = false; // Test problems with volume 82...

  float minR = 0.;
  float maxR = 1000.;
  float minPhi = -Geom::pi();
  float maxPhi = Geom::pi();
  float minZ = -1600.;
  float maxZ = 1600.;


  if (barrelOnly) {
    minZ = -662.;
    maxZ = 662.;  
  }  

  if (test82) {
     minZ = -660;
     maxZ = 660.;  
     minR = 411.5; //V81
     maxR = 447.; //V83
     minPhi= 104./180.*Geom::pi();
     maxPhi= 113./180.*Geom::pi();
  }

  if (zSymmetric) maxZ=0.;

  // Test some known overlaps/gaps
  if (true) {
    cout << "Known points:" << endl;
    testInside(GlobalPoint(0.,0.,0.));
    testInside(GlobalPoint(331.358,-278.042,-648.788));//V_ZN_81 V_ZN_82
    testInside(GlobalPoint(-426.188,75.1483,-533.075));//V_ZN_81 V_ZN_82
    testInside(GlobalPoint(-586.27,157.094,150.702));  //V_ZN_170 V_ZN_174
    testInside(GlobalPoint(-465.562,-465.616,-222.224));//203 205
    testInside(GlobalPoint(637.078,170.737,-222.632));  //203 205
    testInside(GlobalPoint(162.971,-608.294,-306.791));  //203 205
    testInside(GlobalPoint(-633.925,-169.839,-390.589)); //203 205
    testInside(GlobalPoint(170.358,635.857,-535.735));  //207 209
    testInside(GlobalPoint(166.836,622.58,-513.872));  //207 209
    testInside(GlobalPoint(-12.7086,-3.99366,-1313.01));  //53 56
    testInside(GlobalPoint(106.178,-538.867,-69.2348));  //147 148
    testInside(GlobalPoint(19.1496,522.831,-1333));  //50 64
    testInside(GlobalPoint(355.516,-479.729,-1333.01));  //50 64
    testInside(GlobalPoint(157.96,-389.223,-1333.01)); //50 64
    testInside(GlobalPoint(-464.338,464.234,-473.616));  // 207 209
    testInside(GlobalPoint(17.474,-669.279,-1333.01)); // 50 64
    testInside(GlobalPoint(-453.683,-453.601,-204.895)); // 203 205
    testInside(GlobalPoint(165.092,-615.998,-494.625)); // 207 209
    testInside(GlobalPoint(-586.27,157.094,-617.882)); // 177 177
    testInside(GlobalPoint(-142.226,390.763,-553.809)); //81 79
    testInside(GlobalPoint(-141.701,389.32,-142.088));  //81 79
  }
  
  cout << "Random points:" << endl;
  for (int i = 0; i<ntry; ++i) {
    float R = RandFlat::shoot(minR,maxR);
    float Z = RandFlat::shoot(minZ,maxZ);
    float phi = RandFlat::shoot(minPhi,maxPhi);

    if (i%1000==0) cout << "test # " << i << endl;
    
    GlobalPoint gp(GlobalPoint::Cylindrical(R,phi,Z));

    if (barrelOnly && !(theGeometry->inBarrel(gp))) continue;// Barrel
    
    testInside(gp);

  }
}


//----------------------------------------------------------------------
// Check that the given point is inside() one and only one volume.
// This is not always the case due to thin gaps and tolerance...
void MagGeometryExerciser::testInside(const GlobalPoint & gp){
  //FIXME  static const double tolerance = SimpleConfigurable<double>(0.,"MagGeometryExerciser:tolerance"); // 300 micron thin gaps
  static const double tolerance = 0.;

  bool reportSuccess = false;

  vector<MagVolume6Faces*>& vols = volumes;
  // or use only barrel volumes:
  // const vector<MagVolume6Faces*>& vols = theGeometry->barrelVolumes();

  MagVolume6Faces * found = 0;
  for (vector<MagVolume6Faces*>::const_iterator v = vols.begin();
       v!=vols.end(); ++v){
    if ((*v)==0) {
      cout << endl << "ERROR: no magvlolume" << endl;
      continue;
    }
    if ((*v)->inside(gp, tolerance)) {
      if (reportSuccess) cout << gp  << " is inside vol: " << (*v)->name;
      if (found!=0) {
	cout << " ***ERROR: for " << gp << " found " << (*v)->name
	     << " volume already found: " << found->name << endl;
      }
      found = (*v);
    }
  }
   
  if (found==0) {
    cout << gp << " ***ERROR no volume found! "  << endl;
  }  
}
