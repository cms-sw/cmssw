//#include "Utilities/Configuration/interface/Architecture.h"

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2013/04/15 16:22:20 $
 *  $Revision: 1.10 $
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/GeomBuilder/test/stubs/MagGeometryExerciser.h"
#include "MagneticField/VolumeBasedEngine/interface/MagGeometry.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"
#include "GlobalPointProvider.h"

#include <algorithm>

using namespace std;

MagGeometryExerciser::MagGeometryExerciser(const MagGeometry * g) : theGeometry(g) {
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
  
  cout <<endl
       << "-----------------------------------------------------" << endl
       << " findVolume(random) test" << endl;

  // Test  known overlaps/gaps
  if (true) {
    cout << "Known points:" << endl;
    testFindVolume(GlobalPoint(0,0,0));
  }

  GlobalPointProvider p(0.,900., -Geom::pi(), Geom::pi(), -1600, 1600);

  cout << "Random points:" << endl;
  int success = 0;
  for (int i = 0; i<ntry; ++i) {
    if (testFindVolume(p.getPoint())) {
      ++success;
    }
  }

  cout << " Tested " <<  ntry << " Failures: " << ntry - success << endl
       << "-----------------------------------------------------" << endl;
  
}

//----------------------------------------------------------------------
// Check if findVolume succeeds for the given point.
bool MagGeometryExerciser::testFindVolume(const GlobalPoint & gp){
  float tolerance = 0.;
  //  float tolerance = 0.03;  // Note: findVolume should handle tolerance himself.
  MagVolume6Faces* vol = (MagVolume6Faces*) theGeometry->findVolume(gp, tolerance);
  bool ok = (vol!=0);

  if (vol==0) {
    cout << "ERROR no volume found! " 
	 << gp << " " << gp.z() << " " << gp.perp()
	 << " isBarrel: " << theGeometry->inBarrel(gp) 
	 << endl;
  

    // Try with a linear search
    vol =  (MagVolume6Faces*) theGeometry->findVolume1(gp,tolerance);
    cout << "Was in volume: "
	 << (vol !=0 ? vol->volumeNo : -1)
	 << " (tolerance = " << tolerance << ")"
	 << endl;
  }

  return ok;
}



//----------------------------------------------------------------------
// Check that a set of points is inside() one and only one volume.
void MagGeometryExerciser::testInside(int ntry, float tolerance) {

  cout << "-----------------------------------------------------" << endl
       << " inside(random) test" << endl;


  // Test random points: they should be found inside() one and only one volume

  // Test some known overlaps/gaps
  if (false) { //FIXME
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


  // Full CMS
  GlobalPointProvider p(0,900,-Geom::pi(),Geom::pi(),-1999.9,1999.9);

  // Zoom of one sector
  //  GlobalPointProvider p(350.,900.,-0.27,0.27,-1999.9,1999.9);
  //  GlobalPointProvider p(0.,900.,-0.27,0.27,-1999.9,1999.9);

  cout << "Random points:" << volumes.size() << " volumes" << endl;
  for (int i = 0; i<ntry; ++i) {
    if (i%1000==0) cout << "test # " << i << endl;    
    testInside(p.getPoint(), tolerance);
  }
}


//----------------------------------------------------------------------
// Check that the given point is inside() one and only one volume.
// This is not always the case due to thin gaps and tolerance...
bool MagGeometryExerciser::testInside(const GlobalPoint & gp, float tolerance){

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
      if (reportSuccess) cout << gp  << " is inside vol: " << (*v)->volumeNo;
      if (found!=0) {
	cout << " ***ERROR: for " << gp << " found " << (*v)->volumeNo
	     << " volume already found: " << found->volumeNo << endl;
      }
      found = (*v);
    }
  }
  
  

  if (found==0) {
    MagVolume6Faces * foundP = 0;
    MagVolume6Faces * foundN = 0;
    // Look for the closest neighbouring volumes
    const float phi=gp.phi();
    GlobalPoint gpP, gpN;
    int ntry=0;
    while ((foundP==0 || foundP==0) && ntry < 60) {
      ++ntry;
      for (vector<MagVolume6Faces*>::const_iterator v = vols.begin();
	   v!=vols.end(); ++v){
	if (foundP==0) {
	  float phiP=phi+ntry*0.008727;
	  GlobalPoint gpP(GlobalPoint::Cylindrical(gp.perp(),phiP,gp.z()));
	  if ((*v)->inside(gpP)) {
	    foundP=(*v);
	  }
	}
	if (foundN==0) {
	  float phiN=phi-ntry*0.008727;
	  GlobalPoint gpN(GlobalPoint::Cylindrical(gp.perp(),phiN,gp.z()));
	  if ((*v)->inside(gpN)) {
	    foundN=(*v);
	  }
	}
      }
    }
    
    cout << gp << " ***ERROR no volume found! : closests: "
	 << ((foundP==0) ? -1 : foundP->volumeNo)
	 << " at dphi: " << gpP.phi()-phi << " "
	 << ((foundN==0) ? -1 :foundN->volumeNo)
	 << " at dphi: " << gpN.phi()-phi 
	 << endl;
  }
  
  return true; //FIXME
}

//----------------------------------------------------------------------

//   if (test82) {
//      minZ = -660;
//      maxZ = 660.;  
//      minR = 411.5; //V81
//      maxR = 447.; //V83
//      minPhi= 104./180.*Geom::pi();
//      maxPhi= 113./180.*Geom::pi();
//   }
