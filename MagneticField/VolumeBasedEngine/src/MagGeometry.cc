/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/04/20 10:14:57 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/VolumeBasedEngine/interface/MagGeometry.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"
#include "MagneticField/Layers/interface/MagBLayer.h"
#include "MagneticField/Layers/interface/MagESector.h"

#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"

#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "Utilities/Timing/interface/TimingReport.h"
#include "MagneticField/Layers/interface/MagVerbosity.h"

using namespace std;
using namespace edm;

MagGeometry::MagGeometry(const edm::ParameterSet& config, std::vector<MagBLayer *> tbl,
			 std::vector<MagESector *> tes,
			 std::vector<MagVolume6Faces*> tbv,
			 std::vector<MagVolume6Faces*> tev) : 
  lastVolume(0), theBLayers(tbl), theESectors(tes), theBVolumes(tbv), theEVolumes(tev)
{
  
  tolerance = config.getParameter<double>("findVolumeTolerance");
  cacheLastVolume = config.getUntrackedParameter<bool>("cacheLastVolume");
  timerOn = config.getUntrackedParameter<bool>("timerOn", false);


  TimeMe t1("MagGeometry:build",false);

  cout << endl
       << "         ___________________________________        " << endl
       << "      .-'                                   '-.     " << endl
       << "    .'      Magnetic Field Geometry built      `.   " << endl
       << "   /                                             \\  " << endl
       << "  ;         Barrel volumes: " << theBVolumes.size() << "                  ; " << endl
       << "  ;         Endcap volumes: " << theEVolumes.size() << "                   ; " << endl
       << "   \\                                             /  " << endl
       << "    `.          Fasten your seatbelt.          .'   " << endl
       << "      `-.___________________________________.-'     " << endl
       << endl;

  vector<double> rBorders;

  for (vector<MagBLayer *>::const_iterator ilay = theBLayers.begin();
       ilay != theBLayers.end(); ++ilay) {
    if (verbose::debugOut) cout << "  Barrel layer at " << (*ilay)->minR() <<endl;
    //FIXME assume layers are already sorted in minR
    rBorders.push_back((*ilay)->minR());
  }

  theBarrelBinFinder = new MagBinFinders::GeneralBinFinderInR<double>(rBorders);

  if (verbose::debugOut) {
    for (vector<MagESector *>::const_iterator isec = theESectors.begin();
	 isec != theESectors.end(); ++isec) {
      cout << "  Endcap sector at " << (*isec)->minPhi() << endl;
    }
  }

  //FIXME assume sectors are already sorted in phi
  //FIXME: PeriodicBinFinderInPhi gets *center* of first bin
  theEndcapBinFinder = new PeriodicBinFinderInPhi<float>(theESectors.front()->minPhi()+Geom::pi()/12., 12);

  // Disable timers to save CPU
  (*TimingReport::current()).switchOn("MagGeometry::fieldInTesla",timerOn);
  (*TimingReport::current()).switchOn("MagGeometry::fieldInTesla:VolumeQuery",timerOn);
  (*TimingReport::current()).switchOn("MagGeometry::findVolume",timerOn);
  (*TimingReport::current()).switchOn("MagGeometry::findVolume1",timerOn);
  (*TimingReport::current()).switchOn("MagGeometry::findVolume2",timerOn);  

}

MagGeometry::~MagGeometry(){
  delete theBarrelBinFinder;
  delete theEndcapBinFinder;

  for (vector<MagBLayer *>::const_iterator ilay = theBLayers.begin();
       ilay != theBLayers.end(); ++ilay) {
    delete (*ilay);
  }

  for (vector<MagESector *>::const_iterator ilay = theESectors.begin();
       ilay != theESectors.end(); ++ilay) {
    delete (*ilay);
  }
}


// Return field vector at the specified global point
GlobalVector MagGeometry::fieldInTesla(const GlobalPoint & gp) const {
  static TimingReport::Item & timer1 = (*TimingReport::current())["MagGeometry::fieldInTesla"];
  static TimingReport::Item & timer2 = (*TimingReport::current())["MagGeometry::fieldInTesla:VolumeQuery"];
  TimeMe t1(timer1,false);
    
  // If point is outside magfield map, return 0 field.
  if (abs(gp.z()) > 1600. || gp.perp() > 1000.) return GlobalVector();

  GlobalPoint gpSym = gp;
  bool atMinusZ = true;
  if (gpSym.z()>0.) {
    atMinusZ = false;
    gpSym=GlobalPoint(gp.x(), gp.y(), -gp.z()); 
  }

  MagVolume * v = findVolume(gpSym);
  if (v!=0) {
    TimeMe t2(timer2,false);
    GlobalVector result = v->fieldInTesla(gpSym);
    if (atMinusZ) return result;
    else return GlobalVector(-result.x(), -result.y(), result.z());
  } else {
    cout << "MagGeometry::fieldInTesla: failed to find volume for " << gpSym << endl;
    return GlobalVector();
  }
}



MagVolume * MagGeometry::findVolume(const GlobalPoint & gp) const {
//   static const double tolerance = (SimpleConfigurable<double>(0.,"MagGeometry:FindVolumeTolerance"));
//   static const bool cacheLastVolume = (SimpleConfigurable<bool>(true,"MagGeometry:cacheLastVolume"));

  static TimingReport::Item & timer = (*TimingReport::current())["MagGeometry::findVolume"];
  TimeMe t(timer,false);

  if (cacheLastVolume && lastVolume!=0 && lastVolume->inside(gp, tolerance)){
    return lastVolume;
  }
  return (lastVolume = findVolume2(gp, tolerance));
}


// Linear search implementation (just for testing)
MagVolume* 
MagGeometry::findVolume1(const GlobalPoint & gp, double tolerance) const {
  static TimingReport::Item & timer = (*TimingReport::current())["MagGeometry::findVolume1"];
  TimeMe t(timer,false);

  //FIXME: perform the search only in negative Z volumes
  GlobalPoint gpSym(gp.x(), gp.y(), (gp.z()<0? gp.z() : -gp.z()));

  MagVolume6Faces * found = 0;
  if (inBarrel(gpSym)) { // Barrel
    for (vector<MagVolume6Faces*>::const_iterator v = theBVolumes.begin();
	 v!=theBVolumes.end(); ++v){
      if ((*v)==0) {
	cout << endl << "***ERROR: MagGeometry::findVolume: MagVolume not set" << endl;
	continue;
      }
      if ((*v)->inside(gpSym,tolerance)) {
	found = (*v);
	break;
      }
    }
  } else { // Endcaps
    for (vector<MagVolume6Faces*>::const_iterator v = theEVolumes.begin();
	 v!=theEVolumes.end(); ++v){
      if ((*v)==0) {
	cout << endl << "***ERROR: MagGeometry::findVolume: MagVolume not set" << endl;
	continue;
      }
      if ((*v)->inside(gpSym,tolerance)) {
	found = (*v);
	break;
      }
    }
  }  
  
  return found;
}

// Use hierarchical structure for fast lookup.
//FIXME: The search is performed only in negative Z volumes (gp.z() must be <=0)
MagVolume* 
MagGeometry::findVolume2(const GlobalPoint & gp, double tolerance) const{
  MagVolume * result=0;
  static TimingReport::Item & timer = (*TimingReport::current())["MagGeometry::findVolume2"];
  TimeMe t(timer,false);

  //  GlobalPoint gpSym(gp.x(), gp.y(), (gp.z()<0? gp.z() : -gp.z()));

  if (inBarrel(gp)) { // Barrel
    double R = gp.perp();
    int bin = theBarrelBinFinder->binIndex(R);
    
    for (int bin1 = bin; bin1 >= max(0,bin-2); --bin1) {
      if (verbose::debugOut) cout << "Trying layer at R " << theBLayers[bin1]->minR()
		      << " " << R << endl ;
      result = theBLayers[bin1]->findVolume(gp, tolerance);
      if (verbose::debugOut) cout << "***In blayer " << bin1-bin << " " 
		      << (result==0? " failed " : " OK ") <<endl;
      if (result != 0) break;
    }

  } else { // Endcaps
    Geom::Phi<float> phi = gp.phi();
    int bin = theEndcapBinFinder->binIndex(phi);
    if (verbose::debugOut) cout << "Trying endcap sector at phi "
		    << theESectors[bin]->minPhi() << " " << phi << endl ;
    result = theESectors[bin]->findVolume(gp, tolerance);
    if (verbose::debugOut) cout << "***In guessed esector "
		    << (result==0? " failed " : " OK ") <<endl;
  }


  if (result == 0 && tolerance < 0.0001) {
    // Try increasing the tolerance to 300 micron
    // FIXME: this is a temporary hack for thin gaps on air-iron boundaries,
    // which will not be present anymore once surfaces are matched.
    if (verbose::debugOut) cout << "Increasing the tolerance to 0.03" <<endl;
    result = findVolume2(gp, 0.03); 
  }

  return result;
}




bool MagGeometry::inBarrel(const GlobalPoint& gp) const {
  // FIXME! hardcoded boundary between barrel and endcaps!
  float Z = gp.z();
  float R = gp.perp();
  return (fabs(Z)<634.49 || (R>308.755 && fabs(Z)<661.01));
}

