/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/03/29 14:21:57 $
 *  $Revision: 1.11 $
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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace edm;

MagGeometry::MagGeometry(const edm::ParameterSet& config, std::vector<MagBLayer *> tbl,
			 std::vector<MagESector *> tes,
			 std::vector<MagVolume6Faces*> tbv,
			 std::vector<MagVolume6Faces*> tev) : 
  lastVolume(0), theBLayers(tbl), theESectors(tes), theBVolumes(tbv), theEVolumes(tev)
{
  
  cacheLastVolume = config.getUntrackedParameter<bool>("cacheLastVolume", true);
  timerOn = config.getUntrackedParameter<bool>("timerOn", false);
  v_85l = (config.getParameter<std::string>("version")=="grid_85l_030919");

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

  // FIXME: Get these dimensions from the builder. 
  // For this we can wait the next generation of tables, when the picture 
  // may be more complicated
  if (v_85l){
    theZ1 = 634.49;
  } else {
    theZ1 = 633.29;
  }
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

  // Map version 85l is Z-symmetric; -> implement Z reflection
  if (v_85l) { 
    GlobalPoint gpSym = gp;
    bool atMinusZ = true;
    if (gpSym.z()>0.) {
      atMinusZ = false;
      gpSym=GlobalPoint(gp.x(), gp.y(), -gp.z()); 
    }
    
    MagVolume * v = 0;

    // Check volume cache
    if (cacheLastVolume && lastVolume!=0 && lastVolume->inside(gpSym)){
      v = lastVolume;
    } else {
      v = findVolume(gpSym);
    }
    

    if (v==0) {
      // If search fails, retry with a 300 micron tolerance.
      // This is a hack for thin gaps on air-iron boundaries,
      // which will not be present anymore once surfaces are matched.
      if (verbose::debugOut) cout << "Increasing the tolerance to 0.03" <<endl;
      v = findVolume(gpSym, 0.03);
    }
    
    if (v!=0) {
      GlobalVector bresult = v->fieldInTesla(gpSym);
      if (atMinusZ) return bresult;
      else return GlobalVector(-bresult.x(), -bresult.y(), bresult.z());
    }
    

    // Map versions 1103l is not Z symmetric; dumb volume search for the time being.
  } else {
    
    MagVolume * v = 0;

    // Check volume cache
    if (cacheLastVolume && lastVolume!=0 && lastVolume->inside(gp)){
      v = lastVolume;
    } else {
      // FIXME: endcap layers already built -> optimized search!!!
      if (inBarrel(gp)) {
	v = findVolume1(gp);
      } else {
	v = findVolume(gp);
      }
    }
    
    // If search fails, retry with increased tolerance
    if (v==0) {
      // If search fails, retry with a 300 micron tolerance.
      // This is a hack for thin gaps on air-iron boundaries,
      // which will not be present anymore once surfaces are matched.
      if (verbose::debugOut) cout << "Increasing the tolerance to 0.03" <<endl;
      // FIXME: endcap layers already built -> optimized search!!!
       if (inBarrel(gp)) {
	 v = findVolume1(gp, 0.03);
       } else {
	 v = findVolume(gp, 0.03);
       }
    }
    
    if (v!=0) {
      // cout << "inside: " << ((MagVolume6Faces*) v)->name << endl;
      return v->fieldInTesla(gp);
    }  
  }
  
  // Fall-back case: no volume found
  
  if (isnan(gp.mag())) {
    LogWarning("InvalidInput") << "Input value invalid (not a number): " << gp << endl;
      
  } else {
    LogWarning("MagneticField") << "MagGeometry::fieldInTesla: failed to find volume for " << gp << endl;
  }
  return GlobalVector();
}


// Linear search implementation (just for testing)
MagVolume* 
MagGeometry::findVolume1(const GlobalPoint & gp, double tolerance) const {  

  MagVolume6Faces * found = 0;

  if (inBarrel(gp)) { // Barrel
    for (vector<MagVolume6Faces*>::const_iterator v = theBVolumes.begin();
	 v!=theBVolumes.end(); ++v){
      if ((*v)==0) { //FIXME: remove this check
	cout << endl << "***ERROR: MagGeometry::findVolume: MagVolume not set" << endl;
	continue;
      }
      if ((*v)->inside(gp,tolerance)) {
	found = (*v);
	break;
      }
    }

  } else { // Endcaps
    for (vector<MagVolume6Faces*>::const_iterator v = theEVolumes.begin();
	 v!=theEVolumes.end(); ++v){
      if ((*v)==0) {  //FIXME: remove this check
	cout << endl << "***ERROR: MagGeometry::findVolume: MagVolume not set" << endl;
	continue;
      }
      if ((*v)->inside(gp,tolerance)) {
	found = (*v);
	break;
      }
    }
  }  
  
  return found;
}

// Use hierarchical structure for fast lookup.
MagVolume* 
MagGeometry::findVolume(const GlobalPoint & gp, double tolerance) const{
  MagVolume * result=0;

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

  return result;
}




bool MagGeometry::inBarrel(const GlobalPoint& gp) const {
  // FIXME! hardcoded boundary between barrel and endcaps!
  float Z = gp.z();
  float R = gp.perp();
  return (fabs(Z)< theZ1 || (R>308.755 && fabs(Z)<661.01));
}





