// #include "Utilities/Configuration/interface/Architecture.h"

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/05/31 13:52:51 $
 *  $Revision: 1.7 $
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/GeomBuilder/src/MagGeoBuilderFromDDD.h"
#include "MagneticField/GeomBuilder/src/volumeHandle.h"
#include "MagneticField/GeomBuilder/src/bSlab.h"
#include "MagneticField/GeomBuilder/src/bRod.h"
#include "MagneticField/GeomBuilder/src/bSector.h"
#include "MagneticField/GeomBuilder/src/bLayer.h"
#include "MagneticField/GeomBuilder/src/eSector.h"
#include "MagneticField/GeomBuilder/src/eLayer.h"

#include "MagneticField/Layers/interface/MagBLayer.h"
#include "MagneticField/Layers/interface/MagESector.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"

#include "Utilities/BinningTools/interface/ClusterizingHistogram.h"
#include "CLHEP/Units/SystemOfUnits.h"

#include "MagneticField/Interpolation/interface/MagProviderInterpol.h"
#include "MagneticField/Interpolation/interface/MFGridFactory.h"
#include "MagneticField/Interpolation/interface/MFGrid.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"
#include "MagneticField/VolumeGeometry/interface/MagExceptions.h"
#include "MagneticField/Layers/interface/MagVerbosity.h"

#include "DataFormats/GeometryVector/interface/Pi.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <map>
#include <set>
#include "Utilities/General/interface/precomputed_value_sort.h"

#include "MagneticField/GeomBuilder/src/VolumeBasedMagneticFieldESProducer.h"

using namespace std;

MagGeoBuilderFromDDD::MagGeoBuilderFromDDD()  {
  if (bldVerb::debugOut) cout << "Constructing a MagGeoBuilderFromDDD" <<endl;
}

MagGeoBuilderFromDDD::~MagGeoBuilderFromDDD(){
  for (handles::const_iterator i=bVolumes.begin();
       i!=bVolumes.end(); ++i){
    delete (*i);
  }
  
  for (handles::const_iterator i=eVolumes.begin();
       i!=eVolumes.end(); ++i){
    delete (*i);
  }
}


void MagGeoBuilderFromDDD::summary(handles & volumes){  
  // The final countdown.
  int ivolumes  = volumes.size();  // number of volumes
  int isurfaces = ivolumes*6;       // number of individual surfaces
  int iassigned = 0;                // How many have been assigned
  int iunique   = 0;                // number of unique surfaces
  int iref_ass  = 0;
  int iref_nass = 0;

  set<int> ptrs;

  handles::const_iterator first = volumes.begin();
  handles::const_iterator last = volumes.end();

  for (handles::const_iterator i=first; i!=last; ++i){
    if (int((*i)->shape())>4) continue; // FIXME: missing shapes...
    for (int side = 0; side < 6; ++side) {
      int references = 	(*i)->references(side);
      if ((*i)->isPlaneMatched(side)) {
	++iassigned;
	bool firstOcc = (ptrs.insert((int) &((*i)->surface(side)))).second;
	if (firstOcc) iref_ass+=references;
	if (references<2){  
	  cout << "*** Only 1 ref, vol: " << (*i)->name << " # "
	       << (*i)->copyno << " side: " << side << endl;
	}	
      } else {
	iref_nass+=references;
	if (references>1){
	  cout << "*** Ref_nass >1 " <<endl;
	}
      }
    }
  }
  iunique = ptrs.size();

  cout << "    volumes   " << ivolumes  << endl
       << "    surfaces  " << isurfaces << endl
       << "    assigned  " << iassigned << endl
       << "    unique    " << iunique << endl
       << "    iref_ass  " << iref_ass << endl
       << "    iref_nass " << iref_nass << endl;
}


void MagGeoBuilderFromDDD::build(const DDCompactView & cpva)
{
//    DDCompactView cpv;
  DDExpandedView fv(cpva);

  if (bldVerb::debugOut) cout << "**********************************************************" <<endl;

  // The actual field interpolators
  map<string, MagProviderInterpol*> bInterpolators;
  map<string, MagProviderInterpol*> eInterpolators;
  
  // Counter of different (FIXME can be removed)
  int bVolCount = 0;
  int eVolCount = 0;


  // Look for MAGF tree (any better way to find out???)
  //  fv.reset();

  if (fv.logicalPart().name().name()!="MAGF") {
     std::string topNodeName(fv.logicalPart().name().name());

     //see if one of the children is MAGF
     bool doSubDets = fv.firstChild();
     
     bool go=true;
     while(go&& doSubDets) {
	if (fv.logicalPart().name().name()=="MAGF")
	   break;
	else
	   go = fv.nextSibling();
     }
     if (!go) {
	throw cms::Exception("NoMAGFinDDD")<<" Neither he top node, nor any child node of the DDCompactView is \"MAGF\" but the top node is instead \""<<topNodeName<<"\"";
     }
  }
  // Loop over MAGF volumes and create volumeHandles. 
  if (bldVerb::debugOut) cout << endl << "*** In MAGF: " << endl;
  bool doSubDets = fv.firstChild();
  while (doSubDets){
    
    string name = fv.logicalPart().name().name();
    if (bldVerb::debugOut) cout << endl << "Name: " << name << endl
			       << "      " << fv.geoHistory() <<endl;

    // Build only the z-negative volumes, assuming symmetry
    // FIXME: should not use name but center...
    // even better, we should fix the XML!
    if (name.substr(2,2)=="ZP") {
      doSubDets = fv.nextSibling();
      continue;
    }
    
    bool mergeCylinders=true;

    // In the barrel, cylinders sectors will be skipped to build full 
    // cylinders out of sector copyno #1.
    // (these should be just volumes 1,2,4)
    bool expand = false;
    if (mergeCylinders) {
      if (name == "V_ZN_1"
	  || name == "V_ZN_2") {
	if (bldVerb::debugOut && fv.logicalPart().solid().shape()!=ddtubs) {
	  cout << "ERROR: MagGeoBuilderFromDDD::build: volume " << name
	       << " should be a cylinder" << endl;
	}
	if(fv.copyno()==1) {
	  // FIXME expand = true;
	} else {
	  //cout << "... to be skipped: "
	  //     << name << " " << fv.copyno() << endl;
	  //FIXME continue;
	}
      }
    }

    volumeHandle* v = new volumeHandle(fv, expand);
    
    // Select volumes, build volume handles.
    float Z = v->center().z();
    float R = v->center().perp();

    // Barrel is everything up to |Z| = 6610, excluding 
    // volume #7, centered at 6477.5
    // FIXME: misalignment?
    if (fabs(Z)<647. || (R>350. && fabs(Z)<662.)) { // Barrel
      if (bldVerb::debugOut) cout << " (Barrel)" <<endl;
      bVolumes.push_back(v);
      // Build the interpolator of the "master" volume (the one which is
      // not replicated, i.e. copy number #1)
      if (v->copyno==1) {
	buildInterpolator(v, bInterpolators);
	++bVolCount;
      }
    } else {               // Endcaps
      if (bldVerb::debugOut) cout << " (Endcaps)" <<endl;
      eVolumes.push_back(v);
      if (v->copyno==1) { 
	buildInterpolator(v, eInterpolators);
	++eVolCount;
      }
    }

    doSubDets = fv.nextSibling(); // end of loop over MAGF
  }
    
  if (bldVerb::debugOut) {
    cout << "Number of volumes (barrel): " << bVolumes.size() <<endl
		  << "Number of volumes (endcap): " << eVolumes.size() <<endl;
    cout << "**********************************************************" <<endl;
  }

  // Now all volumeHandles are there, and parameters for each of the planes
  // are calculated.

  //----------------------------------------------------------------------
  // Print summary information

  if (bldVerb::debugOut) {
    cout << "-----------------------" << endl;
    cout << "SUMMARY: Barrel " << endl;
    summary(bVolumes);
    
    cout << endl << "SUMMARY: Endcaps " << endl;
    summary(eVolumes);
    cout << "-----------------------" << endl;
  }


  //----------------------------------------------------------------------
  // Find barrel layers.

  vector<bLayer> layers; // the barrel layers
  precomputed_value_sort(bVolumes.begin(), bVolumes.end(), ExtractRN());

  // Find the layers (in R)
  const float resolution = 1.; // cm
  float rmin = bVolumes.front()->RN()-resolution;
  float rmax = bVolumes.back()->RN()+resolution;
  ClusterizingHistogram  hisR( int((rmax-rmin)/resolution) + 1, rmin, rmax);

  if (bldVerb::debugOut) cout << " R layers: " << rmin << " " << rmax << endl;

  handles::const_iterator first = bVolumes.begin();
  handles::const_iterator last = bVolumes.end();  

  for (handles::const_iterator i=first; i!=last; ++i){
    hisR.fill((*i)->RN());
  }
  vector<float> rClust = hisR.clusterize(resolution);

  handles::const_iterator ringStart = first;
  handles::const_iterator separ = first;

  for (unsigned int i=0; i<rClust.size() - 1; ++i) {
    if (bldVerb::debugOut) cout << " Layer at RN = " << rClust[i];
    float rSepar = (rClust[i] + rClust[i+1])/2.f;
    while ((*separ)->RN() < rSepar) ++separ;

    bLayer thislayer(ringStart, separ);
    layers.push_back(thislayer);
    ringStart = separ;
  }
  {
    if (bldVerb::debugOut) cout << " Layer at RN = " << rClust.back();
    bLayer thislayer(separ, last);
    layers.push_back(thislayer);
  }

  if (bldVerb::debugOut) cout << "Barrel: Found " << rClust.size() << " clusters in R, "
		  << layers.size() << " layers " << endl << endl;


  //----------------------------------------------------------------------
  // Find endcap sectors

  vector<eSector> sectors; // the endcap sectors
  precomputed_value_sort(eVolumes.begin(), eVolumes.end(), ExtractPhi()); 
 
  // ASSUMPTION: There are 12 sectors and each sector is 30 deg wide.
  for (int i = 0; i<12; ++i) {
    int offset = eVolumes.size()/12;
    //    int isec = (i+binOffset)%12;
    if (bldVerb::debugOut) cout << " Sector at phi = "
		    << (*(eVolumes.begin()+((i)*offset)))->center().phi()
		    << endl;
    sectors.push_back(eSector(eVolumes.begin()+((i)*offset),
			      eVolumes.begin()+((i+1)*offset)));
  }
   
  if (bldVerb::debugOut) cout << "Endcap: Found " 
		  << sectors.size() << " sectors " << endl;
 
  

  //----------------------------------------------------------------------  
  // Match surfaces.

//  cout << "------------------" << endl << "Now associating planes..." << endl;

//   // Loop on layers
//   for (vector<bLayer>::const_iterator ilay = layers.begin();
//        ilay!= layers.end(); ++ilay) {
//     cout << "On Layer: " << ilay-layers.begin() << " RN: " << (*ilay).RN()
// 	 <<endl;     

//     // Loop on wheels
//     for (vector<bWheel>::const_iterator iwheel = (*ilay).wheels.begin();
// 	 iwheel != (*ilay).wheels.end(); ++iwheel) {
//       cout << "  On Wheel: " << iwheel- (*ilay).wheels.begin()<< " Z: "
// 	   << (*iwheel).minZ() << " " << (*iwheel).maxZ() << " " 
// 	   << ((*iwheel).minZ()+(*iwheel).maxZ())/2. <<endl;

//       // Loop on sectors.
//       for (int isector = 0; isector<12; ++isector) {
// 	// FIXME: create new constructor...
// 	bSectorNavigator navy(layers,
// 			      ilay-layers.begin(),
// 			      iwheel-(*ilay).wheels.begin(),isector);
	
// 	const bSector & isect = (*iwheel).sector(isector);
	
// 	isect.matchPlanes(navy); //FIXME refcount
//       }
//     }
//   }


  //----------------------------------------------------------------------
  // Build MagVolumes and the MagGeometry hierarchy.

  //--- Barrel

  // Build MagVolumes and associate interpolators to them
  buildMagVolumes(bVolumes, bInterpolators);

  // Build MagBLayers
  for (vector<bLayer>::const_iterator ilay = layers.begin();
       ilay!= layers.end(); ++ilay) {
    mBLayers.push_back((*ilay).buildMagBLayer());
  }

  if (bldVerb::debugOut) {  
    cout << "*** BARREL ********************************************" << endl
	 << "Number of different volumes   = " << bVolCount << endl
	 << "Number of interpolators built = " << bInterpolators.size() << endl
    	 << "Number of MagBLayers built    = " << mBLayers.size() << endl;

    testInside(bVolumes); // Fixme: all volumes should be checked in one go.
  }
  
  //--- Endcap
  // Build MagVolumes  and associate interpolators to them
  buildMagVolumes(eVolumes, eInterpolators);

  // Build the MagESectors
  for (vector<eSector>::const_iterator isec = sectors.begin();
       isec!= sectors.end(); ++isec) {
    mESectors.push_back((*isec).buildMagESector());
  }

  if (bldVerb::debugOut) {
    cout << "*** ENDCAP ********************************************" << endl
	 << "Number of different volumes   = " << eVolCount << endl
	 << "Number of interpolators built = " << eInterpolators.size() << endl
    	 << "Number of MagESector built    = " << mESectors.size() << endl;

    testInside(eVolumes); // Fixme: all volumes should be checked in one go.
  }
}


void MagGeoBuilderFromDDD::buildMagVolumes(const handles & volumes, map<string, MagProviderInterpol*> & interpolators) {
  // Build all MagVolumes setting the MagProviderInterpol
  for (handles::const_iterator vol=volumes.begin(); vol!=volumes.end();
       ++vol){
    const MagProviderInterpol* mp = 0;
    if (interpolators.find((*vol)->magFile)!=interpolators.end()) {
      mp = interpolators[(*vol)->magFile];
    } else {
      cout << "No interpolator found for file " << (*vol)->magFile
	   << " vol: " << (*vol)->name << endl;
      cout << interpolators.size() <<endl;
      continue;
    }
      
    const GloballyPositioned<float> * gpos = (*vol)->placement();
    // FIXME check pos, rot corrsponds
    (*vol)->magVolume = new MagVolume6Faces(gpos->position(),
					    gpos->rotation(),
					    (*vol)->shape(),
					    (*vol)->sides(),
					    mp);

    // FIXME: bldVerb::debugOut, to be removed
    (*vol)->magVolume->name = (*vol)->name;  
  }
}


void MagGeoBuilderFromDDD::buildInterpolator(const volumeHandle * vol, map<string, MagProviderInterpol*> & interpolators){
  // Interpolators should be built only for volumes on NEGATIVE z 
  // (Z symmetry in field tables)
  if (vol->center().z()>0) return;

  // Remember: should build for volumes with negative Z only (Z simmetry)
  if (bldVerb::debugOut) cout << "Building interpolator from "
		  << vol->name << " copyno " << vol->copyno
		  << " at " << vol->center()
		  << " phi: " << vol->center().phi()
		  << " file: " << vol->magFile
		  << endl;
  
  if(bldVerb::debugOut && ( fabs(vol->center().phi() - Geom::pi()/2) > Geom::pi()/9.)){
    cout << "***WARNING wrong sector? " << endl;
  }


  // FIXME: should be a configurable parameter
  string version="grid_85l_030919"; 
  string fullPath;

  try {
    edm::FileInPath mydata("MagneticField/Interpolation/data/"+version+"/"+vol->magFile);
    fullPath = mydata.fullPath();
  } catch (edm::Exception& exc) {
    cerr << "MagGeoBuilderFromDDD: exception in reading table; " << exc.what() << endl;
    throw;
  }
  
  try{
    if (vol->toExpand()){
      //FIXME
//       interpolators[vol->magFile] =
// 	MFGridFactory::build( fullPath, *(vol->placement()), vol->minPhi(), vol->maxPhi());
    } else {
      interpolators[vol->magFile] =
	MFGridFactory::build( fullPath, *(vol->placement()));
    }
  } catch (MagException& exc) {
    cout << exc.what() << endl;
  }
}



void MagGeoBuilderFromDDD::testInside(handles & volumes) {
  // test inside() for all volumes.
  cout << "--------------------------------------------------" << endl;
  cout << " inside(center) test" << endl;
  for (handles::const_iterator vol=volumes.begin(); vol!=volumes.end();
       ++vol){
    for (handles::const_iterator i=volumes.begin(); i!=volumes.end();
	 ++i){
      if ((*i)==(*vol)) continue;
      //if ((*i)->magVolume == 0) continue;
      if ((*i)->magVolume->inside((*vol)->center())) {
	cout << "*** ERROR: center of " << (*vol)->name << " is inside " 
	     << (*i)->name <<endl;
      }
    }
    
    if ((*vol)->magVolume->inside((*vol)->center())) {
      cout << (*vol)->name << " OK " << endl;
    } else {
      cout << "*** ERROR: center of volume is not inside it, "
	   << (*vol)->name << endl;
    }
  }
  cout << "--------------------------------------------------" << endl;
}


vector<MagBLayer*> MagGeoBuilderFromDDD::barrelLayers() const{
  return mBLayers;
}

vector<MagESector*> MagGeoBuilderFromDDD::endcapSectors() const{
  return mESectors;
}

vector<MagVolume6Faces*> MagGeoBuilderFromDDD::barrelVolumes() const{
  vector<MagVolume6Faces*> v;
  v.reserve(bVolumes.size());
  for (handles::const_iterator i=bVolumes.begin();
       i!=bVolumes.end(); ++i){
    v.push_back((*i)->magVolume);
  }
  return v;
}

vector<MagVolume6Faces*> MagGeoBuilderFromDDD::endcapVolumes() const{
  vector<MagVolume6Faces*> v;
  v.reserve(eVolumes.size());
  for (handles::const_iterator i=eVolumes.begin();
       i!=eVolumes.end(); ++i){
    v.push_back((*i)->magVolume);
  }
  return v;
}
