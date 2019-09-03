/*
 *  See header file for a description of this class.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "MagGeoBuilderFromDDD.h"
#include "volumeHandle.h"
#include "bSlab.h"
#include "bRod.h"
#include "bSector.h"
#include "bLayer.h"
#include "eSector.h"
#include "eLayer.h"
#include "FakeInterpolator.h"

#include "MagneticField/Layers/interface/MagBLayer.h"
#include "MagneticField/Layers/interface/MagESector.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"

#include "Utilities/BinningTools/interface/ClusterizingHistogram.h"

#include "MagneticField/Interpolation/interface/MagProviderInterpol.h"
#include "MagneticField/Interpolation/interface/MFGridFactory.h"
#include "MagneticField/Interpolation/interface/MFGrid.h"

#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"
#include "MagneticField/VolumeGeometry/interface/MagExceptions.h"
#include "MagneticField/Layers/interface/MagVerbosity.h"

#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DataFormats/Math/interface/GeantUnits.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <map>
#include <set>
#include <iomanip>
#include <boost/algorithm/string/replace.hpp>
#include "Utilities/General/interface/precomputed_value_sort.h"

using namespace std;
using namespace magneticfield;

MagGeoBuilderFromDDD::MagGeoBuilderFromDDD(string tableSet_, int geometryVersion_, bool debug_)
    : tableSet(tableSet_), geometryVersion(geometryVersion_), theGridFiles(nullptr), debug(debug_) {
  if (debug)
    cout << "Constructing a MagGeoBuilderFromDDD" << endl;
}

MagGeoBuilderFromDDD::~MagGeoBuilderFromDDD() {
  for (handles::const_iterator i = bVolumes.begin(); i != bVolumes.end(); ++i) {
    delete (*i);
  }

  for (handles::const_iterator i = eVolumes.begin(); i != eVolumes.end(); ++i) {
    delete (*i);
  }
}

void MagGeoBuilderFromDDD::summary(handles& volumes) {
  // The final countdown.
  int ivolumes = volumes.size();  // number of volumes
  int isurfaces = ivolumes * 6;   // number of individual surfaces
  int iassigned = 0;              // How many have been assigned
  int iunique = 0;                // number of unique surfaces
  int iref_ass = 0;
  int iref_nass = 0;

  set<const void*> ptrs;

  handles::const_iterator first = volumes.begin();
  handles::const_iterator last = volumes.end();

  for (handles::const_iterator i = first; i != last; ++i) {
    if (int((*i)->shape()) > 4)
      continue;  // FIXME: implement test for missing shapes...
    for (int side = 0; side < 6; ++side) {
      int references = (*i)->references(side);
      if ((*i)->isPlaneMatched(side)) {
        ++iassigned;
        bool firstOcc = (ptrs.insert(&((*i)->surface(side)))).second;
        if (firstOcc)
          iref_ass += references;
        if (references < 2) {
          cout << "*** Only 1 ref, vol: " << (*i)->volumeno << " # " << (*i)->copyno << " side: " << side << endl;
        }
      } else {
        iref_nass += references;
        if (references > 1) {
          cout << "*** Ref_nass >1 " << endl;
        }
      }
    }
  }
  iunique = ptrs.size();

  cout << "    volumes   " << ivolumes << endl
       << "    surfaces  " << isurfaces << endl
       << "    assigned  " << iassigned << endl
       << "    unique    " << iunique << endl
       << "    iref_ass  " << iref_ass << endl
       << "    iref_nass " << iref_nass << endl;
}

void MagGeoBuilderFromDDD::build(const DDCompactView& cpva) {
  //    DDCompactView cpv;
  DDExpandedView fv(cpva);

  if (debug)
    cout << "**********************************************************" << endl;

  // The actual field interpolators
  map<string, MagProviderInterpol*> bInterpolators;
  map<string, MagProviderInterpol*> eInterpolators;

  // Counter of different volumes
  int bVolCount = 0;
  int eVolCount = 0;

  if (fv.logicalPart().name().name() != "MAGF") {
    std::string topNodeName(fv.logicalPart().name().name());

    //see if one of the children is MAGF
    bool doSubDets = fv.firstChild();

    bool go = true;
    while (go && doSubDets) {
      if (fv.logicalPart().name().name() == "MAGF")
        break;
      else
        go = fv.nextSibling();
    }
    if (!go) {
      throw cms::Exception("NoMAGFinDDD")
          << " Neither the top node, nor any child node of the DDCompactView is \"MAGF\" but the top node is instead \""
          << topNodeName << "\"";
    }
  }
  // Loop over MAGF volumes and create volumeHandles.
  if (debug) {
    cout << endl
         << "*** MAGF: " << fv.geoHistory() << endl
         << "translation: " << fv.translation() << endl
         << " rotation: " << fv.rotation() << endl;
  }

  bool doSubDets = fv.firstChild();
  while (doSubDets) {
    string name = fv.logicalPart().name().name();
    if (debug)
      cout << endl << "Name: " << name << endl << "      " << fv.geoHistory() << endl;

    // FIXME: single-volyme cylinders - this is feature has been removed
    //        and should be revisited.
    //    bool mergeCylinders=false;

    // If true, In the barrel, cylinders sectors will be skipped to build full
    // cylinders out of sector copyno #1.
    bool expand = false;

    //     if (mergeCylinders) {
    //       if (name == "V_ZN_1"
    // 	  || name == "V_ZN_2") {
    // 	if (debug && fv.logicalPart().solid().shape()!=ddtubs) {
    // 	  cout << "ERROR: MagGeoBuilderFromDDD::build: volume " << name
    // 	       << " should be a cylinder" << endl;
    // 	}
    // 	if(fv.copyno()==1) {
    // 	  expand = true;
    // 	} else {
    // 	  //cout << "... to be skipped: "
    // 	  //     << name << " " << fv.copyno() << endl;
    // 	}
    //       }
    //     }

    volumeHandle* v = new volumeHandle(fv, expand);

    if (theGridFiles != nullptr) {
      int key = (v->volumeno) * 100 + v->copyno;
      TableFileMap::const_iterator itable = theGridFiles->find(key);
      if (itable == theGridFiles->end()) {
        key = (v->volumeno) * 100;
        itable = theGridFiles->find(key);
      }

      if (itable != theGridFiles->end()) {
        string magFile = (*itable).second.first;
        stringstream conv;
        string svol, ssec;
        conv << setfill('0') << setw(3) << v->volumeno << " " << setw(2)
             << v->copyno;  // volume assumed to have 0s padding to 3 digits; sector assumed to have 0s padding to 2 digits
        conv >> svol >> ssec;
        boost::replace_all(magFile, "[v]", svol);
        boost::replace_all(magFile, "[s]", ssec);
        int masterSector = (*itable).second.second;
        if (masterSector == 0)
          masterSector = v->copyno;
        v->magFile = magFile;
        v->masterSector = masterSector;
      } else {
        edm::LogError("MagGeoBuilderFromDDDbuild")
            << "ERROR: no table spec found for V " << v->volumeno << ":" << v->copyno;
      }
    }

    // Select volumes, build volume handles.
    float Z = v->center().z();
    float R = v->center().perp();

    // v 85l: Barrel is everything up to |Z| = 661.0, excluding
    // volume #7, centered at 6477.5
    // v 1103l: same numbers work fine. #16 instead of #7, same coords;
    // see comment below for V6,7
    //ASSUMPTION: no misalignment is applied to mag volumes.
    //FIXME: implement barrel/endcap flags as DDD SpecPars.
    if ((fabs(Z) < 647. || (R > 350. && fabs(Z) < 662.)) &&
        !(fabs(Z) > 480 && R < 172)  // in 1103l we place V_6 and V_7 in the
                                     // endcaps to preserve nice layer structure
                                     // in the barrel. This does not hurt in v85l
                                     // where there is a single V1
    ) {                              // Barrel
      if (debug)
        cout << " (Barrel)" << endl;
      bVolumes.push_back(v);

      // Build the interpolator of the "master" volume (the one which is
      // not replicated in phi)
      // ASSUMPTION: copyno == sector.
      if (v->copyno == v->masterSector) {
        buildInterpolator(v, bInterpolators);
        ++bVolCount;
      }
    } else {  // Endcaps
      if (debug)
        cout << " (Endcaps)" << endl;
      eVolumes.push_back(v);
      if (v->copyno == v->masterSector) {
        buildInterpolator(v, eInterpolators);
        ++eVolCount;
      }
    }

    doSubDets = fv.nextSibling();  // end of loop over MAGF
  }

  if (debug) {
    cout << "Number of volumes (barrel): " << bVolumes.size() << endl
         << "Number of volumes (endcap): " << eVolumes.size() << endl;
    cout << "**********************************************************" << endl;
  }

  // Now all volumeHandles are there, and parameters for each of the planes
  // are calculated.

  //----------------------------------------------------------------------
  // Print summary information

  if (debug) {
    cout << "-----------------------" << endl;
    cout << "SUMMARY: Barrel " << endl;
    summary(bVolumes);

    cout << endl << "SUMMARY: Endcaps " << endl;
    summary(eVolumes);
    cout << "-----------------------" << endl;
  }

  //----------------------------------------------------------------------
  // Find barrel layers.

  vector<bLayer> layers;  // the barrel layers
  precomputed_value_sort(bVolumes.begin(), bVolumes.end(), ExtractRN());

  // Find the layers (in R)
  const float resolution = 1.;  // cm
  float rmin = bVolumes.front()->RN() - resolution;
  float rmax = bVolumes.back()->RN() + resolution;
  ClusterizingHistogram hisR(int((rmax - rmin) / resolution) + 1, rmin, rmax);

  if (debug)
    cout << " R layers: " << rmin << " " << rmax << endl;

  handles::const_iterator first = bVolumes.begin();
  handles::const_iterator last = bVolumes.end();

  for (handles::const_iterator i = first; i != last; ++i) {
    hisR.fill((*i)->RN());
  }
  vector<float> rClust = hisR.clusterize(resolution);

  handles::const_iterator ringStart = first;
  handles::const_iterator separ = first;

  for (unsigned int i = 0; i < rClust.size() - 1; ++i) {
    if (debug)
      cout << " Layer at RN = " << rClust[i];
    float rSepar = (rClust[i] + rClust[i + 1]) / 2.f;
    while ((*separ)->RN() < rSepar)
      ++separ;

    bLayer thislayer(ringStart, separ, debug);
    layers.push_back(thislayer);
    ringStart = separ;
  }
  {
    if (debug)
      cout << " Layer at RN = " << rClust.back();
    bLayer thislayer(separ, last, debug);
    layers.push_back(thislayer);
  }

  if (debug)
    cout << "Barrel: Found " << rClust.size() << " clusters in R, " << layers.size() << " layers " << endl << endl;

  //----------------------------------------------------------------------
  // Find endcap sectors
  vector<eSector> sectors;  // the endcap sectors

  // Find the number of sectors (should be 12 or 24 depending on the geometry model)
  float phireso = 0.05;  // rad
  ClusterizingHistogram hisPhi(int((Geom::ftwoPi()) / phireso) + 1, -Geom::fpi(), Geom::fpi());

  for (handles::const_iterator i = eVolumes.begin(); i != eVolumes.end(); ++i) {
    hisPhi.fill((*i)->minPhi());
  }
  vector<float> phiClust = hisPhi.clusterize(phireso);
  int nESectors = phiClust.size();
  if (debug && (nESectors % 12) != 0)
    cout << "ERROR: unexpected # of sectors: " << nESectors << endl;

  //Sort in phi
  precomputed_value_sort(eVolumes.begin(), eVolumes.end(), ExtractPhi());

  // Handle the -pi/pi boundary: volumes crossing it could be half at the begin and half at end of the sorted list.
  // So, check if any of the volumes that should belong to the first bin (at -phi) are at the end of the list:
  float lastBinPhi = phiClust.back();
  handles::reverse_iterator ri = eVolumes.rbegin();
  while ((*ri)->center().phi() > lastBinPhi) {
    ++ri;
  }
  if (ri != eVolumes.rbegin()) {
    // ri points to the first element that is within the last bin.
    // We need to move the following element (ie ri.base()) to the beginning of the list,
    handles::iterator newbeg = ri.base();
    rotate(eVolumes.begin(), newbeg, eVolumes.end());
  }

  //Group volumes in sectors
  int offset = eVolumes.size() / nESectors;
  for (int i = 0; i < nESectors; ++i) {
    if (debug) {
      cout << " Sector at phi = " << (*(eVolumes.begin() + ((i)*offset)))->center().phi() << endl;
      // Additional x-check: sectors are expected to be made by volumes with the same copyno
      int secCopyNo = -1;
      for (handles::const_iterator iv = eVolumes.begin() + ((i)*offset); iv != eVolumes.begin() + ((i + 1) * offset);
           ++iv) {
        if (secCopyNo >= 0 && (*iv)->copyno != secCopyNo)
          cout << "ERROR: volume copyno" << (*iv)->name << ":" << (*iv)->copyno
               << " differs from others in same sectors " << secCopyNo << endl;
        secCopyNo = (*iv)->copyno;
      }
    }

    sectors.push_back(eSector(eVolumes.begin() + ((i)*offset), eVolumes.begin() + ((i + 1) * offset), debug));
  }

  if (debug)
    cout << "Endcap: Found " << sectors.size() << " sectors " << endl;

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
  for (vector<bLayer>::const_iterator ilay = layers.begin(); ilay != layers.end(); ++ilay) {
    mBLayers.push_back((*ilay).buildMagBLayer());
  }

  if (debug) {
    cout << "*** BARREL ********************************************" << endl
         << "Number of different volumes   = " << bVolCount << endl
         << "Number of interpolators built = " << bInterpolators.size() << endl
         << "Number of MagBLayers built    = " << mBLayers.size() << endl;

    testInside(bVolumes);  // FIXME: all volumes should be checked in one go.
  }

  //--- Endcap
  // Build MagVolumes  and associate interpolators to them
  buildMagVolumes(eVolumes, eInterpolators);

  // Build the MagESectors
  for (vector<eSector>::const_iterator isec = sectors.begin(); isec != sectors.end(); ++isec) {
    mESectors.push_back((*isec).buildMagESector());
  }

  if (debug) {
    cout << "*** ENDCAP ********************************************" << endl
         << "Number of different volumes   = " << eVolCount << endl
         << "Number of interpolators built = " << eInterpolators.size() << endl
         << "Number of MagESector built    = " << mESectors.size() << endl;

    testInside(eVolumes);  // FIXME: all volumes should be checked in one go.
  }
}

void MagGeoBuilderFromDDD::buildMagVolumes(const handles& volumes, map<string, MagProviderInterpol*>& interpolators) {
  // Build all MagVolumes setting the MagProviderInterpol
  for (handles::const_iterator vol = volumes.begin(); vol != volumes.end(); ++vol) {
    const MagProviderInterpol* mp = nullptr;
    if (interpolators.find((*vol)->magFile) != interpolators.end()) {
      mp = interpolators[(*vol)->magFile];
    } else {
      edm::LogError("MagGeoBuilderFromDDDbuildMagVolumes")
          << "No interpolator found for file " << (*vol)->magFile << " vol: " << (*vol)->volumeno << "\n"
          << interpolators.size() << endl;
    }

    // Search for [volume,sector] in the list of scaling factors; sector = 0 handled as wildcard
    // ASSUMPTION: copyno == sector.
    int key = ((*vol)->volumeno) * 100 + (*vol)->copyno;
    map<int, double>::const_iterator isf = theScalingFactors.find(key);
    if (isf == theScalingFactors.end()) {
      key = ((*vol)->volumeno) * 100;
      isf = theScalingFactors.find(key);
    }

    double sf = 1.;
    if (isf != theScalingFactors.end()) {
      sf = (*isf).second;

      edm::LogInfo("MagneticField|VolumeBasedMagneticFieldESProducer")
          << "Applying scaling factor " << sf << " to " << (*vol)->volumeno << "[" << (*vol)->copyno << "] (key:" << key
          << ")" << endl;
    }

    const GloballyPositioned<float>* gpos = (*vol)->placement();
    (*vol)->magVolume = new MagVolume6Faces(gpos->position(), gpos->rotation(), (*vol)->sides(), mp, sf);

    if ((*vol)->copyno == (*vol)->masterSector) {
      (*vol)->magVolume->ownsFieldProvider(true);
    }

    (*vol)->magVolume->setIsIron((*vol)->isIron());

    // The name and sector of the volume are saved for debug purposes only. They may be removed at some point...
    (*vol)->magVolume->volumeNo = (*vol)->volumeno;
    (*vol)->magVolume->copyno = (*vol)->copyno;
  }
}

void MagGeoBuilderFromDDD::buildInterpolator(const volumeHandle* vol,
                                             map<string, MagProviderInterpol*>& interpolators) {
  // Phi of the master sector
  double masterSectorPhi = (vol->masterSector - 1) * Geom::pi() / 6.;

  if (debug) {
    cout << "Building interpolator from " << vol->volumeno << " copyno " << vol->copyno << " at " << vol->center()
         << " phi: " << vol->center().phi() << " file: " << vol->magFile << " master : " << vol->masterSector << endl;

    if (fabs(vol->center().phi() - masterSectorPhi) > Geom::pi() / 9.) {
      cout << "***WARNING wrong sector? " << endl;
    }
  }

  if (tableSet == "fake" || vol->magFile == "fake") {
    interpolators[vol->magFile] = new magneticfield::FakeInterpolator();
    return;
  }

  string fullPath;

  try {
    edm::FileInPath mydata("MagneticField/Interpolation/data/" + tableSet + "/" + vol->magFile);
    fullPath = mydata.fullPath();
  } catch (edm::Exception& exc) {
    cerr << "MagGeoBuilderFromDDD: exception in reading table; " << exc.what() << endl;
    if (!debug)
      throw;
    return;
  }

  try {
    if (vol->toExpand()) {
      //FIXME: see discussion on mergeCylinders above.
      //       interpolators[vol->magFile] =
      // 	MFGridFactory::build( fullPath, *(vol->placement()), vol->minPhi(), vol->maxPhi());
    } else {
      // If the table is in "local" coordinates, must create a reference
      // frame that is appropriately rotated along the CMS Z axis.

      GloballyPositioned<float> rf = *(vol->placement());

      if (vol->masterSector != 1) {
        typedef Basic3DVector<float> Vector;

        GloballyPositioned<float>::RotationType rot(Vector(0, 0, 1), -masterSectorPhi);
        Vector vpos(vol->placement()->position());

        rf = GloballyPositioned<float>(GloballyPositioned<float>::PositionType(rot.multiplyInverse(vpos)),
                                       vol->placement()->rotation() * rot);
      }

      interpolators[vol->magFile] = MFGridFactory::build(fullPath, rf);
    }
  } catch (MagException& exc) {
    cout << exc.what() << endl;
    interpolators.erase(vol->magFile);
    if (!debug)
      throw;
    return;
  }

  if (debug) {
    // Check that all grid points of the interpolator are inside the volume.
    const MagVolume6Faces tempVolume(
        vol->placement()->position(), vol->placement()->rotation(), vol->sides(), interpolators[vol->magFile]);

    const MFGrid* grid = dynamic_cast<const MFGrid*>(interpolators[vol->magFile]);
    if (grid != nullptr) {
      Dimensions sizes = grid->dimensions();
      cout << "Grid has 3 dimensions "
           << " number of nodes is " << sizes.w << " " << sizes.h << " " << sizes.d << endl;

      const double tolerance = 0.03;

      size_t dumpCount = 0;
      for (int j = 0; j < sizes.h; j++) {
        for (int k = 0; k < sizes.d; k++) {
          for (int i = 0; i < sizes.w; i++) {
            MFGrid::LocalPoint lp = grid->nodePosition(i, j, k);
            if (!tempVolume.inside(lp, tolerance)) {
              if (++dumpCount < 2) {
                MFGrid::GlobalPoint gp = tempVolume.toGlobal(lp);
                cout << "GRID ERROR: " << i << " " << j << " " << k << " local: " << lp << " global: " << gp
                     << " R= " << gp.perp() << " phi=" << gp.phi() << endl;
              }
            }
          }
        }
      }

      cout << "Volume:" << vol->volumeno << " : Number of grid points outside the MagVolume: " << dumpCount << "/"
           << sizes.w * sizes.h * sizes.d << endl;
    }
  }
}

void MagGeoBuilderFromDDD::testInside(handles& volumes) {
  // test inside() for all volumes.
  cout << "--------------------------------------------------" << endl;
  cout << " inside(center) test" << endl;
  for (handles::const_iterator vol = volumes.begin(); vol != volumes.end(); ++vol) {
    for (handles::const_iterator i = volumes.begin(); i != volumes.end(); ++i) {
      if ((*i) == (*vol))
        continue;
      //if ((*i)->magVolume == 0) continue;
      if ((*i)->magVolume->inside((*vol)->center())) {
        cout << "*** ERROR: center of V " << (*vol)->volumeno << ":" << (*vol)->copyno << " is inside V "
             << (*i)->volumeno << ":" << (*i)->copyno << endl;
      }
    }

    if ((*vol)->magVolume->inside((*vol)->center())) {
      cout << "V " << (*vol)->volumeno << " OK " << endl;
    } else {
      cout << "*** ERROR: center of volume is not inside it, " << (*vol)->volumeno << endl;
    }
  }
  cout << "--------------------------------------------------" << endl;
}

vector<MagBLayer*> MagGeoBuilderFromDDD::barrelLayers() const { return mBLayers; }

vector<MagESector*> MagGeoBuilderFromDDD::endcapSectors() const { return mESectors; }

vector<MagVolume6Faces*> MagGeoBuilderFromDDD::barrelVolumes() const {
  vector<MagVolume6Faces*> v;
  v.reserve(bVolumes.size());
  for (handles::const_iterator i = bVolumes.begin(); i != bVolumes.end(); ++i) {
    v.push_back((*i)->magVolume);
  }
  return v;
}

vector<MagVolume6Faces*> MagGeoBuilderFromDDD::endcapVolumes() const {
  vector<MagVolume6Faces*> v;
  v.reserve(eVolumes.size());
  for (handles::const_iterator i = eVolumes.begin(); i != eVolumes.end(); ++i) {
    v.push_back((*i)->magVolume);
  }
  return v;
}

float MagGeoBuilderFromDDD::maxR() const {
  //FIXME: should get it from the actual geometry
  return 900.;
}

float MagGeoBuilderFromDDD::maxZ() const {
  //FIXME: should get it from the actual geometry
  if (geometryVersion >= 160812)
    return 2400.;
  else if (geometryVersion >= 120812)
    return 2000.;
  else
    return 1600.;
}

void MagGeoBuilderFromDDD::setScaling(const std::vector<int>& keys, const std::vector<double>& values) {
  if (keys.size() != values.size()) {
    throw cms::Exception("InvalidParameter")
        << "Invalid field scaling parameters 'scalingVolumes' and 'scalingFactors' ";
  }
  for (unsigned int i = 0; i < keys.size(); ++i) {
    theScalingFactors[keys[i]] = values[i];
  }
}

void MagGeoBuilderFromDDD::setGridFiles(const TableFileMap& gridFiles) { theGridFiles = &gridFiles; }
