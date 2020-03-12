/*
 *  See header file for a description of this class.
 *
 */

#include "DD4hep_MagGeoBuilder.h"
#include "bLayer.h"
#include "eSector.h"
#include "FakeInterpolator.h"

#include "MagneticField/Layers/interface/MagBLayer.h"
#include "MagneticField/Layers/interface/MagESector.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"

#include "Utilities/BinningTools/interface/ClusterizingHistogram.h"

#include "MagneticField/Interpolation/interface/MagProviderInterpol.h"
#include "MagneticField/Interpolation/interface/MFGridFactory.h"
#include "MagneticField/Interpolation/interface/MFGrid.h"

#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"
#include "MagneticField/VolumeGeometry/interface/MagExceptions.h"
#include "MagneticField/Layers/interface/MagVerbosity.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Utilities/General/interface/precomputed_value_sort.h"

#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <map>
#include <set>
#include <boost/algorithm/string/replace.hpp>

using namespace std;
using namespace magneticfield;
using namespace edm;
using namespace angle_units::operators;

MagGeoBuilder::MagGeoBuilder(string tableSet, int geometryVersion, bool debug)
    : tableSet_(tableSet), geometryVersion_(geometryVersion), theGridFiles_(nullptr), debug_(debug) {
  LogTrace("MagGeoBuilder") << "Constructing a MagGeoBuilder";
}

MagGeoBuilder::~MagGeoBuilder() {
  for (auto i : bVolumes_) {
    delete i;
  }
  for (auto i : eVolumes_) {
    delete i;
  }
}

void MagGeoBuilder::summary(handles& volumes) const {
  // The final countdown.
  int ivolumes = volumes.size();  // number of volumes
  int isurfaces = ivolumes * 6;   // number of individual surfaces
  int iassigned = 0;              // How many have been assigned
  int iunique = 0;                // number of unique surfaces
  int iref_ass = 0;
  int iref_nass = 0;

  set<const void*> ptrs;

  for (auto i : volumes) {
    DDSolidShape theShape = i->shape();
    if (theShape == DDSolidShape::ddbox || theShape == DDSolidShape::ddcons || theShape == DDSolidShape::ddtrap ||
        theShape == DDSolidShape::ddtubs) {
      for (int side = 0; side < 6; ++side) {
        int references = i->references(side);
        if (i->isPlaneMatched(side)) {
          ++iassigned;
          bool firstOcc = (ptrs.insert(&(i->surface(side)))).second;
          if (firstOcc)
            iref_ass += references;
          if (references < 2) {
            LogTrace("MagGeoBuilder") << "*** Only 1 ref, vol: " << i->volumeno << " # " << i->copyno
                                      << " side: " << side;
          }
        } else {
          iref_nass += references;
          if (references > 1) {
            LogTrace("MagGeoBuilder") << "*** Ref_nass >1 ";
          }
        }
      }
    }  // end if theShape
  }    // end for
  iunique = ptrs.size();

  LogTrace("MagGeoBuilder") << "    volumes   " << ivolumes << newln << "    surfaces  " << isurfaces << newln
                            << "    assigned  " << iassigned << newln << "    unique    " << iunique << newln
                            << "    iref_ass  " << iref_ass << newln << "    iref_nass " << iref_nass;
}

void MagGeoBuilder::build(const cms::DDDetector* det) {
  cms::Volume top = det->worldVolume();
  cms::DDFilteredView fv(det, top);
  if (fv.next(0) == false) {
    LogError("MagGeoBuilder") << "Filtered view is empty. Cannot build.";
    return;
  }

  // The actual field interpolators
  map<string, MagProviderInterpol*> bInterpolators;
  map<string, MagProviderInterpol*> eInterpolators;

  // Counter of different volumes
  int bVolCount = 0;
  int eVolCount = 0;

  const string magfStr{"MAGF"};
  const string magfStr2{"cmsMagneticField:MAGF"};
  if (fv.name() != magfStr && fv.name() != magfStr2) {
    std::string topNodeName(fv.name());
    LogTrace("MagGeoBuilder") << "Filtered view top node name is " << topNodeName << ".";

    //see if one of the children is MAGF
    bool doSubDets = fv.next(0);

    bool go = true;
    while (go && doSubDets) {
      LogTrace("MagGeoBuilder") << "Next node name is " << fv.name() << ".";
      if (fv.name() == magfStr)
        break;
      else
        go = fv.next(0);
    }
    if (!go) {
      throw cms::Exception("NoMAGF")
          << " Neither the top node, nor any child node of the filtered view is \"MAGF\" but the top node is instead \""
          << topNodeName << "\"";
    }
  }

  // Loop over MAGF volumes and create volumeHandles.
  bool doSubDets = fv.next(0);
  if (doSubDets == false) {
    LogError("MagGeoBuilder") << "Filtered view has no node. Cannot build.";
    return;
  }
  while (doSubDets) {
    string name = fv.volume().volume().name();
    LogTrace("MagGeoBuilder") << "Name: " << name;

    bool expand = false;
    volumeHandle* v = new volumeHandle(fv, expand, debug_);

    if (theGridFiles_ != nullptr) {
      int key = (v->volumeno) * 100 + v->copyno;
      TableFileMap::const_iterator itable = theGridFiles_->find(key);
      if (itable == theGridFiles_->end()) {
        key = (v->volumeno) * 100;
        itable = theGridFiles_->find(key);
      }

      if (itable != theGridFiles_->end()) {
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
        edm::LogError("MagGeoBuilderbuild") << "ERROR: no table spec found for V " << v->volumeno << ":" << v->copyno;
      }
    }

    // Select volumes, build volume handles.
    float Z = v->center().z();
    float R = v->center().perp();
    LogTrace("MagGeoBuilder") << " Vol R and Z values determine barrel or endcap. R = " << R << ", Z = " << Z;

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
      LogTrace("MagGeoBuilder") << " (Barrel)";
      bVolumes_.push_back(v);

      // Build the interpolator of the "master" volume (the one which is
      // not replicated in phi)
      // ASSUMPTION: copyno == sector.
      if (v->copyno == v->masterSector) {
        buildInterpolator(v, bInterpolators);
        ++bVolCount;
      }
    } else {  // Endcaps
      LogTrace("MagGeoBuilder") << " (Endcaps)";
      eVolumes_.push_back(v);
      if (v->copyno == v->masterSector) {
        buildInterpolator(v, eInterpolators);
        ++eVolCount;
      }
    }
    doSubDets = fv.next(0);  // end of loop over MAGF
  }

  LogTrace("MagGeoBuilder") << "Number of volumes (barrel): " << bVolumes_.size() << newln
                            << "Number of volumes (endcap): " << eVolumes_.size();
  LogTrace("MagGeoBuilder") << "**********************************************************";

  // Now all volumeHandles are there, and parameters for each of the planes
  // are calculated.

  //----------------------------------------------------------------------
  // Print summary information

  if (debug_) {
    LogTrace("MagGeoBuilder") << "-----------------------";
    LogTrace("MagGeoBuilder") << "SUMMARY: Barrel ";
    summary(bVolumes_);

    LogTrace("MagGeoBuilder") << "SUMMARY: Endcaps ";
    summary(eVolumes_);
    LogTrace("MagGeoBuilder") << "-----------------------";
  }

  //----------------------------------------------------------------------
  // Find barrel layers.

  if (bVolumes_.empty()) {
    LogError("MagGeoBuilder") << "Error: Barrel volumes are missing. Terminating build.";
    return;
  }
  vector<bLayer> layers;  // the barrel layers
  precomputed_value_sort(bVolumes_.begin(), bVolumes_.end(), ExtractRN());

  // Find the layers (in R)
  const float resolution = 1.;  // cm
  float rmin = bVolumes_.front()->RN() - resolution;
  float rmax = bVolumes_.back()->RN() + resolution;
  ClusterizingHistogram hisR(int((rmax - rmin) / resolution) + 1, rmin, rmax);

  LogTrace("MagGeoBuilder") << " R layers: " << rmin << " " << rmax;

  handles::const_iterator first = bVolumes_.begin();
  handles::const_iterator last = bVolumes_.end();

  for (auto i : bVolumes_) {
    hisR.fill(i->RN());
  }
  vector<float> rClust = hisR.clusterize(resolution);

  handles::const_iterator ringStart = first;
  handles::const_iterator separ = first;

  for (unsigned int i = 0; i < rClust.size() - 1; ++i) {
    if (debug_)
      LogTrace("MagGeoBuilder") << " Layer at RN = " << rClust[i];
    float rSepar = (rClust[i] + rClust[i + 1]) / 2.f;
    while ((*separ)->RN() < rSepar)
      ++separ;

    bLayer thislayer(ringStart, separ, debug_);
    layers.push_back(thislayer);
    ringStart = separ;
  }
  {
    if (debug_)
      LogTrace("MagGeoBuilder") << " Layer at RN = " << rClust.back();
    bLayer thislayer(separ, last, debug_);
    layers.push_back(thislayer);
  }

  LogTrace("MagGeoBuilder") << "Barrel: Found " << rClust.size() << " clusters in R, " << layers.size() << " layers ";

  //----------------------------------------------------------------------
  // Find endcap sectors
  vector<eSector> sectors;  // the endcap sectors

  // Find the number of sectors (should be 12 or 24 depending on the geometry model)
  constexpr float phireso = 0.05;  // rad
  constexpr int twoPiOverPhiReso = static_cast<int>(2._pi / phireso) + 1;
  ClusterizingHistogram hisPhi(twoPiOverPhiReso, -1._pi, 1._pi);

  for (auto i : eVolumes_) {
    hisPhi.fill(i->minPhi());
  }
  vector<float> phiClust = hisPhi.clusterize(phireso);
  int nESectors = phiClust.size();
  if (nESectors <= 0) {
    LogError("MagGeoBuilder") << "ERROR: Endcap sectors are missing.  Terminating build.";
    return;
  }
  if (debug_ && (nESectors % 12) != 0) {
    LogTrace("MagGeoBuilder") << "ERROR: unexpected # of endcap sectors: " << nESectors;
  }

  //Sort in phi
  precomputed_value_sort(eVolumes_.begin(), eVolumes_.end(), ExtractPhi());

  // Handle the -pi/pi boundary: volumes crossing it could be half at the begin and half at end of the sorted list.
  // So, check if any of the volumes that should belong to the first bin (at -phi) are at the end of the list:
  float lastBinPhi = phiClust.back();
  handles::reverse_iterator ri = eVolumes_.rbegin();
  while ((*ri)->center().phi() > lastBinPhi) {
    ++ri;
  }
  if (ri != eVolumes_.rbegin()) {
    // ri points to the first element that is within the last bin.
    // We need to move the following element (ie ri.base()) to the beginning of the list,
    handles::iterator newbeg = ri.base();
    rotate(eVolumes_.begin(), newbeg, eVolumes_.end());
  }

  //Group volumes in sectors
  int offset = eVolumes_.size() / nESectors;
  for (int i = 0; i < nESectors; ++i) {
    if (debug_) {
      LogTrace("MagGeoBuilder") << " Sector at phi = " << (*(eVolumes_.begin() + ((i)*offset)))->center().phi();
      // Additional x-check: sectors are expected to be made by volumes with the same copyno
      int secCopyNo = -1;
      for (handles::const_iterator iv = eVolumes_.begin() + ((i)*offset); iv != eVolumes_.begin() + ((i + 1) * offset);
           ++iv) {
        if (secCopyNo >= 0 && (*iv)->copyno != secCopyNo)
          LogTrace("MagGeoBuilder") << "ERROR: volume copyno " << (*iv)->name << ":" << (*iv)->copyno
                                    << " differs from others in same sectors with copyno = " << secCopyNo;
        secCopyNo = (*iv)->copyno;
      }
    }

    sectors.push_back(eSector(eVolumes_.begin() + ((i)*offset), eVolumes_.begin() + ((i + 1) * offset), debug_));
  }

  LogTrace("MagGeoBuilder") << "Endcap: Found " << sectors.size() << " sectors ";

  //----------------------------------------------------------------------
  // Build MagVolumes and the MagGeometry hierarchy.

  //--- Barrel

  // Build MagVolumes and associate interpolators to them
  buildMagVolumes(bVolumes_, bInterpolators);

  // Build MagBLayers
  for (auto ilay : layers) {
    mBLayers_.push_back(ilay.buildMagBLayer());
  }
  LogTrace("MagGeoBuilder") << "*** BARREL ********************************************" << newln
                            << "Number of different volumes   = " << bVolCount << newln
                            << "Number of interpolators built = " << bInterpolators.size() << newln
                            << "Number of MagBLayers built    = " << mBLayers_.size();
  if (debug_) {
    testInside(bVolumes_);  // FIXME: all volumes should be checked in one go.
  }
  //--- Endcap
  // Build MagVolumes  and associate interpolators to them
  buildMagVolumes(eVolumes_, eInterpolators);

  // Build the MagESectors
  for (auto isec : sectors) {
    mESectors_.push_back(isec.buildMagESector());
  }
  LogTrace("MagGeoBuilder") << "*** ENDCAP ********************************************" << newln
                            << "Number of different volumes   = " << eVolCount << newln
                            << "Number of interpolators built = " << eInterpolators.size() << newln
                            << "Number of MagESector built    = " << mESectors_.size();
  if (debug_) {
    testInside(eVolumes_);  // FIXME: all volumes should be checked in one go.
  }
}

void MagGeoBuilder::buildMagVolumes(const handles& volumes, map<string, MagProviderInterpol*>& interpolators) {
  // Build all MagVolumes setting the MagProviderInterpol
  for (auto vol : volumes) {
    const MagProviderInterpol* mp = nullptr;
    if (interpolators.find(vol->magFile) != interpolators.end()) {
      mp = interpolators[vol->magFile];
    } else {
      edm::LogError("MagGeoBuilder|buildMagVolumes")
          << "No interpolator found for file " << vol->magFile << " vol: " << vol->volumeno << "\n"
          << interpolators.size();
    }

    // Search for [volume,sector] in the list of scaling factors; sector = 0 handled as wildcard
    // ASSUMPTION: copyno == sector.
    int key = (vol->volumeno) * 100 + vol->copyno;
    map<int, double>::const_iterator isf = theScalingFactors_.find(key);
    if (isf == theScalingFactors_.end()) {
      key = (vol->volumeno) * 100;
      isf = theScalingFactors_.find(key);
    }

    double sf = 1.;
    if (isf != theScalingFactors_.end()) {
      sf = (*isf).second;

      LogTrace("MagGeoBuilder|buildMagVolumes") << "Applying scaling factor " << sf << " to " << vol->volumeno << "["
                                                << vol->copyno << "] (key:" << key << ")";
    }

    const GloballyPositioned<float>* gpos = vol->placement();
    vol->magVolume = new MagVolume6Faces(gpos->position(), gpos->rotation(), vol->sides(), mp, sf);

    if (vol->copyno == vol->masterSector) {
      vol->magVolume->ownsFieldProvider(true);
    }

    vol->magVolume->setIsIron(vol->isIron());

    // The name and sector of the volume are saved for debug purposes only. They may be removed at some point...
    vol->magVolume->volumeNo = vol->volumeno;
    vol->magVolume->copyno = vol->copyno;
  }
}

void MagGeoBuilder::buildInterpolator(const volumeHandle* vol, map<string, MagProviderInterpol*>& interpolators) {
  // Phi of the master sector
  double masterSectorPhi = (vol->masterSector - 1) * 1._pi / 6.;

  LogTrace("MagGeoBuilder") << "Building interpolator from " << vol->volumeno << " copyno " << vol->copyno << " at "
                            << vol->center() << " phi: " << static_cast<double>(vol->center().phi()) / 1._pi
                            << " pi,  file: " << vol->magFile << " master: " << vol->masterSector;
  if (debug_) {
    double delta = std::abs(vol->center().phi() - masterSectorPhi);
    if (delta > (1._pi / 9.)) {
      LogTrace("MagGeoBuilder") << "***WARNING wrong sector? Vol delta from master sector is " << delta / 1._pi
                                << " pi";
    }
  }

  if (tableSet_ == "fake" || vol->magFile == "fake") {
    interpolators[vol->magFile] = new magneticfield::FakeInterpolator();
    return;
  }

  string fullPath;

  try {
    edm::FileInPath mydata("MagneticField/Interpolation/data/" + tableSet_ + "/" + vol->magFile);
    fullPath = mydata.fullPath();
  } catch (edm::Exception& exc) {
    cerr << "MagGeoBuilder: exception in reading table; " << exc.what() << endl;
    if (!debug_)
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
    LogTrace("MagGeoBuilder") << exc.what();
    interpolators.erase(vol->magFile);
    if (!debug_)
      throw;
    return;
  }

  if (debug_) {
    // Check that all grid points of the interpolator are inside the volume.
    const MagVolume6Faces tempVolume(
        vol->placement()->position(), vol->placement()->rotation(), vol->sides(), interpolators[vol->magFile]);

    const MFGrid* grid = dynamic_cast<const MFGrid*>(interpolators[vol->magFile]);
    if (grid != nullptr) {
      Dimensions sizes = grid->dimensions();
      LogTrace("MagGeoBuilder") << "Grid has 3 dimensions "
                                << " number of nodes is " << sizes.w << " " << sizes.h << " " << sizes.d;

      const double tolerance = 0.03;

      size_t dumpCount = 0;
      for (int j = 0; j < sizes.h; j++) {
        for (int k = 0; k < sizes.d; k++) {
          for (int i = 0; i < sizes.w; i++) {
            MFGrid::LocalPoint lp = grid->nodePosition(i, j, k);
            if (!tempVolume.inside(lp, tolerance)) {
              if (++dumpCount < 2) {
                MFGrid::GlobalPoint gp = tempVolume.toGlobal(lp);
                LogTrace("MagGeoBuilder") << "GRID ERROR: " << i << " " << j << " " << k << " local: " << lp
                                          << " global: " << gp << " R= " << gp.perp() << " phi=" << gp.phi();
              }
            }
          }
        }
      }

      LogTrace("MagGeoBuilder") << "Volume:" << vol->volumeno
                                << " : Number of grid points outside the MagVolume: " << dumpCount << "/"
                                << sizes.w * sizes.h * sizes.d;
    }
  }
}

void MagGeoBuilder::testInside(handles& volumes) {
  // test inside() for all volumes.
  LogTrace("MagGeoBuilder") << "--------------------------------------------------";
  LogTrace("MagGeoBuilder") << " inside(center) test";
  for (auto vol : volumes) {
    for (auto i : volumes) {
      if (i == vol)
        continue;
      //if (i->magVolume == 0) continue;
      if (i->magVolume->inside(vol->center())) {
        LogTrace("MagGeoBuilder") << "*** ERROR: center of V " << vol->volumeno << ":" << vol->copyno << " is inside V "
                                  << i->volumeno << ":" << i->copyno;
      }
    }

    if (vol->magVolume->inside(vol->center())) {
      LogTrace("MagGeoBuilder") << "V " << vol->volumeno << ":" << vol->copyno << " OK ";
    } else {
      LogTrace("MagGeoBuilder") << "*** ERROR: center of volume is not inside it, " << vol->volumeno << ":"
                                << vol->copyno;
    }
  }
  LogTrace("MagGeoBuilder") << "--------------------------------------------------";
}

vector<MagBLayer*> MagGeoBuilder::barrelLayers() const { return mBLayers_; }

vector<MagESector*> MagGeoBuilder::endcapSectors() const { return mESectors_; }

vector<MagVolume6Faces*> MagGeoBuilder::barrelVolumes() const {
  vector<MagVolume6Faces*> v;
  v.reserve(bVolumes_.size());
  for (auto i : bVolumes_) {
    v.push_back(i->magVolume);
  }
  return v;
}

vector<MagVolume6Faces*> MagGeoBuilder::endcapVolumes() const {
  vector<MagVolume6Faces*> v;
  v.reserve(eVolumes_.size());
  for (auto i : eVolumes_) {
    v.push_back(i->magVolume);
  }
  return v;
}

float MagGeoBuilder::maxR() const {
  //FIXME: should get it from the actual geometry
  return 900.;
}

float MagGeoBuilder::maxZ() const {
  //FIXME: should get it from the actual geometry
  if (geometryVersion_ >= 160812)
    return 2400.;
  else if (geometryVersion_ >= 120812)
    return 2000.;
  else
    return 1600.;
}

void MagGeoBuilder::setScaling(const std::vector<int>& keys, const std::vector<double>& values) {
  if (keys.size() != values.size()) {
    throw cms::Exception("InvalidParameter")
        << "Invalid field scaling parameters 'scalingVolumes' and 'scalingFactors' ";
  }
  for (unsigned int i = 0; i < keys.size(); ++i) {
    theScalingFactors_[keys[i]] = values[i];
  }
}

void MagGeoBuilder::setGridFiles(const TableFileMap& gridFiles) { theGridFiles_ = &gridFiles; }
