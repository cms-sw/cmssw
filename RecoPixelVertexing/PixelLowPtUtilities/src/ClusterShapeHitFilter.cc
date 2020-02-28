#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"

#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"

#include <fstream>
#include <cassert>

using namespace std;

/*****************************************************************************/
ClusterShapeHitFilter::ClusterShapeHitFilter(const TrackerGeometry* theTracker_,
                                             const TrackerTopology* theTkTopol_,
                                             const MagneticField* theMagneticField_,
                                             const SiPixelLorentzAngle* theSiPixelLorentzAngle_,
                                             const SiStripLorentzAngle* theSiStripLorentzAngle_,
                                             const std::string& pixelShapeFile_,
                                             const std::string& pixelShapeFileL1_)
    : theTracker(theTracker_),
      theTkTopol(theTkTopol_),
      theMagneticField(theMagneticField_),
      theSiPixelLorentzAngle(theSiPixelLorentzAngle_),
      theSiStripLorentzAngle(theSiStripLorentzAngle_) {
  // Load pixel limits
  loadPixelLimits(pixelShapeFile_, pixelLimits);
  loadPixelLimits(pixelShapeFileL1_, pixelLimitsL1);
  fillPixelData();

  // Load strip limits
  loadStripLimits();
  fillStripData();
  cutOnPixelCharge_ = cutOnStripCharge_ = false;
  cutOnPixelShape_ = cutOnStripShape_ = true;
}

/*****************************************************************************/
ClusterShapeHitFilter::~ClusterShapeHitFilter() {}

/*****************************************************************************/
void ClusterShapeHitFilter::loadPixelLimits(std::string const& file, PixelLimits* plim) {
  edm::FileInPath fileInPath(file.c_str());
  ifstream inFile(fileInPath.fullPath().c_str());

  while (inFile.eof() == false) {
    int part, dx, dy;

    inFile >> part;  // 0or 1
    inFile >> dx;    // 0 to 10
    inFile >> dy;    // 0 to 15 ...

    const PixelKeys key(part, dx, dy);
    auto& pl = plim[key];

    for (int b = 0; b < 2; b++)      // branch
      for (int d = 0; d < 2; d++)    // direction
        for (int k = 0; k < 2; k++)  // lower and upper
          inFile >> pl.data[b][d][k];

    double f;
    int d;

    inFile >> f;  // density
    inFile >> d;  // points
    inFile >> f;  // density
    inFile >> d;  // points
  }

  inFile.close();

  LogTrace("MinBiasTracking|ClusterShapeHitFilter") << " [ClusterShapeHitFilter] pixel-cluster-shape filter loaded";
}

/*****************************************************************************/
void ClusterShapeHitFilter::loadStripLimits() {
  // Load strip
  edm::FileInPath fileInPath("RecoPixelVertexing/PixelLowPtUtilities/data/stripShape.par");
  ifstream inFile(fileInPath.fullPath().c_str());

  while (inFile.eof() == false) {
    int dx;
    inFile >> dx;

    StripKeys key(dx);
    auto& sl = stripLimits[key];

    for (int b = 0; b < 2; b++)    // branch
      for (int k = 0; k < 2; k++)  // lower and upper
        inFile >> sl.data[b][k];
  }

  inFile.close();

  LogTrace("MinBiasTracking|ClusterShapeHitFilter") << " [ClusterShapeHitFilter] strip-cluster-width filter loaded";
}

void ClusterShapeHitFilter::fillPixelData() {
  //barrel
  for (auto det : theTracker->detsPXB()) {
    // better not to fail..
    const PixelGeomDetUnit* pixelDet = dynamic_cast<const PixelGeomDetUnit*>(det);
    assert(pixelDet);
    PixelData& pd = pixelData[pixelDet->geographicalId()];
    pd.det = pixelDet;
    pd.part = 0;
    pd.layer = theTkTopol->pxbLayer(pixelDet->geographicalId());
    pd.cotangent = getCotangent(pixelDet);
    pd.drift = getDrift(pixelDet);
  }

  //endcap
  for (auto det : theTracker->detsPXF()) {
    // better not to fail..
    const PixelGeomDetUnit* pixelDet = dynamic_cast<const PixelGeomDetUnit*>(det);
    assert(pixelDet);
    PixelData& pd = pixelData[pixelDet->geographicalId()];
    pd.det = pixelDet;
    pd.part = 1;
    pd.layer = 0;
    pd.cotangent = getCotangent(pixelDet);
    pd.drift = getDrift(pixelDet);
  }
}

void ClusterShapeHitFilter::fillStripData() {
  // copied from StripCPE (FIXME maybe we should move all this in LocalReco)
  auto const& geom_ = *theTracker;
  auto const& dus = geom_.detUnits();
  auto offset = dus.size();
  for (unsigned int i = 1; i < 7; ++i) {
    if (geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i]) != dus.size() &&
        dus[geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i])]->type().isTrackerStrip()) {
      if (geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i]) < offset)
        offset = geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i]);
    }
  }

  for (auto i = offset; i != dus.size(); ++i) {
    const StripGeomDetUnit* stripdet = (const StripGeomDetUnit*)(dus[i]);
    assert(stripdet->index() == int(i));
    assert(stripdet->type().isTrackerStrip());  // not pixel
    auto const& bounds = stripdet->specificSurface().bounds();
    auto detid = stripdet->geographicalId();
    auto& p = stripData[detid];
    p.det = stripdet;
    p.topology = (StripTopology*)(&stripdet->topology());
    p.drift = getDrift(stripdet);
    p.thickness = bounds.thickness();
    p.nstrips = p.topology->nstrips();
  }
}

/*****************************************************************************/
pair<float, float> ClusterShapeHitFilter::getCotangent(const PixelGeomDetUnit* pixelDet) const {
  pair<float, float> cotangent;

  cotangent.first = pixelDet->surface().bounds().thickness() / pixelDet->specificTopology().pitch().first;
  cotangent.second = pixelDet->surface().bounds().thickness() / pixelDet->specificTopology().pitch().second;

  return cotangent;
}

/*****************************************************************************/
float ClusterShapeHitFilter::getCotangent(const StripData& sd, const LocalPoint& pos) const {
  // FIXME may be problematic in case of RadialStriptolopgy
  return sd.thickness / sd.topology->localPitch(pos);
}

/*****************************************************************************/
pair<float, float> ClusterShapeHitFilter::getDrift(const PixelGeomDetUnit* pixelDet) const {
  LocalVector lBfield = (pixelDet->surface()).toLocal(theMagneticField->inTesla(pixelDet->surface().position()));

  double theTanLorentzAnglePerTesla = theSiPixelLorentzAngle->getLorentzAngle(pixelDet->geographicalId().rawId());

  pair<float, float> dir;
  dir.first = -theTanLorentzAnglePerTesla * lBfield.y();
  dir.second = theTanLorentzAnglePerTesla * lBfield.x();

  return dir;
}

/*****************************************************************************/
float ClusterShapeHitFilter::getDrift(const StripGeomDetUnit* stripDet) const {
  LocalVector lBfield = (stripDet->surface()).toLocal(theMagneticField->inTesla(stripDet->surface().position()));

  double theTanLorentzAnglePerTesla = theSiStripLorentzAngle->getLorentzAngle(stripDet->geographicalId().rawId());

  float dir = theTanLorentzAnglePerTesla * lBfield.y();

  return dir;
}

/*****************************************************************************/
bool ClusterShapeHitFilter::isNormalOriented(const GeomDetUnit* geomDet) const {
  if (geomDet->type().isBarrel()) {  // barrel
    float perp0 = geomDet->toGlobal(Local3DPoint(0., 0., 0.)).perp();
    float perp1 = geomDet->toGlobal(Local3DPoint(0., 0., 1.)).perp();
    return (perp1 > perp0);
  } else {  // endcap
    float rot = geomDet->toGlobal(LocalVector(0., 0., 1.)).z();
    float pos = geomDet->toGlobal(Local3DPoint(0., 0., 0.)).z();
    return (rot * pos > 0);
  }
}

/*****************************************************************************/
/*****************************************************************************/

bool ClusterShapeHitFilter::getSizes(const SiPixelRecHit& recHit,
                                     const LocalVector& ldir,
                                     const SiPixelClusterShapeCache& clusterShapeCache,
                                     int& part,
                                     ClusterData::ArrayType& meas,
                                     pair<float, float>& pred,
                                     PixelData const* ipd) const {
  // Get detector
  const PixelData& pd = getpd(recHit, ipd);

  // Get shape information
  const SiPixelClusterShapeData& data = clusterShapeCache.get(recHit.cluster(), pd.det);
  bool usable = (data.isStraight() && data.isComplete());

  // Usable?
  //if(usable)
  {
    part = pd.part;

    // Predicted size
    pred.first = ldir.x() / ldir.z();
    pred.second = ldir.y() / ldir.z();

    SiPixelClusterShapeData::Range sizeRange = data.size();
    if (sizeRange.first->second < 0)
      pred.second = -pred.second;

    meas.clear();
    assert(meas.capacity() >= std::distance(sizeRange.first, sizeRange.second));
    for (auto s = sizeRange.first; s != sizeRange.second; ++s) {
      meas.push_back_unchecked(*s);
    }
    if (sizeRange.first->second < 0) {
      for (auto& s : meas)
        s.second = -s.second;
    }

    // Take out drift
    std::pair<float, float> const& drift = pd.drift;
    pred.first += drift.first;
    pred.second += drift.second;

    // Apply cotangent
    std::pair<float, float> const& cotangent = pd.cotangent;
    pred.first *= cotangent.first;
    pred.second *= cotangent.second;
  }

  // Usable?
  return usable;
}

bool ClusterShapeHitFilter::isCompatible(const SiPixelRecHit& recHit,
                                         const LocalVector& ldir,
                                         const SiPixelClusterShapeCache& clusterShapeCache,
                                         PixelData const* ipd) const {
  // Get detector
  if (cutOnPixelCharge_ && (!checkClusterCharge(recHit.geographicalId(), *(recHit.cluster()), ldir)))
    return false;
  if (!cutOnPixelShape_)
    return true;

  const PixelData& pd = getpd(recHit, ipd);

  int part;
  ClusterData::ArrayType meas;
  pair<float, float> pred;

  PixelLimits const* pl = pd.layer == 1 ? pixelLimitsL1 : pixelLimits;
  if (getSizes(recHit, ldir, clusterShapeCache, part, meas, pred, &pd)) {
    for (const auto& m : meas) {
      PixelKeys key(part, m.first, m.second);
      if (!key.isValid())
        return true;  // FIXME original logic
      if (pl[key].isInside(pred))
        return true;
    }
    // none of the choices worked
    return false;
  }
  // not usable
  return true;
}

bool ClusterShapeHitFilter::isCompatible(const SiPixelRecHit& recHit,
                                         const GlobalVector& gdir,
                                         const SiPixelClusterShapeCache& clusterShapeCache,
                                         PixelData const* ipd) const {
  // Get detector
  const PixelData& pd = getpd(recHit, ipd);

  LocalVector ldir = pd.det->toLocal(gdir);

  return isCompatible(recHit, ldir, clusterShapeCache, &pd);
}

/*****************************************************************************/
/*****************************************************************************/
bool ClusterShapeHitFilter::getSizes(DetId id,
                                     const SiStripCluster& cluster,
                                     const LocalPoint& lpos,
                                     const LocalVector& ldir,
                                     int& meas,
                                     float& pred) const {
  // Get detector
  auto const& p = getsd(id);

  // Measured width
  meas = cluster.amplitudes().size();

  // Usable?
  int fs = cluster.firstStrip();
  int ns = p.nstrips;
  // bool usable = (fs > 1 && fs + meas - 1 < ns);
  bool usable = (fs >= 1) & (fs + meas <= ns + 1);

  // Usable?
  //if(usable)
  {
    // Predicted width
    pred = ldir.x() / ldir.z();

    // Take out drift
    float drift = p.drift;
    ;
    pred += drift;

    // Apply cotangent
    pred *= getCotangent(p, lpos);
  }

  return usable;
}

/*****************************************************************************/
bool ClusterShapeHitFilter::isCompatible(DetId detId,
                                         const SiStripCluster& cluster,
                                         const LocalPoint& lpos,
                                         const LocalVector& ldir) const {
  int meas;
  float pred;

  if (cutOnStripCharge_ && (!checkClusterCharge(detId, cluster, ldir)))
    return false;
  if (!cutOnStripShape_)
    return true;

  if (getSizes(detId, cluster, lpos, ldir, meas, pred)) {
    StripKeys key(meas);
    if (key.isValid())
      return stripLimits[key].isInside(pred);
  }

  // Not usable or no limits
  return true;
}

/*****************************************************************************/
bool ClusterShapeHitFilter::isCompatible(DetId detId,
                                         const SiStripCluster& cluster,
                                         const GlobalPoint& gpos,
                                         const GlobalVector& gdir) const {
  const GeomDet* det = getsd(detId).det;
  LocalVector ldir = det->toLocal(gdir);
  LocalPoint lpos = det->toLocal(gpos);
  // now here we do the transformation
  lpos -= ldir * lpos.z() / ldir.z();
  return isCompatible(detId, cluster, lpos, ldir);
}
bool ClusterShapeHitFilter::isCompatible(DetId detId, const SiStripCluster& cluster, const GlobalVector& gdir) const {
  return isCompatible(detId, cluster, getsd(detId).det->toLocal(gdir));
}

#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"

bool ClusterShapeHitFilter::checkClusterCharge(DetId detId,
                                               const SiStripCluster& cluster,
                                               const LocalVector& ldir) const {
  return siStripClusterTools::chargePerCM(detId, cluster, ldir) > minGoodStripCharge_;
}

bool ClusterShapeHitFilter::checkClusterCharge(DetId detId,
                                               const SiPixelCluster& cluster,
                                               const LocalVector& ldir) const {
  return siStripClusterTools::chargePerCM(detId, cluster, ldir) > minGoodPixelCharge_;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(ClusterShapeHitFilter);
