#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/HitInfo.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShape.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterData.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"

#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"

#include <fstream>
using namespace std;

/*****************************************************************************/
// singleton begin
ClusterShapeHitFilter * ClusterShapeHitFilter::_instance = 0;
int ClusterShapeHitFilter::_refCount = 0;

ClusterShapeHitFilter * ClusterShapeHitFilter::Instance
  (const edm::EventSetup& es, const string & caller)
{
  if( _instance == 0 )
  {
    _instance = new ClusterShapeHitFilter(es);
    LogTrace("MinBiasTracking")
      << " [ClusterShapeHitFilter] creating instance for " << caller;
  }

  _refCount++;

  LogTrace("MinBiasTracking")
    << " [ClusterShapeHitFilter] referencing instance for " << caller
    << " [" << _refCount << "]" ;

  return _instance;
}
 
void ClusterShapeHitFilter::Release()
{
  LogTrace("MinBiasTracking")
    << " [ClusterShapeHitFilter] releasing instance"
    << " [" << _refCount << "]" ;

  if( --_refCount < 1 ) Destroy();
}
 
void ClusterShapeHitFilter::Destroy()
{
  if( _instance != 0 )
  {
    LogTrace("MinBiasTracking")
      << " [ClusterShapeHitFilter] deleting instance";

    delete( _instance );
    _instance = 0;
  }
}
// singleton end

/*****************************************************************************/
ClusterShapeHitFilter::ClusterShapeHitFilter
  (const edm::EventSetup& es)
{ // called from ClusterShapeTrackFilter, TripletFilter, ClusterShapeExtractor

  // Get tracker geometry
  edm::ESHandle<GlobalTrackingGeometry> tracker;
  es.get<GlobalTrackingGeometryRecord>().get(tracker);
  theTracker = tracker.product();

  // Get magnetic field
  edm::ESHandle<MagneticField> field;
  es.get<IdealMagneticFieldRecord>().get(field);
  theMagneticField = field.product();

  // Get Lorentz angle for pixels
  edm::ESHandle<SiPixelLorentzAngle>   pixel;
  es.get<SiPixelLorentzAngleRcd>().get(pixel);
  theSiPixelLorentzAngle =             pixel.product();

  // Get Lorentz angle for strips
  edm::ESHandle<SiStripLorentzAngle>   strip;
  es.get<SiStripLorentzAngleRcd>().get(strip);
  theSiStripLorentzAngle =             strip.product();

  // Set to zero
  for(int i = GeomDetEnumerators::PixelBarrel;
          i<= GeomDetEnumerators::TEC; i++)
    theAngle[i] = 0;

  // Load pixel limits
  loadPixelLimits();

  // Load strip limits
  loadStripLimits();
}

/*****************************************************************************/
ClusterShapeHitFilter::ClusterShapeHitFilter
  (const GlobalTrackingGeometry * theTracker_,
   const MagneticField * theMagneticField_)
   : theTracker(theTracker_),
     theMagneticField(theMagneticField_)
{ // called from ClusterShapeTrajectoryFilter

  // Hardwired numbers, since no access to Lorentz
  theAngle[GeomDetEnumerators::PixelBarrel] = 0.106;
  theAngle[GeomDetEnumerators::PixelEndcap] = 0.0912;

  theAngle[GeomDetEnumerators::TIB] = 0.032;
  theAngle[GeomDetEnumerators::TOB] = 0.032;
  theAngle[GeomDetEnumerators::TID] = 0.032;
  theAngle[GeomDetEnumerators::TEC] = 0.032;

  // Load pixel limits
  loadPixelLimits();

  // Load strip limits
  loadStripLimits();
}

/*****************************************************************************/
ClusterShapeHitFilter::~ClusterShapeHitFilter()
{
}

/*****************************************************************************/
void ClusterShapeHitFilter::loadPixelLimits()
{
  edm::FileInPath
    fileInPath("RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape.par");
  ifstream inFile(fileInPath.fullPath().c_str());

                vector<float>     v1(2,0);
         vector<vector<float> >   v2(2,v1);
  vector<vector<vector<float> > > v3(2,v2);

  while(inFile.eof() == false)
  {
    int part,dx,dy;

    inFile >> part;
    inFile >> dx;
    inFile >> dy;

    for(int b = 0; b<2 ; b++) // branch
    for(int d = 0; d<2 ; d++) // direction
    for(int k = 0; k<2 ; k++) // lower and upper
      inFile >> v3[b][d][k];

    PixelKeys key(part,dx,dy);
    pixelLimits[key] = v3;

    double f;
    int d;

    inFile >> f; // density
    inFile >> d; // points
    inFile >> f; // density
    inFile >> d; // points
  }
  
  inFile.close();
  
  LogTrace("MinBiasTracking")
    << " [ClusterShapeHitFilter] pixel-cluster-shape filter loaded";
 }

/*****************************************************************************/
void ClusterShapeHitFilter::loadStripLimits()
{
  // Load strip
  edm::FileInPath
    fileInPath("RecoPixelVertexing/PixelLowPtUtilities/data/stripShape.par");
  ifstream inFile(fileInPath.fullPath().c_str());

           vector<float> v1(2,0);
  vector<vector<float> > v2(2,v1);
  
  while(inFile.eof() == false)
  {
    int dx;
    inFile >> dx;

    for(int b = 0; b<2 ; b++) // branch
    for(int k = 0; k<2 ; k++) // lower and upper
      inFile >> v2[b][k];

    StripKeys key(dx);
    stripLimits[key] = v2;
  } 
  
  inFile.close();
  
  LogTrace("MinBiasTracking")
    << " [ClusterShapeHitFilter] strip-cluster-width filter loaded";
}

/*****************************************************************************/
bool ClusterShapeHitFilter::isInside
  (const vector<vector<float> > limit, const pair<float,float> pred)
{ // pixel
  return (pred.first  > limit[0][0] && pred.first  < limit[0][1] &&
          pred.second > limit[1][0] && pred.second < limit[1][1]);
}  

/*****************************************************************************/
bool ClusterShapeHitFilter::isInside
  (const vector<float> limit, const float pred)
{ // strip
  return (pred > limit[0] && pred < limit[1]);
}  

/*****************************************************************************/
pair<float,float> ClusterShapeHitFilter::getCotangent
  (const PixelGeomDetUnit * pixelDet)
{
  pair<float,float> cotangent;

  cotangent.first  = pixelDet->surface().bounds().thickness() /
                     pixelDet->specificTopology().pitch().first;
  cotangent.second = pixelDet->surface().bounds().thickness() /
                     pixelDet->specificTopology().pitch().second;

  return cotangent;
}

/*****************************************************************************/
float ClusterShapeHitFilter::getCotangent
  (const StripGeomDetUnit * stripDet)
{
  // FIXME may be problematic in case of RadialStriptolopgy
  return stripDet->surface().bounds().thickness() /
         stripDet->specificTopology().localPitch(LocalPoint(0,0,0));
}

/*****************************************************************************/
pair<float,float> ClusterShapeHitFilter::getDrift
  (const PixelGeomDetUnit * pixelDet)
{
  LocalVector lBfield =
     (pixelDet->surface()).toLocal(
      theMagneticField->inTesla(
      pixelDet->surface().position()));

  double theTanLorentzAnglePerTesla = theAngle[pixelDet->type().subDetector()];

  if(theTanLorentzAnglePerTesla == 0.)
     theTanLorentzAnglePerTesla =
         theSiPixelLorentzAngle->getLorentzAngle(
           pixelDet->geographicalId().rawId());

  pair<float,float> dir;
  dir.first  = - theTanLorentzAnglePerTesla * lBfield.y();
  dir.second =   theTanLorentzAnglePerTesla * lBfield.x();

  return dir;
}

/*****************************************************************************/
float ClusterShapeHitFilter::getDrift(const StripGeomDetUnit * stripDet)
{
  LocalVector lBfield =
     (stripDet->surface()).toLocal(
      theMagneticField->inTesla(
      stripDet->surface().position()));
    
  double theTanLorentzAnglePerTesla = theAngle[stripDet->type().subDetector()];

  if(theTanLorentzAnglePerTesla == 0.)
     theTanLorentzAnglePerTesla =
         theSiStripLorentzAngle->getLorentzAngle(
           stripDet->geographicalId().rawId());
    
  float dir = theTanLorentzAnglePerTesla * lBfield.y();

  return dir;
}

/*****************************************************************************/
bool ClusterShapeHitFilter::isNormalOriented
  (const GeomDetUnit * geomDet)
{
  if(geomDet->type().isBarrel())
  { // barrel
    float perp0 = geomDet->toGlobal( Local3DPoint(0.,0.,0.) ).perp();
    float perp1 = geomDet->toGlobal( Local3DPoint(0.,0.,1.) ).perp();
    return (perp1 > perp0);
  }
  else
  { // endcap
    float rot = geomDet->toGlobal( LocalVector (0.,0.,1.) ).z();
    float pos = geomDet->toGlobal( Local3DPoint(0.,0.,0.) ).z();
    return (rot * pos > 0);
  }
}

/*****************************************************************************/
bool ClusterShapeHitFilter::getSizes
  (const SiPixelRecHit & recHit, const LocalVector & ldir,
   int & part, pair<int,int> & meas, pair<float,float> & pred)
{
  ClusterData data;
  ClusterShape theClusterShape;

  DetId id = recHit.geographicalId();
  const PixelGeomDetUnit* pixelDet =
    dynamic_cast<const PixelGeomDetUnit*> (theTracker->idToDet(id));

  theClusterShape.getExtra(*pixelDet, recHit, data);

  part = (pixelDet->type().isBarrel() ? 0 : 1);

  meas = data.size;

  int orient = (isNormalOriented(pixelDet) ? 1 : -1);

  pred.first  = ldir.x() / fabs(ldir.z()) * orient;
  pred.second = ldir.y() / fabs(ldir.z()) * orient;

  if(meas.second < 0)
  { meas.second = abs(meas.second); pred.second = - pred.second; }

  // Take out drift 
  pair<float,float> drift = getDrift(pixelDet);
  pred.first  += drift.first;
  pred.second += drift.second;

  pair<float,float> cotangent = getCotangent(pixelDet);
  pred.first  *= cotangent.first;
  pred.second *= cotangent.second;

  // Usable?
  return (data.isStraight && data.isComplete);
}

/*****************************************************************************/
bool ClusterShapeHitFilter::getSizes
  (const SiStripRecHit2D & recHit, const LocalVector & ldir,
   int & meas, float & pred)
{
  // Get detector
  DetId id = recHit.geographicalId();
  const StripGeomDetUnit* stripDet =
    dynamic_cast<const StripGeomDetUnit*> (theTracker->idToDet(id));

  // Predicted width
  int orient = (isNormalOriented(stripDet) ? 1 : -1);

  pred = ldir.x() / fabs(ldir.z()) + orient * getDrift(stripDet);
  pred *= getCotangent(stripDet);

  // Measured width
  meas  = recHit.cluster()->amplitudes().size();

  // Usable?
  int fs = recHit.cluster()->firstStrip();
  int ns = stripDet->specificTopology().nstrips();

  return (fs > 1 && fs + meas - 1 < ns);
}   

/*****************************************************************************/
bool ClusterShapeHitFilter::isCompatible
  (const SiPixelRecHit & recHit, const LocalVector & ldir)
{
  int part;
  pair<int,int> meas;
  pair<float,float> pred;

  if(getSizes(recHit, ldir, part,meas, pred))
  {
    PixelKeys key(part, meas.first, meas.second);

    if(pixelLimits.count(key) > 0)
      return (isInside(pixelLimits[key][0], pred) ||
              isInside(pixelLimits[key][1], pred));
  }
  
  // Not usable or no limits
  return true;
}

/*****************************************************************************/
bool ClusterShapeHitFilter::isCompatible
  (const SiStripRecHit2D & recHit, const LocalVector & ldir)
{
  int meas;
  float pred;

  if(getSizes(recHit, ldir, meas, pred))
  {
    StripKeys key(meas);

    if(stripLimits.count(key) > 0)
      return (isInside(stripLimits[key][0], pred) ||
              isInside(stripLimits[key][1], pred));
  }

  // Not usable or no limits
  return true;
}

/*****************************************************************************/
bool ClusterShapeHitFilter::isCompatible
  (const SiPixelRecHit & recHit, const GlobalVector & gdir)
{
  LocalVector ldir =
    theTracker->idToDet(recHit.geographicalId())->toLocal(gdir);

  return isCompatible(recHit, ldir);
}

/*****************************************************************************/
bool ClusterShapeHitFilter::isCompatible
  (const SiStripRecHit2D & recHit, const GlobalVector & gdir)
{
  LocalVector ldir =
    theTracker->idToDet(recHit.geographicalId())->toLocal(gdir);

  return isCompatible(recHit, ldir);
}

