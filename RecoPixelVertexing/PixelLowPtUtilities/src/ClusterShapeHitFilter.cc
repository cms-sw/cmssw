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
ClusterShapeHitFilter::ClusterShapeHitFilter
  (const GlobalTrackingGeometry * theTracker_,
   const MagneticField          * theMagneticField_,
   const SiPixelLorentzAngle    * theSiPixelLorentzAngle_,
   const SiStripLorentzAngle    * theSiStripLorentzAngle_)
   : theTracker(theTracker_),
     theMagneticField(theMagneticField_),
     theSiPixelLorentzAngle(theSiPixelLorentzAngle_),
     theSiStripLorentzAngle(theSiStripLorentzAngle_)

{
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

    const PixelKeys key(part,dx,dy);
    pixelLimits[key] = v3;

    double f;
    int d;

    inFile >> f; // density
    inFile >> d; // points
    inFile >> f; // density
    inFile >> d; // points
  }
  
  inFile.close();
  
  LogTrace("ClusterShapeHitFilter|MinBiasTracking")
    << " pixel-cluster-shape filter loaded";
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
  
  LogTrace("MinBiasTracking|ClusterShapeHitFilter")
    << " strip-cluster-width filter loaded";
}

/*****************************************************************************/
bool ClusterShapeHitFilter::isInside
  (const vector<vector<float> > & limit, const pair<float,float> & pred) const
{ // pixel
  return (pred.first  > limit[0][0] && pred.first  < limit[0][1] &&
          pred.second > limit[1][0] && pred.second < limit[1][1]);
}  

/*****************************************************************************/
bool ClusterShapeHitFilter::isInside
  (const vector<float> & limit, const float & pred) const
{ // strip
  return (pred > limit[0] && pred < limit[1]);
}  

/*****************************************************************************/
pair<float,float> ClusterShapeHitFilter::getCotangent
  (const PixelGeomDetUnit * pixelDet) const
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
  (const StripGeomDetUnit * stripDet) const
{
  // FIXME may be problematic in case of RadialStriptolopgy
  return stripDet->surface().bounds().thickness() /
         stripDet->specificTopology().localPitch(LocalPoint(0,0,0));
}

/*****************************************************************************/
pair<float,float> ClusterShapeHitFilter::getDrift
  (const PixelGeomDetUnit * pixelDet) const
{
  LocalVector lBfield =
     (pixelDet->surface()).toLocal(
      theMagneticField->inTesla(
      pixelDet->surface().position()));

  double theTanLorentzAnglePerTesla =
         theSiPixelLorentzAngle->getLorentzAngle(
           pixelDet->geographicalId().rawId());

  pair<float,float> dir;
  dir.first  = - theTanLorentzAnglePerTesla * lBfield.y();
  dir.second =   theTanLorentzAnglePerTesla * lBfield.x();

  return dir;
}

/*****************************************************************************/
float ClusterShapeHitFilter::getDrift(const StripGeomDetUnit * stripDet)
const
{
  LocalVector lBfield =
     (stripDet->surface()).toLocal(
      theMagneticField->inTesla(
      stripDet->surface().position()));
    
  double theTanLorentzAnglePerTesla =
         theSiStripLorentzAngle->getLorentzAngle(
           stripDet->geographicalId().rawId());
    
  float dir = theTanLorentzAnglePerTesla * lBfield.y();

  return dir;
}

/*****************************************************************************/
bool ClusterShapeHitFilter::isNormalOriented
  (const GeomDetUnit * geomDet) const
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
   int & part, vector<pair<int,int> > & meas, pair<float,float> & pred) const
{
  // Get detector
  DetId id = recHit.geographicalId();
  const PixelGeomDetUnit* pixelDet =
    dynamic_cast<const PixelGeomDetUnit*> (theTracker->idToDet(id));

  // Get shape information
  ClusterData data;
  ClusterShape theClusterShape;
  theClusterShape.determineShape(*pixelDet, recHit, data);
  bool usable = (data.isStraight && data.isComplete);
 
  // Usable?
  if(usable)
  {
    part = (pixelDet->type().isBarrel() ? 0 : 1);

    // Predicted size
    pred.first  = ldir.x() / ldir.z();
    pred.second = ldir.y() / ldir.z();

    if(data.size.front().second < 0)
      pred.second = - pred.second;

    for(vector<pair<int,int> >::const_iterator s = data.size.begin();
                                               s!= data.size.end(); s++)
    {
      meas.push_back(*s);

      if(data.size.front().second < 0)
        meas.back().second = - meas.back().second;
    }

    // Take out drift 
    pair<float,float> drift = getDrift(pixelDet);
    pred.first  += drift.first;
    pred.second += drift.second;

    // Apply cotangent
    pair<float,float> cotangent = getCotangent(pixelDet);
    pred.first  *= cotangent.first;
    pred.second *= cotangent.second;
  }

  // Usable?
  return usable;
}

/*****************************************************************************/
bool ClusterShapeHitFilter::getSizes
  (const SiStripRecHit2D & recHit, const LocalVector & ldir,
   int & meas, float & pred) const 
{
  // Get detector
  DetId id = recHit.geographicalId();
  const StripGeomDetUnit* stripDet =
    dynamic_cast<const StripGeomDetUnit*> (theTracker->idToDet(id));

  // Measured width
  meas   = recHit.cluster()->amplitudes().size();

  // Usable?
  int fs = recHit.cluster()->firstStrip();
  int ns = stripDet->specificTopology().nstrips();
  bool usable = (fs > 1 && fs + meas - 1 < ns);

  // Usable?
  if(usable)
  {
    // Predicted width
    pred = ldir.x() / ldir.z();
  
    // Take out drift
    float drift = getDrift(stripDet);
    pred += drift;
  
    // Apply cotangent
    pred *= getCotangent(stripDet);
  }

  return usable;
}   

/*****************************************************************************/
bool ClusterShapeHitFilter::isCompatible
  (const SiPixelRecHit & recHit, const LocalVector & ldir) const
{
  int part;
  vector<pair<int,int> > meas;
  pair<float,float> pred;

  if(getSizes(recHit, ldir, part,meas, pred))
  {
    for(vector<pair<int,int> >::const_iterator m = meas.begin();
                                               m!= meas.end(); m++)
    {
      PixelKeys key(part, (*m).first, (*m).second);

      PixelLimitsMap::const_iterator i = pixelLimits.find(key);
      if(i != pixelLimits.end())
      { 
        // inside on of the boxes
        if (isInside((i->second)[0], pred) ||
  	    isInside((i->second)[1], pred))
          return true;
      }
      else
      {
        // out of the map
        return true;
      }
    }

    // none of the choices worked
    return false;
  }
  else
  {
    // not usable
    return true;
  }
}

/*****************************************************************************/
bool ClusterShapeHitFilter::isCompatible
  (const SiStripRecHit2D & recHit, const LocalVector & ldir) const
{
  int meas;
  float pred;

  if(getSizes(recHit, ldir, meas, pred))
  {
    StripKeys key(meas);

    StripLimitsMap::const_iterator i=stripLimits.find(key);
    if (i!=stripLimits.end())
      return (isInside((i->second)[0], pred) ||
              isInside((i->second)[1], pred));
    
  }

  // Not usable or no limits
  return true;
}

/*****************************************************************************/
bool ClusterShapeHitFilter::isCompatible
  (const SiPixelRecHit & recHit, const GlobalVector & gdir) const
{
  LocalVector ldir =
    theTracker->idToDet(recHit.geographicalId())->toLocal(gdir);

  return isCompatible(recHit, ldir);
}

/*****************************************************************************/
bool ClusterShapeHitFilter::isCompatible
  (const SiStripRecHit2D & recHit, const GlobalVector & gdir) const
{
  LocalVector ldir =
    theTracker->idToDet(recHit.geographicalId())->toLocal(gdir);

  return isCompatible(recHit, ldir);
}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
EVENTSETUP_DATA_REG(ClusterShapeHitFilter);

