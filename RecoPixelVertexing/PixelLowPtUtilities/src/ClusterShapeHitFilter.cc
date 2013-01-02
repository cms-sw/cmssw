#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/HitInfo.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShape.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterData.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
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
#include<cassert>

using namespace std;


/*****************************************************************************/
ClusterShapeHitFilter::ClusterShapeHitFilter
  (const TrackerGeometry * theTracker_,
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
  fillPixelData();

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


  while(inFile.eof() == false)
  {
    int part,dx,dy;

    inFile >> part; // 0or 1
    inFile >> dx;   // 0 to 10
    inFile >> dy;   // 0 to 15 ...

    const PixelKeys key(part,dx,dy);
    auto & pl = pixelLimits[key];

    for(int b = 0; b<2 ; b++) // branch
    for(int d = 0; d<2 ; d++) // direction
    for(int k = 0; k<2 ; k++) // lower and upper
      inFile >> pl.data[b][d][k];


    double f;
    int d;

    inFile >> f; // density
    inFile >> d; // points
    inFile >> f; // density
    inFile >> d; // points
  }
  
  inFile.close();
  
  LogTrace("MinBiasTracking|ClusterShapeHitFilter")
    << " [ClusterShapeHitFilter] pixel-cluster-shape filter loaded";
 }

/*****************************************************************************/
void ClusterShapeHitFilter::loadStripLimits()
{
  // Load strip
  edm::FileInPath
    fileInPath("RecoPixelVertexing/PixelLowPtUtilities/data/stripShape.par");
  ifstream inFile(fileInPath.fullPath().c_str());

  
  while(inFile.eof() == false)
  {
    int dx;
    inFile >> dx;

    StripKeys key(dx);
    auto & sl = stripLimits[key];

    for(int b = 0; b<2 ; b++) // branch
    for(int k = 0; k<2 ; k++) // lower and upper
      inFile >> sl.data[b][k];

  } 
  
  inFile.close();
  
  LogTrace("MinBiasTracking|ClusterShapeHitFilter")
    << " [ClusterShapeHitFilter] strip-cluster-width filter loaded";
}



void ClusterShapeHitFilter::fillPixelData() {

  //barrel
  for (auto det : theTracker->detsPXB()) {
    // better not to fail..
    const PixelGeomDetUnit * pixelDet =
      dynamic_cast<const PixelGeomDetUnit*>(det);
    assert(pixelDet);
    PixelData & pd = pixelData[pixelDet->geographicalId()];
    pd.det = pixelDet;
    pd.part=0;
    pd.cotangent=getCotangent(pixelDet);
    pd.drift=getDrift(pixelDet);
  }

  //endcap
  for (auto det : theTracker->detsPXF()) {
    // better not to fail..
    const PixelGeomDetUnit * pixelDet =
      dynamic_cast<const PixelGeomDetUnit*>(det);
    assert(pixelDet);
    PixelData & pd = pixelData[pixelDet->geographicalId()];
    pd.det = pixelDet;
    pd.part=1;
    pd.cotangent=getCotangent(pixelDet);
    pd.drift=getDrift(pixelDet);
  }

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
/*****************************************************************************/

bool ClusterShapeHitFilter::getSizes
  (const SiPixelRecHit & recHit, const LocalVector & ldir,
   int & part, vector<pair<int,int> > & meas, pair<float,float> & pred,
   PixelData const * ipd) const
{
  // Get detector
  const PixelData & pd = getpd(recHit,ipd);

  // Get shape information
  ClusterData data;
  ClusterShape theClusterShape;
  theClusterShape.determineShape(*pd.det, recHit, data);
  bool usable = (data.isStraight && data.isComplete);
 
  // Usable?
  //if(usable)
  {
    part = pd.part;

    // Predicted size
    pred.first  = ldir.x() / ldir.z();
    pred.second = ldir.y() / ldir.z();

    if(data.size.front().second < 0)
      pred.second = - pred.second;

    meas.reserve(data.size.size());
    for(vector<pair<int,int> >::const_iterator s = data.size.begin();
	s!= data.size.end(); s++)
      {
	meas.push_back(*s);
	
	if(data.size.front().second < 0)
	  meas.back().second = - meas.back().second;
      }

    // Take out drift 
    std::pair<float,float> const & drift = pd.drift;
    pred.first  += drift.first;
    pred.second += drift.second;

    // Apply cotangent
    std::pair<float,float> const & cotangent = pd.cotangent;
    pred.first  *= cotangent.first;
    pred.second *= cotangent.second;
  }

  // Usable?
  return usable;
}

bool ClusterShapeHitFilter::isCompatible
  (const SiPixelRecHit & recHit, const LocalVector & ldir,
		    PixelData const * ipd) const
{
 // Get detector
  const PixelData & pd = getpd(recHit,ipd);

  int part;
  vector<pair<int,int> > meas;
  pair<float,float> pred;

  if(getSizes(recHit, ldir, part,meas, pred,&pd))
  {
    for(vector<pair<int,int> >::const_iterator m = meas.begin();
                                               m!= meas.end(); m++)
    {
      PixelKeys key(part, (*m).first, (*m).second);
      if (!key.isValid()) return true; // FIXME original logic
      if (pixelLimits[key].isInside(pred)) return true;
    }
    // none of the choices worked
    return false;
  }
  // not usable
  return true;
}

bool ClusterShapeHitFilter::isCompatible
  (const SiPixelRecHit & recHit, const GlobalVector & gdir,
		    PixelData const * ipd) const
{
 // Get detector
  const PixelData & pd = getpd(recHit,ipd);

  LocalVector ldir =pd.det->toLocal(gdir);

  return isCompatible(recHit, ldir,&pd);
}


/*****************************************************************************/
/*****************************************************************************/
bool ClusterShapeHitFilter::getSizes
  (DetId id, const SiStripCluster & cluster, const LocalVector & ldir,
   int & meas, float & pred) const 
{
  // Get detector
  const StripGeomDetUnit* stripDet =
    dynamic_cast<const StripGeomDetUnit*> (theTracker->idToDet(id));

  // Measured width
  meas   = cluster.amplitudes().size();

  // Usable?
  int fs = cluster.firstStrip();
  int ns = stripDet->specificTopology().nstrips();
  // bool usable = (fs > 1 && fs + meas - 1 < ns);
  bool usable = (fs >= 1 && fs + meas - 1 <= ns);

  // Usable?
  //if(usable)
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
  (DetId detId, const SiStripCluster & cluster, const LocalVector & ldir) const
{
  int meas;
  float pred;

  if(getSizes(detId, cluster, ldir, meas, pred))
  {
    StripKeys key(meas);
    if (key.isValid())
      return stripLimits[key].isInside(pred);
  }

  // Not usable or no limits
  return true;
}


/*****************************************************************************/
bool ClusterShapeHitFilter::isCompatible
  (DetId detId, const SiStripCluster & cluster, const GlobalVector & gdir) const
{
  LocalVector ldir = theTracker->idToDet(detId)->toLocal(gdir);
  return isCompatible(detId, cluster, ldir);
}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(ClusterShapeHitFilter);

