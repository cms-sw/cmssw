#include "RecoPixelVertexing/PixelLowPtUtilities/interface/TripletFilter.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/HitInfo.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShape.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterData.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include <fstream>
using namespace std;

bool   TripletFilter::isFirst = true;
double TripletFilter::limits[2][MaxSize + 1][MaxSize + 1][2][2][2];

/*****************************************************************************/
TripletFilter::TripletFilter
  (const edm::EventSetup& es)
{
  // Load data
  if(isFirst == true)
  {
    loadClusterLimits();
    isFirst = false;
  }

  // Get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  theTracker = tracker.product();
}

/*****************************************************************************/
TripletFilter::~TripletFilter()
{
}

/*****************************************************************************/
void TripletFilter::loadClusterLimits()
{
  edm::FileInPath
    fileInPath("RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape.par");
  ifstream inFile(fileInPath.fullPath().c_str());

  while(inFile.eof() == false)
  {
    int part,dx,dy;
    
    inFile >> part;
    inFile >> dx;
    inFile >> dy;
    
    for(int b = 0; b<2 ; b++) // branch
    for(int d = 0; d<2 ; d++) // direction
    for(int k = 0; k<2 ; k++) // lower and upper
      inFile >> TripletFilter::limits[part][dx][dy][b][d][k];

    double f;
    inFile >> f; // density
    inFile >> f; 
    
    int d;
    inFile >> d; // points
  } 
  
  inFile.close();

  LogTrace("MinBiasTracking")
    << " [TrackFilter  ] cluster shape filter loaded";
}

/*****************************************************************************/
bool TripletFilter::isInside
  (double a[2][2], pair<double,double> movement)
{
  if(a[0][0] == a[0][1] && a[1][0] == a[1][1])
    return true;
    
  if(movement.first  > a[0][0] && movement.first  < a[0][1] &&
     movement.second > a[1][0] && movement.second < a[1][1])
    return true;
  else
    return false;
}  

/*****************************************************************************/
bool TripletFilter::isCompatible
  (const SiPixelRecHit *recHit, const LocalVector& dir)
{ 
  ClusterData data;
  ClusterShape theClusterShape;

  DetId id = recHit->geographicalId();
  const PixelGeomDetUnit* pixelDet =
    dynamic_cast<const PixelGeomDetUnit*> (theTracker->idToDet(id));
  theClusterShape.getExtra(*pixelDet, *recHit, data);
  
  int dx = data.size.first;
  int dy = data.size.second;
  
  if(data.isStraight && data.isComplete && dx <= MaxSize && abs(dy) <= MaxSize)
  {
    int part   = (data.isInBarrel ? 0 : 1);
    int orient = (data.isNormalOriented ? 1 : -1);
    
    pair<double,double> movement;
    movement.first  = dir.x() / (fabs(dir.z()) * data.tangent.first ) * orient;
    movement.second = dir.y() / (fabs(dir.z()) * data.tangent.second) * orient;
    
    if(dy < 0)
    { dy = abs(dy); movement.second = - movement.second; }

    return (isInside(limits[part][dx][dy][0], movement) ||
            isInside(limits[part][dx][dy][1], movement));
  }
  else
  {
    // Shape is not straight or not complete or too wide
    return true;
  }
}

/*****************************************************************************/
bool TripletFilter::checkTrack
  (vector<const TrackingRecHit*> recHits, vector<LocalVector> localDirs)
{
  bool ok = true;

  vector<LocalVector>::const_iterator localDir = localDirs.begin();
  for(vector<const TrackingRecHit*>::const_iterator recHit = recHits.begin();
                                                    recHit!= recHits.end();
                                                    recHit++)
  {
    const SiPixelRecHit* pixelRecHit =
      dynamic_cast<const SiPixelRecHit *>(*recHit);

    if(pixelRecHit->isValid())
    {
      if(isCompatible(pixelRecHit, *localDir) == false)
      {
        LogTrace("MinBiasTracking")
         << "  [TripletFilter] clusShape problem" << HitInfo::getInfo(**recHit);

        ok = false;
        break;
      }
    }
    else
    { ok = false; break; }

    localDir++;
  }

  return ok;
}

