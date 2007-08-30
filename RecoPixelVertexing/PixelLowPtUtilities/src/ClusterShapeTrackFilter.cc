#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeTrackFilter.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShape.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterData.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include <fstream>

//bool   ClusterShapeTrackFilter::isFirst = true;
//double ClusterShapeTrackFilter::limits[2][MaxSize + 1][MaxSize + 1][2][2][2];

/*****************************************************************************/
ClusterShapeTrackFilter::ClusterShapeTrackFilter
  (const edm::ParameterSet& ps)
//  (const edm::EventSetup& es)
{
  loadClusterLimits();

  // Get tracker geometry
/*
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  theTracker = tracker.product();
*/
}

/*****************************************************************************/
ClusterShapeTrackFilter::~ClusterShapeTrackFilter()
{
}

/*****************************************************************************/
void ClusterShapeTrackFilter::loadClusterLimits()
{
  edm::FileInPath
    fileInPath("RecoPixelVertexing/PixelLowPtUtilities/data/clusterShape.par");
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
      inFile >> limits[part][dx][dy][b][d][k];
//      inFile >> ClusterShapeTrackFilter::limits[part][dx][dy][b][d][k];

    double f;
    inFile >> f; // density
    inFile >> f; 
    
    int d;
    inFile >> d; // points
  } 
  
  inFile.close();

  cerr << "[TrackFilter  ] pixel cluster shape filter loaded" << endl;
}

/*****************************************************************************/
bool ClusterShapeTrackFilter::isInside
  (const double a[2][2], pair<double,double> movement) const
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
bool ClusterShapeTrackFilter::isCompatible
  (const SiPixelRecHit *recHit, const LocalVector& dir) const
{ 
  ClusterData data;
  ClusterShape theClusterShape;

  DetId id = recHit->geographicalId();
// FIXME
  const PixelGeomDetUnit* pixelDet = 0; // =
//    dynamic_cast<const PixelGeomDetUnit*> (theTracker->idToDet(id));
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
bool ClusterShapeTrackFilter::operator()
  (const reco::Track* track, std::vector<const TrackingRecHit *> hits) const
{
  // DEACTIVATED
  return true;

  bool ok = true;
  vector<const TrackingRecHit*> recHits;
  vector<LocalVector> localDirs;

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
      { ok = false; break; }
    }
    else
    { ok = false; break; }

    localDir++;
  }

//  cerr << " [TrackFilter$$] ok = " << (ok == true ? 1 : 0) << endl;

  return ok;
}

