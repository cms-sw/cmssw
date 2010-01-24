#ifndef _ClusterShapeHitFilter_h_
#define _ClusterShapeHitFilter_h_

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include <utility>
#include <map>
#include <vector>
#include <cstring>

//#define MaxSize 20

/*****************************************************************************/
class PixelKeys
{
 public:
  PixelKeys(int part, int dx, int dy) : key1(part), key2(dx), key3(dy) { }

  bool operator<(const PixelKeys & right) const
  {
    if(key1 < right.key1) return true;
    if(key1 > right.key1) return false;

    if(key2 < right.key2) return true;
    if(key2 > right.key2) return false;

    if(key3 < right.key3) return true;
                     else return false;
  }
 
 private:
  unsigned int key1, key2, key3;
};

/*****************************************************************************/
class StripKeys
{
 public:
  StripKeys(int width) : key1(width) { }
  
  bool operator<(const StripKeys & right) const
  { 
    if(key1 < right.key1) return true;
                     else return false;
  }
 
 private:
  unsigned int key1;
};

/*****************************************************************************/
namespace edm { class EventSetup; }

class SiPixelRecHit;
class SiStripRecHit2D;
class GlobalTrackingGeometry;
class MagneticField;
class SiPixelLorentzAngle;
class SiStripLorentzAngle;
class GeomDetUnit;
class PixelGeomDetUnit;
class StripGeomDetUnit;

class ClusterShapeHitFilter
{
 public:
  typedef TrajectoryFilter::Record Record;
  //  typedef CkfComponentsRecord Record;

  ClusterShapeHitFilter(const GlobalTrackingGeometry * theTracker_,
                        const MagneticField          * theMagneticField_,
                        const SiPixelLorentzAngle    * theSiPixelLorentzAngle_,
                        const SiStripLorentzAngle    * theSiStripLorentzAngle_);
  
  ~ClusterShapeHitFilter();

  bool getSizes
    (const SiPixelRecHit & recHit, const LocalVector & ldir,
     int & part, std::vector<std::pair<int,int> > & meas,
     std::pair<float,float> & pred) const;

  bool getSizes
    (const SiStripRecHit2D & recHit, const LocalVector & ldir,
     int & meas, float & pred) const;

  bool isCompatible(const SiPixelRecHit   & recHit,
                    const LocalVector & ldir) const;
  bool isCompatible(const SiStripRecHit2D & recHit,
                    const LocalVector & ldir) const;

  bool isCompatible(const SiPixelRecHit   & recHit,
                    const GlobalVector & gdir) const;
  bool isCompatible(const SiStripRecHit2D & recHit,
                    const GlobalVector & gdir) const;

 private:
  void loadPixelLimits();
  void loadStripLimits();
  
  bool isInside(const std::vector<std::vector<float> > & limit ,
                const std::pair<float,float> & pred) const;
  bool isInside(const std::vector<float> & limit ,
                const float & pred) const;

  std::pair<float,float> getCotangent(const PixelGeomDetUnit * pixelDet) const;
                   float getCotangent(const StripGeomDetUnit * stripDet) const;

  std::pair<float,float> getDrift(const PixelGeomDetUnit * pixelDet) const;
                   float getDrift(const StripGeomDetUnit * stripDet) const;

  bool isNormalOriented(const GeomDetUnit * geomDet) const;

  const GlobalTrackingGeometry * theTracker;
  const MagneticField * theMagneticField;

  const SiPixelLorentzAngle * theSiPixelLorentzAngle;
  const SiStripLorentzAngle * theSiStripLorentzAngle;

  typedef std::map<PixelKeys, std::vector<std::vector<std::vector<float> > > > PixelLimitsMap;
  PixelLimitsMap pixelLimits; // [2][2][2]

  typedef std::map<StripKeys, std::vector<std::vector<float> > > StripLimitsMap;
  StripLimitsMap stripLimits; // [2][2]

  float theAngle[6];
};

#endif

