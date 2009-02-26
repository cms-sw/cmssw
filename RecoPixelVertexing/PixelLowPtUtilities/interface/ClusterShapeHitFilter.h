#ifndef _ClusterShapeHitFilter_h_
#define _ClusterShapeHitFilter_h_

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
 // singleton begin
 protected:
  static ClusterShapeHitFilter * _instance;
  static int _refCount;
 
 public:
  static ClusterShapeHitFilter * Instance(const edm::EventSetup& es,
                                          const std::string & caller);
  static void Release();
  static void Destroy();

 protected:
  ClusterShapeHitFilter();
 // singleton end

 public:
  ClusterShapeHitFilter(const edm::EventSetup& es);
  ClusterShapeHitFilter(const GlobalTrackingGeometry * theTracker_,
                        const MagneticField   * theMagneticField_);

  ~ClusterShapeHitFilter();

  bool getSizes
    (const SiPixelRecHit & recHit, const LocalVector & ldir,
     int & part, std::pair<int,int> & meas, std::pair<float,float> & pred);

  bool getSizes
    (const SiStripRecHit2D & recHit, const LocalVector & ldir,
     int & meas, float & pred);

  bool isCompatible(const SiPixelRecHit   & recHit, const LocalVector & ldir);
  bool isCompatible(const SiStripRecHit2D & recHit, const LocalVector & ldir);

  bool isCompatible(const SiPixelRecHit   & recHit, const GlobalVector & gdir);
  bool isCompatible(const SiStripRecHit2D & recHit, const GlobalVector & gdir);

 private:
  void loadPixelLimits();
  void loadStripLimits();
  
  bool isInside(const std::vector<std::vector<float> > limit,
                const std::pair<float,float> pred);
  bool isInside(const std::vector<float> limit,
                const float pred);

  std::pair<float,float> getCotangent(const PixelGeomDetUnit * pixelDet);
                   float getCotangent(const StripGeomDetUnit * stripDet);

  std::pair<float,float> getDrift(const PixelGeomDetUnit * pixelDet);
                   float getDrift(const StripGeomDetUnit * stripDet);

  bool isNormalOriented(const GeomDetUnit * geomDet);

  const GlobalTrackingGeometry * theTracker;
  const MagneticField * theMagneticField;

  const SiPixelLorentzAngle * theSiPixelLorentzAngle;
  const SiStripLorentzAngle * theSiStripLorentzAngle;
 
  std::map<PixelKeys, std::vector<std::vector<std::vector<float> > > >
                           pixelLimits; // [2][2][2]
  std::map<StripKeys, std::vector<std::vector<float> > > 
                           stripLimits; // [2][2]

  float theAngle[6];
};

#endif

