#ifndef _ClusterShapeHitFilter_h_
#define _ClusterShapeHitFilter_h_

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include <utility>
#include <map>
#include <vector>
#include <cstring>


/*****************************************************************************/
class PixelKeys {
public:
  PixelKeys(int part, int dx, int dy) : key( (part==0) ? barrelPacking(dx,dy) : endcapPacking(dx,dy)) {}
  
  operator unsigned int() const { return key;}
  
  static unsigned char endcapPacking(int dx, int dy) {
    if ( dx<0 || dy<0 ) return N;
    if ( dx>10 || dy>4 ) return N;
  return N_barrel + dx*offset_endcap_dy+dy;  // max 11*5 = 55
  }

  static unsigned char barrelPacking(int dx, int dy) {
    if ( dx<0 || dy<0 ) return N;
    if ( dx>10 || dy>15 ) return N;
    if (dx<8) return dx*16+dy;  // max 8*16=128
    if (dy>2) return N;
    return 128 + (dx-8)*3+dy; // max = 128+9 = 137
  }

  bool isValid() const { return key<N;}


  static const int offset_endcap_dy=5;
  static const int offset_endcap_dx=10;
  static const int N_endcap=55; 
  static const int N_barrel=137; 
  static const int N = N_barrel+N_endcap;

  bool operator<(const PixelKeys & right) const{ return key< right.key;  }
 
 private:
  unsigned char key;
};

class StripKeys
{
 public:

  static const  int N=40;

  StripKeys(int width) : key(width>0 ? width-1 : N) {}
  
  operator unsigned int() const { return key;}

  bool isValid() const { return key<N;}

  bool operator<(const StripKeys & right) const
  { 
    return key < right.key;
  }
 
 private:
  unsigned char key;  // max 40;
};

struct PixelLimits {
  PixelLimits() {
    // init to make sure inside is true;
    auto limit = data[0];
    limit[0][0] = -10e10;
    limit[0][1] =  10e10;
    limit[1][0] = -10e10;
    limit[1][1] =  10e10;
    limit = data[1];
    limit[0][0] = -10e10;
    limit[0][1] =  10e10;
    limit[1][0] = -10e10;
    limit[1][1] =  10e10;
  }

  float data[2][2][2];

  bool isInside( const std::pair<float,float> & pred) const {
    auto limit = data[0];
    bool one = (pred.first  > limit[0][0]) && ( pred.first  < limit[0][1] ) 
						&& (pred.second > limit[1][0]) && (pred.second < limit[1][1]);

    limit = data[1];
    bool two = (pred.first  > limit[0][0]) && ( pred.first  < limit[0][1] ) 
						&& (pred.second > limit[1][0]) && (pred.second < limit[1][1]);
    
    return one || two;
  }


};


struct StripLimits {
  float data[2][2];

  bool isInside( float pred) const {
    float const * limit = data[0];
    bool one = pred > limit[0] && pred < limit[1];
     limit = data[1];
    bool two = pred > limit[0] && pred < limit[1];

    return one || two;

  }
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
  

  std::pair<float,float> getCotangent(const PixelGeomDetUnit * pixelDet) const;
                   float getCotangent(const StripGeomDetUnit * stripDet) const;

  std::pair<float,float> getDrift(const PixelGeomDetUnit * pixelDet) const;
                   float getDrift(const StripGeomDetUnit * stripDet) const;

  bool isNormalOriented(const GeomDetUnit * geomDet) const;

  const GlobalTrackingGeometry * theTracker;
  const MagneticField * theMagneticField;

  const SiPixelLorentzAngle * theSiPixelLorentzAngle;
  const SiStripLorentzAngle * theSiStripLorentzAngle;

  

  PixelLimits pixelLimits[PixelKeys::N+1]; // [2][2][2]

  StripLimits stripLimits[StripKeys::N+1]; // [2][2]

  float theAngle[6];
};

#endif

