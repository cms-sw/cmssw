#ifndef _ClusterShape_h_
#define _ClusterShape_h_

#include <utility>

class PixelGeomDetUnit;
class SiPixelRecHit;
class ClusterData;

class ClusterShape
{
 public:
  ClusterShape();
  ~ClusterShape();
  void determineShape(const PixelGeomDetUnit& pixelDet,
                      const SiPixelRecHit& recHit, ClusterData& data);

 private:
  int getDirection(int low,int hig, int olow,int ohig);
  bool processColumn(std::pair<int,int> pos, bool inTheLoop);
/*
  void determineShape
    (const PixelGeomDetUnit& pixelDet,
     const SiPixelRecHit& recHit,     ClusterData& data);
*/
/*
  void getOrientation
    (const PixelGeomDetUnit& pixelDet,ClusterData& data);
*/

  int x[2],y[2], low,hig, olow,ohig, odir;
};

#endif
