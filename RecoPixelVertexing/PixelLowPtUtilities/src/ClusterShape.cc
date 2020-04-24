#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShape.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterData.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

#include <vector>
#include <fstream>

using namespace std;

/*****************************************************************************/
ClusterShape::ClusterShape()
{
}

/*****************************************************************************/
ClusterShape::~ClusterShape()
{
}

/*****************************************************************************/
int ClusterShape::getDirection(int low,int hig, int olow,int ohig)
{
  if(hig == ohig && low == olow) return  0;
  if(hig >= ohig && low >= olow) return  1; 
  if(hig <= ohig && low <= olow) return -1;
   
  return -2;
}
 
/*****************************************************************************/
bool ClusterShape::processColumn(pair<int,int> pos, bool inTheLoop)
{
  if(x[1] > -1)
  { // Process previous column
    if(low < y[0] || x[1] == x[0]) y[0] = low;
    if(hig > y[1] || x[1] == x[0]) y[1] = hig;

    if(x[1] > x[0])
    { // Determine direction
      int dir = getDirection(low,hig, olow,ohig);

      // no direction
      if(dir == -2) return false;

      if(x[1] > x[0]+1)
      { // Check if direction changes
        if(odir*dir == -1)
        { odir = -2; return false; }
      }

      if(x[1] <= x[0]+1 || odir == 0)
        odir = dir;

    }

    olow = low; ohig = hig;
  }
  else
  { // Very first column, initialize
    x[0] = pos.first;
  }

  // Open new column
  if(inTheLoop)
  {
    x[1] = pos.first;
    low  = pos.second;
    hig  = pos.second;
  }

  return(true);
}

/*****************************************************************************/
struct lessPixel : public binary_function<SiPixelCluster::Pixel,
                                          SiPixelCluster::Pixel,bool>
{
  bool operator()(const SiPixelCluster::Pixel& a,
                  const SiPixelCluster::Pixel& b) const
  {
    // slightly faster by avoiding branches
    return (a.x < b.x) | ((a.x == b.x) & (a.y < b.y));
  }
};

/*****************************************************************************/
void ClusterShape::determineShape
  (const PixelGeomDetUnit& pixelDet,
   const SiPixelRecHit& recHit, ClusterData& data)
{
  determineShape(pixelDet, *(recHit.cluster()), data);
}

void ClusterShape::determineShape
  (const PixelGeomDetUnit& pixelDet,
   const SiPixelCluster& cluster, ClusterData& data)
{
  // Topology
  const PixelTopology * theTopology = (&(pixelDet.specificTopology())); 
 
  // Initialize
  data.isStraight = true;
  data.isComplete = true;
 
  x[0] = -1; x[1] = -1;
  y[0] = -1; y[1] = -1;
  olow = -2; ohig = -2; odir = 0;
  low = 0; hig = 0;

  pair<int,int> pos;
 
  // Get sorted pixels
  size_t npixels = cluster.pixelADC().size();
  pixels_.reserve(npixels);
  for(size_t i=0; i<npixels; ++i) {
    pixels_.push_back(cluster.pixel(i));
  }
  sort(pixels_.begin(),pixels_.end(),lessPixel());

  // Look at all the pixels
  for(const auto& pixel: pixels_)
  {
    // Position
    pos.first  = (int)pixel.x;
    pos.second = (int)pixel.y;

    // Check if at the edge or big 
    if(theTopology->isItEdgePixelInX(pos.first) ||
       theTopology->isItEdgePixelInY(pos.second))
    { data.isComplete = false; } // break; }
 
    // Check if straight
    if(pos.first > x[1])
    { // column ready
      if(processColumn(pos, true) == false)
      { data.isStraight = false; } // break; }
    }
    else
    { // increasing column
      if(pos.second > hig+1) // at least a pixel is missing
      { data.isStraight = false; } // break; }
 
      hig = pos.second;
    }
  }
  pixels_.clear();

  // Check if straight, process last column
  if(processColumn(pos, false) == false)
    data.isStraight = false;

  // Treat clusters with big pixel(s) inside
  const int minPixelRow = cluster.minPixelRow();
  const int maxPixelRow = cluster.maxPixelRow();
  for(int ix = minPixelRow + 1;
          ix < maxPixelRow; ix++)
    x[1] += theTopology->isItBigPixelInX(ix);
 
  const int minPixelCol = cluster.minPixelCol();
  const int maxPixelCol = cluster.maxPixelCol();
  for(int iy = minPixelCol + 1;
          iy < maxPixelCol; iy++)
    y[1] += theTopology->isItBigPixelInY(iy);

  // Treat clusters with bix pixel(s) outside, FIXME FIXME
  unsigned int px = 0;
  px += theTopology->isItBigPixelInX(minPixelRow);
  px += theTopology->isItBigPixelInX(maxPixelRow);

  unsigned int py = 0;
  py += theTopology->isItBigPixelInY(minPixelCol);
  py += theTopology->isItBigPixelInY(maxPixelCol);

  data.hasBigPixelsOnlyInside = (px <= 0 && py <= 0);

  //if( (px > 0 || py > 0) && odir == 0)
  if( !data.hasBigPixelsOnlyInside && odir == 0)
  {
    // if outside and don't know the direction FIXME?
    data.isComplete = false;
  }
  // else
  { // FIXME do it
    assert((px+1)*(py+1) <= data.size.capacity());
    const int pre_dx = x[1] - x[0];
    const int pre_dy = y[1] - y[0];
    for(unsigned int ax = 0; ax <= px; ax++)
    for(unsigned int ay = 0; ay <= py; ay++)
    {
      int dx = pre_dx + ax;
      int dy = pre_dy + ay;
      if(odir != 0) dy *= odir;
  
      pair<int,int> s(dx,dy);
      data.size.push_back_unchecked(s);
    }
  }
}
