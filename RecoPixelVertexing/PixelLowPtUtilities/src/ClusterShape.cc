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
  bool operator()(SiPixelCluster::Pixel a,
                  SiPixelCluster::Pixel b)
  {
    if(a.x < b.x) return true;
    if(a.x > b.x) return false;

    if(a.y < b.y) return true;
    if(a.y > b.y) return false;

    return false;
  }
};

/*****************************************************************************/
void ClusterShape::determineShape
  (const PixelGeomDetUnit& pixelDet,
   const SiPixelRecHit& recHit, ClusterData& data)
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
  vector<SiPixelCluster::Pixel> pixels = recHit.cluster()->pixels();
  sort(pixels.begin(),pixels.end(),lessPixel());

  // Look at all the pixels
  for(vector<SiPixelCluster::Pixel>::const_iterator pixel = pixels.begin();
                                                    pixel!= pixels.end();
                                                    pixel++)
  {
    // Position
    pos.first  = (int)pixel->x;
    pos.second = (int)pixel->y;

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

  // Check if straight, process last column
  if(processColumn(pos, false) == false)
    data.isStraight = false;

  // Treat clusters with big pixel(s) inside
  for(int ix = recHit.cluster()->minPixelRow() + 1;
          ix < recHit.cluster()->maxPixelRow(); ix++)
    if(theTopology->isItBigPixelInX(ix)) x[1]++;
 
  for(int iy = recHit.cluster()->minPixelCol() + 1;
          iy < recHit.cluster()->maxPixelCol(); iy++)
    if(theTopology->isItBigPixelInY(iy)) y[1]++;

  // Treat clusters with bix pixel(s) outside, FIXME FIXME
  int px = 0;
  if(theTopology->isItBigPixelInX(recHit.cluster()->minPixelRow())) px++;
  if(theTopology->isItBigPixelInX(recHit.cluster()->maxPixelRow())) px++;

  int py = 0;
  if(theTopology->isItBigPixelInY(recHit.cluster()->minPixelCol())) py++;
  if(theTopology->isItBigPixelInY(recHit.cluster()->maxPixelCol())) py++;

  if(px > 0 || py > 0)
    data.hasBigPixelsOnlyInside = false;
  else
    data.hasBigPixelsOnlyInside = true;

  if( (px > 0 || py > 0) && odir == 0)
  {
    // if outside and don't know the direction FIXME?
    data.isComplete = false;
  }
  // else
  { // FIXME do it
    data.size.reserve(px*py);
    for(int ax = 0; ax <= px; ax++)
    for(int ay = 0; ay <= py; ay++)
    {
      int dx = x[1] - x[0] + ax;
      int dy = y[1] - y[0] + ay;
      if(odir != 0) dy *= odir;
  
      pair<int,int> s(dx,dy);
      data.size.push_back(s); 
    }
  }
}
