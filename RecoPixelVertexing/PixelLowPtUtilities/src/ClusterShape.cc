#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShape.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterData.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"

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
        { odir = -2; return(false); }
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
 const RectangularPixelTopology * theTopology = 
   dynamic_cast<const RectangularPixelTopology *>
     (&(pixelDet.specificTopology())); 

 // Initialize
 data.isStraight = true;
 data.isComplete = true;

 x[0]=-1; x[1]=-1; olow=-2; ohig=-2; odir=0; low=0; hig=0;
 pair<int,int> pos;

 // Get a sort pixels
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
      theTopology->isItEdgePixelInY(pos.second) ||
      theTopology->isItBigPixelInX(pos.first) ||
      theTopology->isItBigPixelInY(pos.second))
   { data.isComplete = false; break; }

   // Check if straight
   if(pos.first > x[1])
   { // column ready
     if(processColumn(pos, true) == false)
     { data.isStraight = false; break; }
   }
   else
   { // increasing column
     if(pos.second > hig+1) // at least a pixel is missing
     { data.isStraight = false; break; }

     hig = pos.second;
   }
 }

 // Check if straight, process last column
 if(processColumn(pos, false) == false)
   data.isStraight = false;
}

/*****************************************************************************/
void ClusterShape::getExtra
  (const PixelGeomDetUnit& pixelDet,
   const SiPixelRecHit& recHit, ClusterData& data)
{
  determineShape(pixelDet,recHit, data);

  int dx = x[1] - x[0];
  int dy = y[1] - y[0];
  if(odir != 0) dy *= odir;

  data.size.first  = dx;
  data.size.second = dy;
}

