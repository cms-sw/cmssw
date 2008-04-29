#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShape.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

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
 // Dimensions
 int nrows = pixelDet.specificTopology().nrows();
 int ncols = pixelDet.specificTopology().ncolumns();

 // Tangents
 data.tangent.first  = pixelDet.specificTopology().pitch().first/
                       pixelDet.surface().bounds().thickness();    
 data.tangent.second = pixelDet.specificTopology().pitch().second/
                       pixelDet.surface().bounds().thickness();    

 // Initialize
 data.isStraight = true;
 data.isComplete = true;

 x[1]=-1; olow=-2; ohig=-2; odir=0;
 pair<int,int> pos;

 // Process channels
 SiPixelRecHit::ClusterRef const& cluster = recHit.cluster();
 vector<SiPixelCluster::Pixel> pixels = (*cluster).pixels();

 // Sort pixels
 sort(pixels.begin(),pixels.end(),lessPixel());

 int size = pixels.size();
 for(int i=0; i<size; i++)
 {
   // Position
   pos.first  = (int)pixels[i].x;
   pos.second = (int)pixels[i].y;

   // Check if at the edge
   if(pos.first  == 0 || pos.first  == nrows-1 ||
      pos.second == 0 || pos.second == ncols-1)
   { 
     data.isComplete = false;

     if(pos.first == 0 || pos.first  == nrows-1)
     {
       data.isXBorder = true;
       if(pos.first == 0) data.posBorder = 0;
                    else data.posBorder = nrows;
     }

     // overwrite
     if(pos.second == 0 || pos.second  == ncols-1)
     {
       data.isXBorder = false;
       if(pos.second == 0) data.posBorder = 0;
                      else data.posBorder = ncols;
     }

     break;
   }

   // Check if it is big
   if(RectangularPixelTopology::isItBigPixelInX(pos.first) ||
      RectangularPixelTopology::isItBigPixelInY(pos.second))
   { data.isComplete = false; break; }

   if(pos.first > x[1])
   { // Process column
     if(processColumn(pos, true) == false)
     { data.isStraight = false; break; }
   }
   else
   { // Increasing row
     if(pos.second > hig+1)
     { data.isStraight = false; break; }

     hig = pos.second;
   }
 }

 // Process last column
 if(processColumn(pos, false) == false)
   data.isStraight = false;
}

/*****************************************************************************/
void ClusterShape::getOrientation
  (const PixelGeomDetUnit& pixelDet, ClusterData& data)
{
  if(pixelDet.type().subDetector() == GeomDetEnumerators::PixelBarrel)
  {
    data.isInBarrel = true;

    float perp0 = pixelDet.toGlobal( Local3DPoint(0.,0.,0.) ).perp();
    float perp1 = pixelDet.toGlobal( Local3DPoint(0.,0.,1.) ).perp();
    data.isNormalOriented = (perp1 > perp0);
  }
  else
  {
    data.isInBarrel = false;

    float rot = pixelDet.toGlobal( LocalVector (0.,0.,1.) ).z();
    float pos = pixelDet.toGlobal( Local3DPoint(0.,0.,0.) ).z();
    data.isNormalOriented = (rot * pos > 0);
  }
}

/*****************************************************************************/
void ClusterShape::getExtra
  (const PixelGeomDetUnit& pixelDet,
   const SiPixelRecHit& recHit, ClusterData& data)
{
  data.isUnlocked = true;

  getOrientation(pixelDet,        data);
  determineShape(pixelDet,recHit, data);

  int dx = x[1] - x[0];
  int dy = y[1] - y[0];
  if(odir != 0) dy *= odir;

  data.size.first  = (unsigned short int)dx;
  data.size.second = (short int)dy;
}

