// Change the default pixel size from 150*150microns to 100*150microns.
// 9/03 d.k.

#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyBuilder.h"
#include "Geometry/CommonTopologies/interface/RectangularPixelTopology.h"
#include "Geometry/Surface/interface/Bounds.h"

PixelTopologyBuilder::PixelTopologyBuilder(){}

PixelTopology* PixelTopologyBuilder::build(const Bounds* bs,double rocRow,double rocCol,double rocInX,double rocInY,std::string part)
{
  thePixelROCRows = rocRow;
  thePixelBarrelROCsInX = rocInX;
  thePixelROCCols = rocCol;
  thePixelBarrelROCsInY = rocInY;

  float width = bs->width();
  float length = bs->length();

  int nrows = int(thePixelROCRows * thePixelBarrelROCsInX);
  int ncols = int(thePixelROCCols * thePixelBarrelROCsInY);

  float pitchX = width/float(nrows);
  float pitchY = length/float(ncols);

  return new RectangularPixelTopology(nrows,ncols,pitchX,pitchY);

}


