#include "Geometry/TrackerGeometryBuilder/interface/GeomTopologyBuilder.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyBuilder.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripTopologyBuilder.h"



GeomTopologyBuilder::GeomTopologyBuilder(){}

PixelTopology* GeomTopologyBuilder::buildPixel(const Bounds* bs,double rocRow,double rocCol,double rocInX,double rocInY,std::string part,
					       bool upgradeGeometry,
					       int ROWS_PER_ROC, // Num of Rows per ROC
					       int COLS_PER_ROC, // Num of Cols per ROC
					       int BIG_PIX_PER_ROC_X, // in x direction, rows. BIG_PIX_PER_ROC_X = 0 for SLHC
					       int BIG_PIX_PER_ROC_Y, // in y direction, cols. BIG_PIX_PER_ROC_Y = 0 for SLHC
					       int ROCS_X, int ROCS_Y)
{
  PixelTopology* result;
  result = PixelTopologyBuilder().build(bs,rocRow,rocCol,rocInX,rocInY,part,
					upgradeGeometry,
					ROWS_PER_ROC,
					COLS_PER_ROC,
					BIG_PIX_PER_ROC_X,
					BIG_PIX_PER_ROC_Y,
					ROCS_X, ROCS_Y );
  return result;
}
StripTopology* GeomTopologyBuilder::buildStrip(const Bounds* bs,double apvnumb,std::string part)
{
  StripTopology* result;
  result = StripTopologyBuilder().build(bs,apvnumb,part);
  return result;
}
