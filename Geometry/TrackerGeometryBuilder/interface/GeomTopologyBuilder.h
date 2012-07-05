#ifndef Geometry_TrackerGeometryBuilder_GeomTopologyBuilder_H
#define Geometry_TrackerGeometryBuilder_GeomTopologyBuilder_H

#include <string>

class PixelTopology;
class StripTopology;
class Bounds;

/**
 */

class GeomTopologyBuilder {
public:

  GeomTopologyBuilder();

  PixelTopology* buildPixel( const Bounds*, std::string,
			     bool upgradeGeometry,
			     int ROWS_PER_ROC, // Num of Rows per ROC
			     int COLS_PER_ROC, // Num of Cols per ROC
			     int BIG_PIX_PER_ROC_X, // in x direction, rows. BIG_PIX_PER_ROC_X = 0 for SLHC
			     int BIG_PIX_PER_ROC_Y, // in y direction, cols. BIG_PIX_PER_ROC_Y = 0 for SLHC
			     int ROCS_X, int ROCS_Y);
  
  StripTopology* buildStrip( const Bounds* , double ,std::string);

private:

};

#endif
