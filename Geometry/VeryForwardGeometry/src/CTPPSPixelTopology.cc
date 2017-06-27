#include "Geometry/VeryForwardGeometry/interface/CTPPSPixelTopology.h"
#include <iostream>

unsigned short CTPPSPixelTopology::no_of_pixels_simX_ = 160;  
unsigned short CTPPSPixelTopology::no_of_pixels_simY_ = 156;  
unsigned short CTPPSPixelTopology::no_of_pixels_ = 160*156;  

// all in mm
double CTPPSPixelTopology::pitch_simX_ = 100E-3;
double CTPPSPixelTopology::pitch_simY_ = 150E-3;
double CTPPSPixelTopology::thickness_ = 0.23;
double CTPPSPixelTopology::simX_width_ = 16.6;
double CTPPSPixelTopology::simY_width_ = 24.4;

double CTPPSPixelTopology::dead_edge_width_ = 200E-3;


CTPPSPixelTopology::CTPPSPixelTopology()

{
  indici_ = new CTPPSPixelIndices(156,160);
}
CTPPSPixelTopology::~CTPPSPixelTopology()

{
  delete indici_;
}

