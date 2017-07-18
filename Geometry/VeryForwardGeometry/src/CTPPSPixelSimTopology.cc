#include "Geometry/VeryForwardGeometry/interface/CTPPSPixelSimTopology.h"
#include <iostream>

CTPPSPixelSimTopology::CTPPSPixelSimTopology(const edm::ParameterSet &params)
{
  verbosity_ = params.getUntrackedParameter<int>("RPixVerbosity");

  active_edge_sigma_ = params.getParameter<double>("RPixActiveEdgeSmearing");  //[mm]
  
  double phys_active_edge_dist = params.getParameter<double>("RPixActiveEdgePosition");  //[mm]
  

  active_edge_x_ = simX_width_/2.0 - phys_active_edge_dist;
  active_edge_y_ = simY_width_/2.0 - phys_active_edge_dist;
  

}

std::vector<pixel_info> CTPPSPixelSimTopology::GetPixelsInvolved(double x, double y, double sigma, double &hit_pos_x, double &hit_pos_y)
{
  theRelevantPixels_.clear();
//hit position wrt the bottom left corner of the sensor (-8.3, -12.2) in sensor view, rocs behind

  hit_pos_x = x + simX_width_/2.;
  hit_pos_y = y + simY_width_/2.;
  assert(hit_pos_x*hit_pos_y > 0);

  double hit_factor = ActiveEdgeFactor(x, y);

  unsigned int interested_row = Row(x);
  unsigned int interested_col = Col(y);
  double low_pixel_range_x, high_pixel_range_x, low_pixel_range_y, high_pixel_range_y;
  PixelRange(interested_row, interested_col, low_pixel_range_x, high_pixel_range_x, low_pixel_range_y, high_pixel_range_y);


  theRelevantPixels_.push_back(pixel_info( low_pixel_range_x, high_pixel_range_x, low_pixel_range_y, high_pixel_range_y, hit_factor, interested_row, interested_col));

  return theRelevantPixels_;
}
