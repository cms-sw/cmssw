#include "Geometry/VeryForwardGeometry/interface/CTPPSPixelSimTopology.h"

CTPPSPixelSimTopology::CTPPSPixelSimTopology()
{
  active_edge_x_ = simX_width_*0.5 - phys_active_edge_dist_;
  active_edge_y_ = simY_width_*0.5 - phys_active_edge_dist_;
}

CTPPSPixelSimTopology::PixelInfo
CTPPSPixelSimTopology::getPixelsInvolved( double x, double y, double sigma, double& hit_pos_x, double& hit_pos_y ) const
{
  //hit position wrt the bottom left corner of the sensor (-8.3, -12.2) in sensor view, rocs behind
  hit_pos_x = x + simX_width_/2.;
  hit_pos_y = y + simY_width_/2.;
  if( !( hit_pos_x*hit_pos_y > 0 ) )
    throw cms::Exception("CTPPSPixelSimTopology") << "out of reference frame";

  double hit_factor = activeEdgeFactor( x, y );

  unsigned int interested_row = row( x );
  unsigned int interested_col = col( y );
  double low_pixel_range_x, high_pixel_range_x, low_pixel_range_y, high_pixel_range_y;
  pixelRange( interested_row, interested_col, low_pixel_range_x, high_pixel_range_x, low_pixel_range_y, high_pixel_range_y );

  return CTPPSPixelSimTopology::PixelInfo( low_pixel_range_x, high_pixel_range_x, low_pixel_range_y, high_pixel_range_y, hit_factor, interested_row, interested_col );
}
