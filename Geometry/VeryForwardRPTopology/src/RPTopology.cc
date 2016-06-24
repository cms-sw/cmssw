/****************************************************************************
*
* This is a part of TotemDQM and TOTEM offline software.
* Authors:
*   Hubert Niewiadomski
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"
#include <iostream>

const unsigned short RPTopology::no_of_strips_ = 512;  

const double RPTopology::sqrt_2 = std::sqrt(2.0);
// all in mm
const double RPTopology::pitch_ = 66E-3;
const double RPTopology::thickness_ = 0.3;
const double RPTopology::x_width_ = 36.07;
const double RPTopology::y_width_ = 36.07;
const double RPTopology::phys_edge_lenght_ = 22.276; //correct, but of vague impact, check sensitive edge efficiency curve
const double RPTopology::last_strip_to_border_dist_ = 1.4175;  
const double RPTopology::last_strip_to_center_dist_ = RPTopology::x_width_/2. - RPTopology::last_strip_to_border_dist_;   // assumes square shape



RPTopology::RPTopology()
 : strip_readout_direction_(0, 1, 0),
   strip_direction_(1,0,0),
   normal_direction_(0,0,1)
{
}



bool RPTopology::IsHit(double u, double v, double insensitiveMargin)
{
  // assumes square shape

  if (fabs(u) > last_strip_to_center_dist_)
    return false;

  if (fabs(v) > last_strip_to_center_dist_)
    return false;

  double y = (u + v) / sqrt_2;
  double edge_to_ceter_dist = (x_width_ - phys_edge_lenght_ / sqrt_2) / sqrt_2 - insensitiveMargin;
  if (y < -edge_to_ceter_dist)
    return false;

  return true;
}
