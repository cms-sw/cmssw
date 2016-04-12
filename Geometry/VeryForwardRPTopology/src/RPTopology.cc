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

unsigned short RPTopology::no_of_strips_ = 512;  

// all in mm
double RPTopology::pitch_ = 66E-3;
double RPTopology::thickness_ = 0.3;
double RPTopology::x_width_ = 36.07;
double RPTopology::y_width_ = 36.07;
double RPTopology::phys_edge_lenght_ = 22.276; //correct, but of vague impact, check sensitive edge efficiency curve
double RPTopology::last_strip_to_border_dist_ = 1.4175;  
double RPTopology::last_strip_to_center_dist_ = RPTopology::x_width_/2. - RPTopology::last_strip_to_border_dist_;   // assumes square shape



RPTopology::RPTopology()
 : sqrt_2(sqrt(2.0)),
   strip_readout_direction_(0, 1, 0),
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

  double sqrt_2 = sqrt(2.);
  double y = (u + v) / sqrt_2;
  double edge_to_ceter_dist = (x_width_ - phys_edge_lenght_ / sqrt_2) / sqrt_2 - insensitiveMargin;
  if (y < -edge_to_ceter_dist)
    return false;

  return true;
}
