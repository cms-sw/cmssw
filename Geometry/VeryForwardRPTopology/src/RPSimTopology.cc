#include "Geometry/VeryForwardRPTopology/interface/RPSimTopology.h"
#include <iostream>

RPSimTopology::RPSimTopology(const edm::ParameterSet &params)
{
  verbosity_ = params.getParameter<int>("RPVerbosity");
  no_of_sigms_to_include_ = params.getParameter<double>("RPSharingSigmas");

  top_edge_sigma_ = params.getParameter<double>("RPTopEdgeSmearing");  //[mm]
  bot_edge_sigma_ = params.getParameter<double>("RPBottomEdgeSmearing");  //[mm]
  active_edge_sigma_ = params.getParameter<double>("RPActiveEdgeSmearing");  //[mm]
  
  double phys_active_edge_dist = params.getParameter<double>("RPActiveEdgePosition");  //[mm]
  
  active_edge_x_ = -x_width_/2.0 + phys_edge_lenght_/sqrt_2 + phys_active_edge_dist*sqrt_2;
  active_edge_y_ = -y_width_/2.0;
  
  top_edge_x_ = x_width_/2-params.getParameter<double>("RPTopEdgePosition");  //[mm]
  bot_edge_x_ = params.getParameter<double>("RPBottomEdgePosition")-x_width_/2;  //[mm]
}

std::vector<strip_info> RPSimTopology::GetStripsInvolved(double x, double y, double sigma, double &hit_pos)
{
  theRelevantStrips_.clear();
  hit_pos = (no_of_strips_-1)*pitch_ -(y-last_strip_to_border_dist_+y_width_/2.0);  //hit position with respect to the center of the first strip, only in y direction
  double hit_pos_in_strips = hit_pos/pitch_;
  double hit_factor = ActiveEdgeFactor(x, y)*BottomEdgeFactor(x, y)*TopEdgeFactor(x, y);
  double range_of_interest_in_strips = no_of_sigms_to_include_*sigma/pitch_;
  int lowest_strip_no = (int)floor(hit_pos_in_strips - range_of_interest_in_strips+0.5);
  int highest_strip_no = (int)ceil(hit_pos_in_strips + range_of_interest_in_strips-0.5);

  if(verbosity_)
    std::cout<<"lowest_strip_no:"<<lowest_strip_no<<"  highest_strip_no:"<<highest_strip_no<<std::endl;
  
  if(lowest_strip_no<0)
    lowest_strip_no = 0;
  if(highest_strip_no>no_of_strips_-1)
    highest_strip_no = no_of_strips_-1;

  for(int i=lowest_strip_no; i<=highest_strip_no; ++i)
  {
    double low_strip_range = ((double)i-0.5)*pitch_;
    double high_strip_range = low_strip_range+pitch_;
    theRelevantStrips_.push_back(strip_info(low_strip_range, high_strip_range, hit_factor, (unsigned short)i));
  }
  return theRelevantStrips_;
}
