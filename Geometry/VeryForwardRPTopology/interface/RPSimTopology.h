#ifndef Geometry_VeryForwardRPTopology_RPSimTopology
#define Geometry_VeryForwardRPTopology_RPSimTopology

#include "TMath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"

class strip_info
{
  public:
    strip_info(double lower_border, double higher_border, double eff_factor, 
        unsigned short strip_no) : lower_border_(lower_border), 
        higher_border_(higher_border), eff_factor_(eff_factor), strip_no_(strip_no){}
    inline double & HigherBoarder() {return higher_border_;}
    inline double & LowerBoarder() {return lower_border_;}
    inline double & EffFactor() {return eff_factor_;}
    inline unsigned short & StripNo() {return strip_no_;}
    inline double LowerBoarder() const {return lower_border_;}
    inline double HigherBoarder() const {return higher_border_;}
    inline double EffFactor() const {return eff_factor_;}
    inline unsigned short StripNo() const {return strip_no_;}
  private:
    double lower_border_;
    double higher_border_;
    double eff_factor_;
    unsigned short strip_no_;
};

class RPSimTopology : public RPTopology
{
  public:
    RPSimTopology(const edm::ParameterSet &params);
    std::vector<strip_info> GetStripsInvolved(double x, double y, double sigma, double &hit_pos);
      
  private:
    std::vector<strip_info> theRelevantStrips_;
    double no_of_sigms_to_include_;
    //(0,0) in the center of the wafer
    double top_edge_x_;
    double bot_edge_x_;
    double active_edge_x_;
    double active_edge_y_;
    
    double top_edge_sigma_;
    double bot_edge_sigma_;
    double active_edge_sigma_;
    
    int verbosity_;
    
    const LocalVector strip_readout_direction_;
    const LocalVector strip_direction_;
    const LocalVector normal_direction_;
  
    inline double ActiveEdgeFactor(double x, double y)
    {
      return TMath::Erf(DistanceFromActiveEdge(x, y)/sqrt_2/active_edge_sigma_)/2+0.5;
    }
    
    inline double BottomEdgeFactor(double x, double y)
    {
      return TMath::Erf(DistanceFromBottomEdge(x, y)/sqrt_2/bot_edge_sigma_)/2+0.5;
    }
    
    inline double TopEdgeFactor(double x, double y)
    {
      return TMath::Erf(DistanceFromTopEdge(x, y)/sqrt_2/top_edge_sigma_)/2+0.5;
    }
    
    inline double DistanceFromActiveEdge(double x, double y)
    {
      return ((x-active_edge_x_) + (y-active_edge_y_))/sqrt_2;
    }
    inline double DistanceFromBottomEdge(double x, double y)
    {
      return x-bot_edge_x_;
    }
    inline double DistanceFromTopEdge(double x, double y)
    {
      return top_edge_x_ - x;
    }
};

#endif  //Geometry_VeryForwardRPTopology_RPSimTopology
