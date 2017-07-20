#ifndef Geometry_VeryForwardGeometry_CTPPSPixelTopology_H
#define Geometry_VeryForwardGeometry_CTPPSPixelTopology_H

#include "TMath.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelIndices.h"
/**
 *\brief Geometrical and topological information on RPix silicon detector.
 * Uses coordinate a frame with origin in the center of the wafer.
 **/

class CTPPSPixelTopology
{
  public:
    CTPPSPixelTopology();
    ~CTPPSPixelTopology();

    static constexpr double pitch_simY_ = 150E-3;
    static constexpr double pitch_simX_ = 100E-3;
    static constexpr double thickness_ = 0.23;
    static constexpr unsigned short no_of_pixels_simX_ = 160;  
    static constexpr unsigned short no_of_pixels_simY_ = 156;  
    static constexpr unsigned short no_of_pixels_ = 160*156;  
    static constexpr double simX_width_ = 16.6;
    static constexpr double simY_width_ = 24.4;
    static constexpr double dead_edge_width_ = 200E-3;
    static constexpr double active_edge_sigma_ = 0.02;
    static constexpr double phys_active_edge_dist_ = 0.150;

    inline double detPitchSimX() const {return pitch_simX_;}
    inline double detPitchSimY() const {return pitch_simY_;}
    inline double detThickness() const {return thickness_;}
    inline unsigned short detPixelSimXNo() const {return no_of_pixels_simX_;}
    inline unsigned short detPixelSimYNo() const {return no_of_pixels_simY_;}
    inline unsigned short detPixelNo() const {return no_of_pixels_;}
    inline double detXWidth() const {return simX_width_;}
    inline double detYWidth() const {return simY_width_;}
    inline double detDeadEdgeWidth() const {return dead_edge_width_;}
    inline double activeEdgeSigma() const {return active_edge_sigma_;}
    inline double physActiveEdgeDist() const {return phys_active_edge_dist_;}

    CTPPSPixelIndices indici_;    
};

#endif 
