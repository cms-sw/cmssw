#ifndef Geometry_Veryforwardgeometry_RPix_DET_TOPOLOGY_H
#define Geometry_Veryforwardgeometry_RPix_DET_TOPOLOGY_H

#include "TMath.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelIndices.h"

using namespace std; 
namespace HepMC {
	class ThreeVector;
}

/**
 *\brief Geometrical and topological information on RPix silicon detector.
 * Uses coordinate a frame with origin in the center of the wafer.
 **/
class CTPPSPixelTopology
{
  public:
    CTPPSPixelTopology();
    ~CTPPSPixelTopology();


 

    inline double DetXWidth() const {return simX_width_;}
    inline double DetYWidth() const {return simY_width_;}
//   inline double DetEdgeLength() const {return phys_edge_lenght_;}
    inline double DetDeadEdgeWidth() const {return dead_edge_width_;}
    inline double DetThickness() const {return thickness_;}
    inline double DetPitchSimX() const {return pitch_simX_;}
    inline double DetPitchSimY() const {return pitch_simY_;}
    inline unsigned short DetPixelSimXNo() const {return no_of_pixels_simX_;}
    inline unsigned short DetPixelSimYNo() const {return no_of_pixels_simY_;}
    inline unsigned short DetPixelNo() const {return no_of_pixels_;}
    CTPPSPixelIndices *Indici() const {return indici_;}

      
  public:

    static double pitch_simY_;
    static double pitch_simX_;
    static double thickness_;
    static unsigned short no_of_pixels_simX_;  
    static unsigned short no_of_pixels_simY_;  
    static unsigned short no_of_pixels_;  
    static double simX_width_;
    static double simY_width_;
    static double dead_edge_width_;

    CTPPSPixelIndices *indici_;

};

#endif 
