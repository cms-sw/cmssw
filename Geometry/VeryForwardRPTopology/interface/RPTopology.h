/****************************************************************************
*
* This is a part of TotemDQM and TOTEM offline software.
* Authors:
*   Hubert Niewiadomski
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef Geometry_VeryForwardRPTopology_RPTopology
#define Geometry_VeryForwardRPTopology_RPTopology

#include <HepMC/SimpleVector.h>

#include "TMath.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"


using namespace std;

namespace HepMC {
	class ThreeVector;
}

/**
 *\brief Geometrical and topological information on RP silicon detector.
 * Uses coordinate a frame with origin in the center of the wafer.
 **/
class RPTopology
{
  public:
    RPTopology();
    inline const HepMC::ThreeVector& GetStripReadoutAxisDir() const {return strip_readout_direction_;}
    inline const HepMC::ThreeVector& GetStripDirection() const {return strip_direction_;}
    inline const HepMC::ThreeVector& GetNormalDirection() const {return normal_direction_;}

    /// method converts strip number to a hit position [mm] in det readout coordinate 
    /// in the origin in the middle of the si detector
	/// strip_no is assumed in the range 0 ... no_of_strips_ - 1
    inline double GetHitPositionInReadoutDirection(double strip_no) const
//      { return y_width_/2. - last_strip_to_border_dist_ - strip_no * pitch_; }
    {return last_strip_to_border_dist_ + (no_of_strips_-1)*pitch_ - y_width_/2. - strip_no * pitch_;}

    inline double DetXWidth() const {return x_width_;}
    inline double DetYWidth() const {return y_width_;}
    inline double DetEdgeLength() const {return phys_edge_lenght_;}
    inline double DetThickness() const {return thickness_;}
    inline double DetPitch() const {return pitch_;}
    inline unsigned short DetStripNo() const {return no_of_strips_;}

	/// returns true if hit at coordinates u, v (in mm) falls into the sensitive area
	/// can take into account insensitive margin (in mm) at the beam-facing edge
	static bool IsHit(double u, double v, double insensitiveMargin = 0);
      
  public:
    const double sqrt_2;
    
    static double pitch_;
    static double thickness_;
    static unsigned short no_of_strips_;  
    static double x_width_;
    static double y_width_;
    static double phys_edge_lenght_;
    static double last_strip_to_border_dist_;
    static double last_strip_to_center_dist_;

    HepMC::ThreeVector strip_readout_direction_;
    HepMC::ThreeVector strip_direction_;
    HepMC::ThreeVector normal_direction_;
};

#endif  //Geometry_VeryForwardRPTopology_RPTopology
