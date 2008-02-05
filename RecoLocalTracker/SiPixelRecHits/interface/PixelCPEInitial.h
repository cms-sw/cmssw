#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEInitial_H
#define RecoLocalTracker_SiPixelRecHits_PixelCPEInitial_H 1

// Move geomCorrection from the base class, modify it. 
// comment out etaCorrection. d.k. 06/06
// change to use Lorentz angle from DB Lotte Wilke, Jan. 31st, 2008

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
//#include "RecoLocalTracker/SiPixelRecHits/interface/EtaCorrection.h"

// Already in the base class
//#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
//#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
//#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"
//#include "Geometry/CommonDetAlgo/interface/MeasurementPoint.h"
//#include "Geometry/CommonDetAlgo/interface/MeasurementError.h"
//#include "Geometry/Surface/interface/GloballyPositioned.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <utility>
#include <vector>

#if 0
/** \class PixelCPEInitial
 * Perform the position and error evaluation of pixel hits using 
 * the Det angle to estimate the track impact angle 
*/
#endif

class MagneticField;
class PixelCPEInitial : public PixelCPEBase
{
 public:
  // PixelCPEInitial( const DetUnit& det );
  PixelCPEInitial(edm::ParameterSet const& conf, const MagneticField*, const SiPixelLorentzAngle *);
    
  //LocalPoint localPosition(const SiPixelCluster& cl, const GeomDetUnit & det) const ;
  LocalError localError   (const SiPixelCluster& cl, const GeomDetUnit & det) const ;
  

 protected:
  //--------------------------------------------------------------------
  //  Methods.  For now (temporarily) they are all protected.
  //------------------------------------------------------------------

  // Errors squared in x and y
  float err2X(bool&, int&) const;
  float err2Y(bool&, int&) const;

  // Position in x and y
  float xpos( const SiPixelCluster& ) const;
  float ypos( const SiPixelCluster& ) const;

  // Quantities needed to calculate xpos() and ypos()
  float chargeWidthX()const;
  float chargeWidthY()const;
  //float chaWidth2X(const float&) const;  // &&& NOT USED.  Remove?

  float geomCorrectionX(float xpos)const;  // 2nd order correction to the 
  float geomCorrectionY(float ypos)const;  // track angle from detector position

};

#endif




