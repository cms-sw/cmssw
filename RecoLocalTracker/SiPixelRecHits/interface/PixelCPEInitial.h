#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEInitial_H
#define RecoLocalTracker_SiPixelRecHits_PixelCPEInitial_H 1

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/EtaCorrection.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"

#include "Geometry/CommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonDetAlgo/interface/MeasurementError.h"

//--- For the various "Frames"
#include "Geometry/Surface/interface/GloballyPositioned.h"

//--- For the configuration:
#include "FWCore/ParameterSet/interface/ParameterSet.h"


// &&& Let's hope for the best... //#include "CommonDet/BasicDet/interface/Enumerators.h"

#include <utility>
#include <vector>

// &&& Now explicitly included...  //class MeasurementError;
// class GeomDetUnit;

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
  PixelCPEInitial(edm::ParameterSet const& conf, const MagneticField*);
    
  LocalPoint localPosition(const SiPixelCluster& cl, const GeomDetUnit & det) const ;
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
  float chaWidth2X(const float&) const;  // &&& NOT USED.  Remove?
};

#endif




