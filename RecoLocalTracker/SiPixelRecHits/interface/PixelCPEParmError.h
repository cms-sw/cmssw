// change to use Lorentz angle from DB Lotte Wilke, Jan. 31st, 2008

#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEParmError_H
#define RecoLocalTracker_SiPixelRecHits_PixelCPEParmError_H

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/EtaCorrection.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelErrorParametrization.h"


// Already in the base class
//#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
//#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
//#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"
//#include "Geometry/CommonDetAlgo/interface/MeasurementPoint.h"
//#include "Geometry/CommonDetAlgo/interface/MeasurementError.h"
//#include "Geometry/Surface/interface/GloballyPositioned.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"

// &&& Let's hope for the best... //#include "CommonDet/BasicDet/interface/Enumerators.h"

#include <utility>
#include <vector>

// &&& Now explicitly included...  //class MeasurementError;
// class GeomDetUnit;

#if 0
/** \class PixelCPEParmError
 * Perform the position and error evaluation of pixel hits using 
 * the Det angle to estimate the track impact angle 
*/
#endif

class MagneticField;
class PixelCPEParmError : public PixelCPEBase
{
 public:
  // PixelCPEParmError( const DetUnit& det );
  PixelCPEParmError(edm::ParameterSet const& conf, const MagneticField*, const SiPixelLorentzAngle*);
  ~PixelCPEParmError();

  //LocalPoint localPosition(const SiPixelCluster& cl, const GeomDetUnit & det) const ;
  LocalError localError   (const SiPixelCluster& cl, const GeomDetUnit & det) const ;
  
 private:
  //--------------------------------------------------------------------
  //  Methods.  For now (temporarily) they are all protected.
  //------------------------------------------------------------------

  // Position in x and y
  float xpos( const SiPixelCluster& ) const;
  float ypos( const SiPixelCluster& ) const;

  // Quantities needed to calculate xpos() and ypos()
  float chargeWidthX()const;
  float chargeWidthY()const;

 private:
  PixelErrorParametrization * pixelErrorParametrization_;
};

#endif




