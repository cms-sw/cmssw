#ifndef RecoLocalTracker_SiPixelRecHits_CPEFromDetPosition_H
#define RecoLocalTracker_SiPixelRecHits_CPEFromDetPosition_H 1

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/EtaCorrection.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"

//#define TP_OLD
#ifdef TP_OLD
#include "Geometry/CommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonDetAlgo/interface/MeasurementError.h"
#include "Geometry/Surface/interface/GloballyPositioned.h"
#else  // new location
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"
#endif

//--- For the configuration:
#include "FWCore/ParameterSet/interface/ParameterSet.h"


// &&& Let's hope for the best... //#include "CommonDet/BasicDet/interface/Enumerators.h"

#include <utility>
#include <vector>

// &&& Now explicitly included...  //class MeasurementError;
// class GeomDetUnit;

#if 0
/** \class CPEFromDetPosition
 * Perform the position and error evaluation of pixel hits using 
 * the Det angle to estimate the track impact angle 
*/
#endif
class MagneticField;
class CPEFromDetPosition : public PixelClusterParameterEstimator 
{
 public:
  // CPEFromDetPosition( const DetUnit& det );
  CPEFromDetPosition(edm::ParameterSet const& conf, const MagneticField*);
    
  // LocalValues is typedef for pair<LocalPoint,LocalError> 
  LocalValues localParameters( const SiPixelCluster & cl, 
			       const GeomDetUnit    & det ) const {
    return std::make_pair( localPosition(cl,det), localError(cl,det) );
  }

  LocalPoint localPosition(const SiPixelCluster& cl, const GeomDetUnit & det) const ;
  LocalError localError   (const SiPixelCluster& cl, const GeomDetUnit & det) const ;
  
 protected:
  //members
  mutable const PixelGeomDetUnit* theDet;
  mutable const RectangularPixelTopology * theTopol;
  mutable GeomDetType::SubDetector thePart;
  mutable EtaCorrection theEtaFunc;
  mutable float theThickness;
  mutable float thePitchX;
  mutable float thePitchY;
  mutable float theOffsetX;
  mutable float theOffsetY;
  mutable float theNumOfRow;
  mutable float theNumOfCol;
  mutable  float theDetZ;
  mutable float theDetR;
  mutable float theLShiftX;
  mutable float theLShiftY;

  mutable float theSign;

  mutable float theTanLorentzAnglePerTesla;   // Lorentz angle tangent per Tesla
  int   theVerboseLevel;              // algorithm's verbosity

  //magnetic field
  const MagneticField* magfield_;

  // switch on/off E.B effect
  bool  alpha2Order;

  // Private methods
  void       setTheDet( const GeomDetUnit & det ) const ;
  //
  MeasurementPoint measurementPosition( const SiPixelCluster&, 
					const GeomDetUnit & det) const ;
  MeasurementError measurementError   ( const SiPixelCluster&, 
					const GeomDetUnit & det) const ;
  //float chaWidth2X(const float&) const;

  //methods
  float err2X(bool&, int&) const;
  float err2Y(bool&, int&) const;
  bool isFlipped() const;
  // parameters for the position assignment
  float chargeWidthX()const;
  float chargeWidthY()const;
  float geomCorrectionX(float xpos)const;
  float geomCorrectionY(float ypos)const;
  //float geomCorrection()const;
  float estimatedAlphaForBarrel(const float&) const;
  
  // Determine the Lorentz shift correction along local x and y.
  float lorentzShiftX()const;
  float lorentzShiftY()const;

  // returns position in x
  float xpos( const SiPixelCluster& ) const;
  //returns position in y
  float ypos( const SiPixelCluster& ) const;

  // charge on the first, last  and  inner pixels on x and y 
  std::vector<float> 
    xCharge(const std::vector<SiPixelCluster::Pixel>&, const float&, const float&)const; 
  std::vector<float> 
    yCharge(const std::vector<SiPixelCluster::Pixel>&, const float&, const float&)const; 

  LocalVector driftDirection( GlobalVector bfield )const ;
  typedef GloballyPositioned<double> Frame;
};

#endif




