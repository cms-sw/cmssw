#ifndef CPEFromDetPosition_H
#define CPEFromDetPosition_H

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/EtaCorrection.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerSimAlgo/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/RectangularPixelTopology.h"

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
/** \class CPEFromDetPosition
 * Perform the position and error evaluation of pixel hits using 
 * the Det angle to estimate the track impact angle 
*/
#endif

class CPEFromDetPosition : public PixelClusterParameterEstimator 
{
 public:
  // CPEFromDetPosition( const DetUnit& det );
  CPEFromDetPosition(edm::ParameterSet const& conf);
    
  // LocalValues is typedef for pair<LocalPoint,LocalError> 
  LocalValues localParameters( const SiPixelCluster & cl, 
			       const GeomDetUnit    & det ) {
    return std::make_pair( localPosition(cl,det), localError(cl,det) );
  }

  LocalPoint localPosition(const SiPixelCluster& cl, const GeomDetUnit & det);
  LocalError localError   (const SiPixelCluster& cl, const GeomDetUnit & det);
  void       setTheDet( const GeomDetUnit & det );

  //
  MeasurementPoint measurementPosition( const SiPixelCluster&, 
					const GeomDetUnit & det);
  MeasurementError measurementError   ( const SiPixelCluster&, 
					const GeomDetUnit & det);
  //
  float chaWidth2X(const float&) const;
  
 private:
  //members
  const PixelGeomDetUnit* theDet;
  const RectangularPixelTopology * theTopol;
  GeomDetType::SubDetector thePart;
  EtaCorrection theEtaFunc;
  float theThickness;
  float thePitchX;
  float thePitchY;
  float theOffsetX;
  float theOffsetY;
  float theNumOfRow;
  float theNumOfCol;
  float theDetZ;
  float theDetR;
  float theLShift;
  float theSign;

  float theTanLorentzAnglePerTesla;   // Lorentz angle tangent per Tesla
  int   theVerboseLevel;              // algorithm's verbosity

  
  //methods
  float err2X(bool&, int&) const;
  float err2Y(bool&, int&) const;
  bool isFlipped() const;
  // parameters for the position assignment
  float chargeWidthX()const;
  float chargeWidthY()const;
  float geomCorrection()const;
  float estimatedAlphaForBarrel(const float&) const;
  
  // Determine the Lorentz shift correction
  float lorentzShift()const;
  // returns position in x
  float xpos( const SiPixelCluster& ) const;
  //returns position in y
  float ypos( const SiPixelCluster& ) const;

  // charge on the first, last  and  inner pixels on x and y 
  std::vector<float> 
    xCharge(const std::vector<SiPixelCluster::Pixel>&, const float&, const float&)const; 
  std::vector<float> 
    yCharge(const std::vector<SiPixelCluster::Pixel>&, const float&, const float&)const; 

  LocalVector driftDirection( GlobalVector bfield );
  typedef GloballyPositioned<double> Frame;
};

#endif




