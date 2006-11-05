#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPETemplateReco_H
#define RecoLocalTracker_SiPixelRecHits_PixelCPETemplateReco_H

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"

#include "Geometry/CommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonDetAlgo/interface/MeasurementError.h"

//--- For the various "Frames"
#include "Geometry/Surface/interface/GloballyPositioned.h"

//--- For the configuration:
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplate.h"
#else
#include "SiPixelTemplate.h"
#endif

#include <utility>
#include <vector>


#if 0
/** \class PixelCPETemplateReco
 * Perform the position and error evaluation of pixel hits using 
 * the Det angle to estimate the track impact angle 
*/
#endif

class MagneticField;
class PixelCPETemplateReco : public PixelCPEBase
{
 public:
  // PixelCPETemplateReco( const DetUnit& det );
  PixelCPETemplateReco(edm::ParameterSet const& conf, const MagneticField*);
  ~PixelCPETemplateReco();

  // We only need to implement measurementPosition, since localPosition() from
  // PixelCPEBase will call it and do the transformation.
  MeasurementPoint measurementPosition( const SiPixelCluster&, 
					const GeomDetUnit & det) const ;

  // However, we do need to implement localError().
  LocalError localError   (const SiPixelCluster& cl, const GeomDetUnit & det) const ;

 private:
  //--------------------------------------------------------------------
  //  Methods.
  //------------------------------------------------------------------

  // Position in x and y
  float xpos( const SiPixelCluster& ) const;
  float ypos( const SiPixelCluster& ) const;

  // Quantities needed to calculate xpos() and ypos()
  float chargeWidthX() const;
  float chargeWidthY() const;

 private:
  // Template stuff in here &&&

  mutable SiPixelTemplate templ_ ;
  
  // The result
  mutable float templXrec_ ; 
  mutable float templYrec_ ;
  mutable float templSigmaX_ ;
  mutable float templSigmaY_ ;
};

#endif




