#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPETemplateReco_H
#define RecoLocalTracker_SiPixelRecHits_PixelCPETemplateReco_H

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"

// Already in the base class
//#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
//#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
//#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
//#include "Geometry/CommonDetAlgo/interface/MeasurementPoint.h"
//#include "Geometry/CommonDetAlgo/interface/MeasurementError.h"
//#include "Geometry/Surface/interface/GloballyPositioned.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"


#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate.h"
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
  PixelCPETemplateReco(edm::ParameterSet const& conf, const MagneticField *, const SiPixelLorentzAngle *, const SiPixelTemplateDBObject *);
  ~PixelCPETemplateReco();

  // We only need to implement measurementPosition, since localPosition() from
  // PixelCPEBase will call it and do the transformation
  // Gavril : put it back
  LocalPoint localPosition (const SiPixelCluster& cluster, const GeomDetUnit & det) const; 
  
  // However, we do need to implement localError().
  LocalError localError   (const SiPixelCluster& cl, const GeomDetUnit & det) const;

 protected:
  //--- These functions are no longer needed, yet they are declared 
  //--- pure virtual in the base class.
  float xpos( const SiPixelCluster& ) const { return -999000.0; }  // &&& should abort
  float ypos( const SiPixelCluster& ) const { return -999000.0; }  // &&& should abort

 private:
  // Template storage
  mutable SiPixelTemplate templ_ ;
 //---------------------------
  // [Morris, 6/25/08]
  // Cache the template ID number
  mutable int templID_;   // in general this will change in time and via DetID
  
  // The result of PixelTemplateReco2D
  mutable float templXrec_ ; 
  mutable float templYrec_ ;
  mutable float templSigmaX_ ;
  mutable float templSigmaY_ ;
  // Add new information produced by SiPixelTemplateReco::PixelTempReco2D &&&
  // These can only be accessed if we change silicon pixel data formats and add them to the rechit
  mutable float templProbX_ ;
  mutable float templProbY_ ;

  mutable float templProbQ_;

  mutable int templQbin_ ;

  mutable int speed_ ;

  mutable int ierr;

  mutable bool UseClusterSplitter_;

  mutable bool DoCosmics_;

  mutable bool LoadTemplatesFromDB_;

};

#endif




