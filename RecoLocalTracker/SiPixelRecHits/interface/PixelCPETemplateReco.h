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
  struct ClusterParamTemplate : ClusterParam
  {
    ClusterParamTemplate(const SiPixelCluster & cl) : ClusterParam(cl){}
    // The result of PixelTemplateReco2D
    float templXrec_ ; 
    float templYrec_ ;
    float templSigmaX_ ;
    float templSigmaY_ ;
    // Add new information produced by SiPixelTemplateReco::PixelTempReco2D &&&
    // These can only be accessed if we change silicon pixel data formats and add them to the rechit
    float templProbX_ ;
    float templProbY_ ;

    float templProbQ_;

    int templQbin_ ;

    int ierr;

  };

  // PixelCPETemplateReco( const DetUnit& det );
  PixelCPETemplateReco(edm::ParameterSet const& conf, const MagneticField *, const TrackerGeometry&, const TrackerTopology&,
                       const SiPixelLorentzAngle *, const SiPixelTemplateDBObject *);

  ~PixelCPETemplateReco();

 private:
  ClusterParam * createClusterParam(const SiPixelCluster & cl) const;

  // We only need to implement measurementPosition, since localPosition() from
  // PixelCPEBase will call it and do the transformation
  // Gavril : put it back
  LocalPoint localPosition (DetParam const & theDetParam, ClusterParam & theClusterParam) const; 
  
  // However, we do need to implement localError().
  LocalError localError   (DetParam const & theDetParam, ClusterParam & theClusterParam) const;

  // Template storage
  std::vector< SiPixelTemplateStore > thePixelTemp_;
  
  int speed_ ;

  bool UseClusterSplitter_;

  //bool DoCosmics_;
  //bool LoadTemplatesFromDB_;

};

#endif




