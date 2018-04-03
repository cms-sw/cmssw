#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEClusterRepair_H
#define RecoLocalTracker_SiPixelRecHits_PixelCPEClusterRepair_H

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"

// Already in the base class
//#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
//#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
//#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
//#include "Geometry/CommonDetAlgo/interface/MeasurementPoint.h"
//#include "Geometry/CommonDetAlgo/interface/MeasurementError.h"
//#include "Geometry/Surface/interface/GloballyPositioned.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"

// The template header files
//
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateReco.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateReco2D.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate2D.h"
#include "CondFormats/SiPixelObjects/interface/SiPixel2DTemplateDBObject.h"

#include <utility>
#include <vector>


#if 0
/** \class PixelCPEClusterRepair
 * Perform the position and error evaluation of pixel hits using
 * the Det angle to estimate the track impact angle
 */
#endif

class MagneticField;
class PixelCPEClusterRepair : public PixelCPEBase
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
      int   templQbin_ ;
      int   ierr;


      // 2D fit stuff.
      float templProbXY_ ;
      bool  recommended3D_ ;
      int   ierr2;
   };
   
   // PixelCPEClusterRepair( const DetUnit& det );
   PixelCPEClusterRepair(edm::ParameterSet const& conf, const MagneticField *, const TrackerGeometry&, const TrackerTopology&,
			 const SiPixelLorentzAngle *, const SiPixelTemplateDBObject *, const SiPixel2DTemplateDBObject * );
   
   ~PixelCPEClusterRepair() override;
   
private:
   ClusterParam * createClusterParam(const SiPixelCluster & cl) const override;
   
   // Calculate local position.  (Calls TemplateReco)
   LocalPoint localPosition (DetParam const & theDetParam, ClusterParam & theClusterParam) const override;
   // Calculate local error. Note: it MUST be called AFTER localPosition() !!!
   LocalError localError   (DetParam const & theDetParam, ClusterParam & theClusterParam) const override;
   
   // Helper functions: 

   // Call vanilla template reco, then clean-up
   void callTempReco2D( DetParam const & theDetParam, 
			ClusterParamTemplate & theClusterParam, 
			SiPixelTemplateReco::ClusMatrix & clusterPayload,
			int ID, LocalPoint & lp ) const;

   // Call 2D template reco, then clean-up
   void callTempReco3D( DetParam const & theDetParam, 
			ClusterParamTemplate & theClusterParam, 
			SiPixelTemplateReco2D::ClusMatrix & clusterPayload,
			int ID, LocalPoint & lp ) const;
   

   // Template storage
   std::vector< SiPixelTemplateStore >   thePixelTemp_;
   std::vector< SiPixelTemplateStore2D > thePixelTemp2D_;

   int speed_ ;
   
   bool UseClusterSplitter_;

   // Template file management (when not getting the templates from the DB)
   int barrelTemplateID_ ;
   int forwardTemplateID_ ;
   std::string templateDir_ ;

   // Configure 3D reco.
   float minProbY_ ;
   int   maxSizeMismatchInY_ ;
   
   //bool DoCosmics_;
   //bool LoadTemplatesFromDB_;
   
};

#endif




