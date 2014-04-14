#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEBase_H
#define RecoLocalTracker_SiPixelRecHits_PixelCPEBase_H 1

//-----------------------------------------------------------------------------
// \class        PixelCPEBase
// \description  Base class for pixel CPE's, with all the common code.
// Perform the position and error evaluation of pixel hits using 
// the Det angle to estimate the track impact angle.
// Move geomCorrection to the concrete class. d.k. 06/06.
// change to use Lorentz angle from DB Lotte Wilke, Jan. 31st, 2008
// Change to use Generic error & Template calibration from DB - D.Fehling 11/08
//-----------------------------------------------------------------------------

#define NEW_PIXELCPEERROR

#include <utility>
#include <vector>
#include "TMath.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitQuality.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/Topology.h"

//--- For the configuration:
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"

#ifdef NEW_PIXELCPEERROR
// new errors 
#include "CondFormats/SiPixelObjects/interface/SiPixelGenErrorDBObject.h"
#else
// old errors
#include "CondFormats/SiPixelObjects/interface/SiPixelCPEGenericErrorParm.h"
#endif

#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"

#include <unordered_map>

#include <iostream>

class RectangularPixelTopology;
class MagneticField;
class PixelCPEBase : public PixelClusterParameterEstimator 
{
 public:
  struct DetParam
  {
    DetParam() {}
    const PixelGeomDetUnit * theDet;
    // gavril : replace RectangularPixelTopology with PixelTopology
    const PixelTopology * theTopol;
    const RectangularPixelTopology * theRecTopol;

    GeomDetType::SubDetector thePart;
    Local3DPoint theOrigin;
    float theThickness;
    float thePitchX;
    float thePitchY;
    //float theDetR;
    //float theDetZ;
    //float theNumOfRow; //Not used, AH
    //float theNumOfCol; //Not used, AH
    //float theSign; //Not used, AH

    float lAWidth;  // la used to calculate the cluster width from conf.
    float bz; // local Bz
    LocalVector driftDirection;
    float widthLAFraction; // Width-LA to Offset-LA
    float lorentzShiftInCmX;   // a FULL shift, in cm
    float lorentzShiftInCmY;   // a FULL shift, in cm
  };

  struct ClusterParam
  {
    ClusterParam(const SiPixelCluster & cl) : theCluster(&cl), loc_trk_pred(0.0,0.0,0.0,0.0),
      probabilityX_(0.0), probabilityY_(0.0), probabilityQ_(0.0), qBin_(0.0),
      isOnEdge_(false), hasBadPixels_(false), spansTwoROCs_(false), hasFilledProb_(false) {}
    const SiPixelCluster * theCluster;

    //--- Cluster-level quantities (may need more)
    float cotalpha;
    float cotbeta;
    //bool  zneg; // Not used, AH

    // G.Giurgiu (05/14/08) track local coordinates
    float trk_lp_x;
    float trk_lp_y;

    // ggiurgiu@jhu.edu (12/01/2010) : Needed for calling topology methods 
    // with track angles to handle surface deformations (bows/kinks)
    Topology::LocalTrackPred loc_trk_pred;
    //LocalTrajectoryParameters loc_traj_param; // Not used, AH

    // ggiurgiu@jhu.edu (10/18/2008)
    bool with_track_angle; 

    //--- Probability
    float probabilityX_ ; 
    float probabilityY_ ; 
    float probabilityQ_ ; 
    float qBin_ ;
    bool  isOnEdge_ ;
    bool  hasBadPixels_ ;
    bool  spansTwoROCs_ ;
    bool  hasFilledProb_ ;
  };

public:
#ifdef NEW_PIXELCPEERROR
  PixelCPEBase(edm::ParameterSet const& conf, const MagneticField * mag, const TrackerGeometry& geom,
	       const SiPixelLorentzAngle * lorentzAngle, 
	       const SiPixelGenErrorDBObject * genErrorDBObject, 
	       const SiPixelTemplateDBObject * templateDBobject,
	       const SiPixelLorentzAngle * lorentzAngleWidth
	       );  // NEW
#else
  PixelCPEBase(edm::ParameterSet const& conf, const MagneticField * mag, const TrackerGeometry& geom,
	       const SiPixelLorentzAngle * lorentzAngle, 
	       const SiPixelCPEGenericErrorParm * genErrorParm, 
	       const SiPixelTemplateDBObject * templateDBobject,
	       const SiPixelLorentzAngle * lorentzAngleWidth
	       ); // OLD
 #endif 

  //--------------------------------------------------------------------------
  // Allow the magnetic field to be set/updated later.
  //--------------------------------------------------------------------------
  //inline void setMagField(const MagneticField *mag) const { magfield_ = mag; } // Not used, AH
 

  //--------------------------------------------------------------------------
  // Obtain the angles from the position of the DetUnit.
  //--------------------------------------------------------------------------

  inline ReturnType getParameters(const SiPixelCluster & cl, 
				   const GeomDetUnit    & det ) const
    {
#ifdef EDM_ML_DEBUG
      nRecHitsTotal_++ ;
      //std::cout<<" in PixelCPEBase:localParameters(all) - "<<nRecHitsTotal_<<std::endl;  //dk
#endif 

      DetParam const & theDetParam = detParam(det);
      ClusterParam * theClusterParam = createClusterParam(cl);
      setTheDet( theDetParam, *theClusterParam );
      computeAnglesFromDetPosition(theDetParam, *theClusterParam);
      
      // localPosition( cl, det ) must be called before localError( cl, det ) !!!
      LocalPoint lp = localPosition(theDetParam, *theClusterParam);
      LocalError le = localError(theDetParam, *theClusterParam);        
      SiPixelRecHitQuality::QualWordType rqw = rawQualityWord(*theClusterParam);
      auto tuple = std::make_tuple(lp, le , rqw);
      delete theClusterParam;
      
      //std::cout<<" in PixelCPEBase:localParameters(all) - "<<lp.x()<<" "<<lp.y()<<std::endl;  //dk
      return tuple;
    }
  
  //--------------------------------------------------------------------------
  // In principle we could use the track too to obtain alpha and beta.
  //--------------------------------------------------------------------------
  inline ReturnType getParameters(const SiPixelCluster & cl, 
				   const GeomDetUnit    & det, 
				   const LocalTrajectoryParameters & ltp ) const
  {
#ifdef EDM_ML_DEBUG
    nRecHitsTotal_++ ;
    //std::cout<<" in PixelCPEBase:localParameters(on track) - "<<nRecHitsTotal_<<std::endl;  //dk
#endif 

    DetParam const & theDetParam = detParam(det);
    ClusterParam *  theClusterParam = createClusterParam(cl);
    setTheDet( theDetParam, *theClusterParam );
    computeAnglesFromTrajectory(theDetParam, *theClusterParam, ltp);
    
    // localPosition( cl, det ) must be called before localError( cl, det ) !!!
    LocalPoint lp = localPosition(theDetParam, *theClusterParam); 
    LocalError le = localError(theDetParam, *theClusterParam);        
    SiPixelRecHitQuality::QualWordType rqw = rawQualityWord(*theClusterParam);
    auto tuple = std::make_tuple(lp, le , rqw);
    delete theClusterParam;

    //std::cout<<" in PixelCPEBase:localParameters(on track) - "<<lp.x()<<" "<<lp.y()<<std::endl;  //dk
    return tuple;
  } 
  
  
  
private:
  virtual ClusterParam * createClusterParam(const SiPixelCluster & cl) const = 0;

  //--------------------------------------------------------------------------
  // This is where the action happens.
  //--------------------------------------------------------------------------
  virtual LocalPoint localPosition(DetParam const & theDetParam, ClusterParam & theClusterParam) const = 0;
  virtual LocalError localError   (DetParam const & theDetParam, ClusterParam & theClusterParam) const = 0;
  
  void fillDetParams();
  
  //-----------------------------------------------------------------------------
  //! A convenience method to fill a whole SiPixelRecHitQuality word in one shot.
  //! This way, we can keep the details of what is filled within the pixel
  //! code and not expose the Transient SiPixelRecHit to it as well.  The name
  //! of this function is chosen to match the one in SiPixelRecHit.
  //-----------------------------------------------------------------------------
  SiPixelRecHitQuality::QualWordType rawQualityWord(ClusterParam & theClusterParam) const;

 protected:
  //--- All methods and data members are protected to facilitate (for now)
  //--- access from derived classes.

  typedef GloballyPositioned<double> Frame;

  //---------------------------------------------------------------------------
  //  Data members
  //---------------------------------------------------------------------------

  //--- Counters
#ifdef EDM_ML_DEBUG
  mutable int    nRecHitsTotal_ ; //for debugging only
  mutable int    nRecHitsUsedEdge_ ; //for debugging only
#endif 

  // Added new members
  float lAOffset_; // la used to calculate the offset from configuration (for testing) 
  float lAWidthBPix_;  // la used to calculate the cluster width from conf.  
  float lAWidthFPix_;  // la used to calculate the cluster width from conf.
  bool useLAAlignmentOffsets_; // lorentz angle offsets detrmined by alignment
  bool useLAOffsetFromConfig_; // lorentz angle used to calculate the offset
  bool useLAWidthFromConfig_; // lorentz angle used to calculate the cluster width
  bool useLAWidthFromDB_;     // lorentz angle used to calculate the cluster width

  //--- Global quantities
  int     theVerboseLevel;                    // algorithm's verbosity

  const MagneticField * magfield_;          // magnetic field
  const TrackerGeometry & geom_;          // geometry
  
  const SiPixelLorentzAngle * lorentzAngle_;
  const SiPixelLorentzAngle * lorentzAngleWidth_;  // for the charge width (generic)
  

#ifdef NEW_PIXELCPEERROR
  const SiPixelGenErrorDBObject * genErrorDBObject_;  // NEW
#else  
  const SiPixelCPEGenericErrorParm * genErrorParm_;  // OLD
#endif
  
  const SiPixelTemplateDBObject * templateDBobject_;
  bool  alpha2Order;                          // switch on/off E.B effect.
  
  bool DoLorentz_;

  //---------------------------------------------------------------------------
  //  Geometrical services to subclasses.
  //---------------------------------------------------------------------------
private:
  void computeAnglesFromDetPosition( DetParam const & theDetParam, ClusterParam & theClusterParam ) const;
  
  void computeAnglesFromTrajectory ( DetParam const & theDetParam, ClusterParam & theClusterParam,
				    const LocalTrajectoryParameters & ltp) const;

  void  setTheDet( DetParam const &, ClusterParam & theClusterParam ) const ;

  LocalVector driftDirection       (DetParam & theDetParam, GlobalVector bfield ) const ; 
  LocalVector driftDirection       (DetParam & theDetParam, LocalVector bfield ) const ; 
  void computeLorentzShifts(DetParam &) const ;

  bool isFlipped(DetParam const & theDetParam) const;              // is the det flipped or not?

  //---------------------------------------------------------------------------
  //  Cluster-level services.
  //---------------------------------------------------------------------------
   
  DetParam const & detParam(const GeomDetUnit & det) const;
 
  using DetParams=std::vector<DetParam>;
  
  DetParams m_DetParams=DetParams(1440);

};

#endif


