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

#include <utility>
#include <vector>
#include "TMath.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitQuality.h"

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
#include "CondFormats/SiPixelObjects/interface/SiPixelCPEGenericErrorParm.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"



#include <unordered_map>

#include <iostream>

class RectangularPixelTopology;
class MagneticField;
class PixelCPEBase : public PixelClusterParameterEstimator 
{
 public:
  struct Param 
  {
    Param() : bz(-9e10f) {}
    float bz; // local Bz
    LocalVector drift;
    float widthLAFraction; // Width-LA to Offset-LA
  };

public:
  PixelCPEBase(edm::ParameterSet const& conf, const MagneticField * mag = 0, 
	       const SiPixelLorentzAngle * lorentzAngle = 0, 
	       const SiPixelCPEGenericErrorParm * genErrorParm = 0, 
	       const SiPixelTemplateDBObject * templateDBobject = 0,
	       const SiPixelLorentzAngle * lorentzAngleWidth = 0
	       );
  

 //--------------------------------------------------------------------------
  // Allow the magnetic field to be set/updated later.
  //--------------------------------------------------------------------------
  inline void setMagField(const MagneticField *mag) const { magfield_ = mag; }
 

  //--------------------------------------------------------------------------
  // Obtain the angles from the position of the DetUnit.
  // LocalValues is typedef for pair<LocalPoint,LocalError> 
  //--------------------------------------------------------------------------
  inline LocalValues localParameters( const SiPixelCluster & cl, 
				      const GeomDetUnit    & det ) const 
    {
      nRecHitsTotal_++ ;
      //std::cout<<" in PixelCPEBase:localParameters(all) - "<<nRecHitsTotal_<<std::endl;  //dk
      setTheDet( det, cl );
      computeAnglesFromDetPosition(cl);
      
      // localPosition( cl, det ) must be called before localError( cl, det ) !!!
      LocalPoint lp = localPosition( cl);
      LocalError le = localError( cl);        
      
      //std::cout<<" in PixelCPEBase:localParameters(all) - "<<lp.x()<<" "<<lp.y()<<std::endl;  //dk

      return std::make_pair( lp, le );
    }
  
  //--------------------------------------------------------------------------
  // In principle we could use the track too to obtain alpha and beta.
  //--------------------------------------------------------------------------
  LocalValues localParameters( const SiPixelCluster & cl,
			       const GeomDetUnit    & det, 
			       const LocalTrajectoryParameters & ltp) const 
  {
    nRecHitsTotal_++ ;

    //std::cout<<" in PixelCPEBase:localParameters(on track) - "<<nRecHitsTotal_<<std::endl;  //dk

    setTheDet( det, cl );
    computeAnglesFromTrajectory(cl, ltp);
    
    // localPosition( cl, det ) must be called before localError( cl, det ) !!!
    LocalPoint lp = localPosition( cl); 
    LocalError le = localError( cl);        

    //std::cout<<" in PixelCPEBase:localParameters(on track) - "<<lp.x()<<" "<<lp.y()<<std::endl;  //dk
    
    return std::make_pair( lp, le );
  } 
  
  
  
private:
  //--------------------------------------------------------------------------
  // This is where the action happens.
  //--------------------------------------------------------------------------
  virtual LocalPoint localPosition(const SiPixelCluster& cl) const = 0;
  virtual LocalError localError   (const SiPixelCluster& cl) const = 0;
  
  
public:  
  //--------------------------------------------------------------------------
  //--- Accessors of other auxiliary quantities
  inline float probabilityX()  const { return probabilityX_ ;  }
  inline float probabilityY()  const { return probabilityY_ ;  }
  inline float probabilityXY() const {
    if ( probabilityX_ !=0 && probabilityY_ !=0 ) 
      {
	return probabilityX_ * probabilityY_ * (1.f - std::log(probabilityX_ * probabilityY_) ) ;
      }
    else 
      return 0;
  }
  
  inline float probabilityQ()  const { return probabilityQ_ ;  }
  inline float qBin()          const { return qBin_ ;          }
  inline bool  isOnEdge()      const { return isOnEdge_ ;      }
  inline bool  hasBadPixels()  const { return hasBadPixels_ ;  }
  inline bool  spansTwoRocks() const { return spansTwoROCs_ ;  }
  inline bool  hasFilledProb() const { return hasFilledProb_ ; }
  
  
  //-----------------------------------------------------------------------------
  //! A convenience method to fill a whole SiPixelRecHitQuality word in one shot.
  //! This way, we can keep the details of what is filled within the pixel
  //! code and not expose the Transient SiPixelRecHit to it as well.  The name
  //! of this function is chosen to match the one in SiPixelRecHit.
  //-----------------------------------------------------------------------------
  SiPixelRecHitQuality::QualWordType rawQualityWord() const;


 protected:
  //--- All methods and data members are protected to facilitate (for now)
  //--- access from derived classes.

  typedef GloballyPositioned<double> Frame;

  //---------------------------------------------------------------------------
  //  Data members
  //---------------------------------------------------------------------------
  //--- Detector-level quantities
  mutable const PixelGeomDetUnit * theDet;
  
  // gavril : replace RectangularPixelTopology with PixelTopology
  mutable const PixelTopology * theTopol;
  mutable const RectangularPixelTopology * theRecTopol;

  mutable Param const * theParam;

  mutable GeomDetType::SubDetector thePart;
  mutable  Local3DPoint theOrigin;
  mutable float theThickness;
  mutable float thePitchX;
  mutable float thePitchY;
  mutable float theNumOfRow;
  mutable float theNumOfCol;
  mutable float theDetZ;
  mutable float theDetR;
  mutable float theSign;

  //--- Cluster-level quantities (may need more)
  mutable float cotalpha_;
  mutable float cotbeta_;
  mutable bool  zneg;

  // G.Giurgiu (05/14/08) track local coordinates
  mutable float trk_lp_x;
  mutable float trk_lp_y;

  //--- Counters
  mutable int    nRecHitsTotal_ ;
  mutable int    nRecHitsUsedEdge_ ;

  // ggiurgiu@jhu.edu (10/18/2008)
  mutable bool with_track_angle; 

  //--- Probability
  mutable float probabilityX_ ; 
  mutable float probabilityY_ ; 
  mutable float probabilityQ_ ; 
  mutable float qBin_ ;
  mutable bool  isOnEdge_ ;
  mutable bool  hasBadPixels_ ;
  mutable bool  spansTwoROCs_ ;
  mutable bool  hasFilledProb_ ;


  //---------------------------
  mutable LocalVector driftDirection_;  // drift direction cached // &&&
  mutable float lorentzShiftInCmX_;   // a FULL shift, in cm
  mutable float lorentzShiftInCmY_;   // a FULL shift, in cm

  // Added new members
  float lAOffset_; // la used to calculate the offset from configuration (for testing) 
  float lAWidthBPix_;  // la used to calculate the cluster width from conf.  
  float lAWidthFPix_;  // la used to calculate the cluster width from conf.
  mutable float lAWidth_;  // la used to calculate the cluster width from conf.
  bool useLAAlignmentOffsets_; // lorentz angle offsets detrmined by alignment
  bool useLAOffsetFromConfig_; // lorentz angle used to calculate the offset
  bool useLAWidthFromConfig_; // lorentz angle used to calculate the cluster width
  bool useLAWidthFromDB_;     // lorentz angle used to calculate the cluster width
  mutable float widthLAFraction_; // ratio of with-LA to offset-LA 

  //--- Global quantities
  int     theVerboseLevel;                    // algorithm's verbosity

  mutable const MagneticField * magfield_;          // magnetic field
  
  mutable const SiPixelLorentzAngle * lorentzAngle_;
  mutable const SiPixelLorentzAngle * lorentzAngleWidth_;  // for the charge width (generic)
  
  mutable const SiPixelCPEGenericErrorParm * genErrorParm_;
  
  mutable const SiPixelTemplateDBObject * templateDBobject_;
  
  bool  alpha2Order;                          // switch on/off E.B effect.
  
  // ggiurgiu@jhu.edu (12/01/2010) : Needed for calling topology methods 
  // with track angles to handle surface deformations (bows/kinks)
  mutable Topology::LocalTrackPred loc_trk_pred_;

  mutable LocalTrajectoryParameters loc_traj_param_;

  //---------------------------------------------------------------------------
  //  Geometrical services to subclasses.
  //---------------------------------------------------------------------------
private:
  void computeAnglesFromDetPosition(const SiPixelCluster & cl ) const;
  
  void computeAnglesFromTrajectory (const SiPixelCluster & cl, 
				    const LocalTrajectoryParameters & ltp) const;

protected:
  void  setTheDet( const GeomDetUnit & det, const SiPixelCluster & cluster ) const ;

  LocalVector driftDirection       ( GlobalVector bfield ) const ; 
  LocalVector driftDirection       ( LocalVector bfield ) const ; 
  void computeLorentzShifts() const ;

  bool isFlipped() const;              // is the det flipped or not?

  //---------------------------------------------------------------------------
  //  Cluster-level services.
  //---------------------------------------------------------------------------
   
  LocalVector const & getDrift() const {return  driftDirection_ ;}
 
  Param const & param() const;
 
 private:
  using Params=std::vector<Param>;
  
  mutable Params m_Params=Params(1440);
  


};

#endif


