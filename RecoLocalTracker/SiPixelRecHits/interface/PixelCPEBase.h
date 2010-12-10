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
#include "RecoLocalTracker/SiPixelRecHits/interface/EtaCorrection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitQuality.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/Topology.h"

//--- For the configuration:
#include "FWCore/ParameterSet/interface/ParameterSet.h"

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

#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCPEGenericErrorParm.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"



#include <ext/hash_map>

#include <iostream>



class MagneticField;
class PixelCPEBase : public PixelClusterParameterEstimator 
{
 public:
  // PixelCPEBase( const DetUnit& det );
  PixelCPEBase(edm::ParameterSet const& conf, const MagneticField * mag = 0, 
	       const SiPixelLorentzAngle * lorentzAngle = 0, const SiPixelCPEGenericErrorParm * genErrorParm = 0, 
	       const SiPixelTemplateDBObject * templateDBobject = 0);
  
  //--------------------------------------------------------------------------
  // Obtain the angles from the position of the DetUnit.
  // LocalValues is typedef for pair<LocalPoint,LocalError> 
  //--------------------------------------------------------------------------
  inline LocalValues localParameters( const SiPixelCluster & cl, 
				      const GeomDetUnit    & det ) const 
    {
      nRecHitsTotal_++ ;
      setTheDet( det, cl );
      computeAnglesFromDetPosition(cl, det);
      
      // localPosition( cl, det ) must be called before localError( cl, det ) !!!
      LocalPoint lp = localPosition( cl, det );
      LocalError le = localError( cl, det );        
      
      return std::make_pair( lp, le );
    }
  
  //--------------------------------------------------------------------------
  // In principle we could use the track too to obtain alpha and beta.
  //--------------------------------------------------------------------------
  inline LocalValues localParameters( const SiPixelCluster & cl,
				      const GeomDetUnit    & det, 
				      const LocalTrajectoryParameters & ltp) const 
    {
      nRecHitsTotal_++ ;
      setTheDet( det, cl );
      computeAnglesFromTrajectory(cl, det, ltp);
      
      // localPosition( cl, det ) must be called before localError( cl, det ) !!!
      LocalPoint lp = localPosition( cl, det ); 
      LocalError le = localError( cl, det );        
      
      return std::make_pair( lp, le );
    } 
  
  //--------------------------------------------------------------------------
  // The third one, with the user-supplied alpha and beta
  //--------------------------------------------------------------------------
  inline LocalValues localParameters( const SiPixelCluster & cl,
				      const GeomDetUnit    & det, 
				      float alpha, float beta) const 
    {
      nRecHitsTotal_++ ;
      alpha_ = alpha;
      beta_  = beta;
      double HalfPi = 0.5*TMath::Pi();
      cotalpha_ = tan(HalfPi - alpha_);
      cotbeta_  = tan(HalfPi - beta_ );
      setTheDet( det, cl );
      
      // localPosition( cl, det ) must be called before localError( cl, det ) !!!
      LocalPoint lp = localPosition( cl, det ); 
      LocalError le = localError( cl, det );        
      
      return std::make_pair( lp, le );
    }
  
  
  //--------------------------------------------------------------------------
  // Allow the magnetic field to be set/updated later.
  //--------------------------------------------------------------------------
  inline void setMagField(const MagneticField *mag) const { magfield_ = mag; }
  
  //--------------------------------------------------------------------------
  // This is where the action happens.
  //--------------------------------------------------------------------------
  virtual LocalPoint localPosition(const SiPixelCluster& cl, const GeomDetUnit & det) const;  // = 0, take out dk 8/06
  virtual LocalError localError   (const SiPixelCluster& cl, const GeomDetUnit & det) const = 0;
  
  
  
  //--------------------------------------------------------------------------
  //--- Accessors of other auxiliary quantities
  inline float probabilityX()  const { return probabilityX_ ;  }
  inline float probabilityY()  const { return probabilityY_ ;  }
  inline float probabilityXY() const {
    if ( probabilityX_ !=0 && probabilityY_ !=0 ) 
      {
	return probabilityX_ * probabilityY_ * (1 - log(probabilityX_ * probabilityY_) ) ;
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
  
  //--- Flag to control how SiPixelRecHits compute clusterProbability().
  //--- Note this is set via the configuration file, and it's simply passed
  //--- to each TSiPixelRecHit.
  inline unsigned int clusterProbComputationFlag() const 
    { 
      return clusterProbComputationFlag_ ; 
    }
  
  
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
  //mutable const RectangularPixelTopology * theTopol;
  mutable const PixelTopology * theTopol;
  

  mutable GeomDetType::SubDetector thePart;
  mutable EtaCorrection theEtaFunc;
  mutable float theThickness;
  mutable float thePitchX;
  mutable float thePitchY;
  //mutable float theOffsetX;
  //mutable float theOffsetY;
  mutable float theNumOfRow;
  mutable float theNumOfCol;
  mutable float theDetZ;
  mutable float theDetR;
  mutable float theLShiftX;
  mutable float theLShiftY;
  mutable float theSign;

  //--- Cluster-level quantities (may need more)
  mutable float alpha_;
  mutable float beta_;

  // G.Giurgiu (12/13/06)-----
  mutable float cotalpha_;
  mutable float cotbeta_;

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

  //--- A flag that could be used to change the behavior of
  //--- clusterProbability() in TSiPixelRecHit (the *transient* one).  
  //--- The problem is that the transient hits are made after the CPE runs
  //--- and they don't get the access to the PSet, so we pass it via the
  //--- CPE itself...
  //
  unsigned int clusterProbComputationFlag_ ;

  //---------------------------

  // [Petar, 2/23/07]
  // Since the sign of the Lorentz shift appears to
  // be computed *incorrectly* (i.e. there's a bug) we add new variables
  // so that we can study the effect of the bug.
  mutable LocalVector driftDirection_;  // drift direction cached // &&&
  mutable double lorentzShiftX_;   // a FULL shift, not 1/2 like theLShiftX!
  mutable double lorentzShiftY_;   // a FULL shift, not 1/2 like theLShiftY!
  mutable double lorentzShiftInCmX_;   // a FULL shift, in cm
  mutable double lorentzShiftInCmY_;   // a FULL shift, in cm


  //--- Global quantities
//   mutable float theTanLorentzAnglePerTesla;   // tan(Lorentz angle)/Tesla
  int     theVerboseLevel;                    // algorithm's verbosity

  mutable const MagneticField * magfield_;          // magnetic field
  
  mutable const SiPixelLorentzAngle * lorentzAngle_;
  
  mutable const SiPixelCPEGenericErrorParm * genErrorParm_;
  
  mutable const SiPixelTemplateDBObject * templateDBobject_;
  
  bool  alpha2Order;                          // switch on/off E.B effect.
  
  // ggiurgiu@jhu.edu (12/01/2010) : Needed for calling topology methods 
  // with track angles to handle surface deformations (bows/kinks)
  //mutable Topology::LocalTrackPred* loc_trk_pred;
  mutable Topology::LocalTrackPred loc_trk_pred_;

  mutable LocalTrajectoryParameters loc_traj_param_;
  
  //---------------------------------------------------------------------------
  //  Methods.
  //---------------------------------------------------------------------------
  void       setTheDet( const GeomDetUnit & det, const SiPixelCluster & cluster ) const ;
  //
  MeasurementPoint measurementPosition( const SiPixelCluster&, 
					const GeomDetUnit & det) const ;
  MeasurementError measurementError   ( const SiPixelCluster&, 
					const GeomDetUnit & det) const ;

  //---------------------------------------------------------------------------
  //  Geometrical services to subclasses.
  //---------------------------------------------------------------------------

  //--- Estimation of alpha_ and beta_
  //float estimatedAlphaForBarrel(float centerx) const;
  void computeAnglesFromDetPosition(const SiPixelCluster & cl, 
				    const GeomDetUnit    & det ) const;
  void computeAnglesFromTrajectory (const SiPixelCluster & cl,
				    const GeomDetUnit    & det, 
				    const LocalTrajectoryParameters & ltp) const;
  LocalVector driftDirection       ( GlobalVector bfield ) const ; //wrong sign
  LocalVector driftDirectionCorrect( GlobalVector bfield ) const ;
  void computeLorentzShifts() const ;

  bool isFlipped() const;              // is the det flipped or not?

  //---------------------------------------------------------------------------
  //  Cluster-level services.
  //---------------------------------------------------------------------------
   
  //--- Charge on the first, last  and  inner pixels on x and y 
  void xCharge(const std::vector<SiPixelCluster::Pixel>&, 
	       const int&, const int&, float& q1, float& q2) const; 
  void yCharge(const std::vector<SiPixelCluster::Pixel>&, 
	       const int&, const int&, float& q1, float& q2) const; 

  // Temporary fix for older classes
  //std::vector<float> 
  //xCharge(const std::vector<SiPixelCluster::Pixel>& pixelsVec, 
  //    const float& xmin, const float& xmax) const {
  // return xCharge(pixelsVec, int(xmin), int(xmax)); 
  //}
  //std::vector<float> 
  // yCharge(const std::vector<SiPixelCluster::Pixel>& pixelsVec, 
  //    const float& xmin, const float& xmax) const {
  // return yCharge(pixelsVec, int(xmin), int(xmax)); 
  //}



  //---------------------------------------------------------------------------
  //  Various position corrections.
  //---------------------------------------------------------------------------
  //float geomCorrection()const;

  //--- The Lorentz shift correction
  virtual float lorentzShiftX() const;
  virtual float lorentzShiftY() const;
 
  //--- Position in X and Y
  virtual float xpos( const SiPixelCluster& ) const = 0;
  virtual float ypos( const SiPixelCluster& ) const = 0;
  
  
  
 public:
  struct Param 
  {
    Param() : topology(0), drift(0.0, 0.0, 0.0) {}
    
    // giurgiu@jhu.edu 12/09/2010: switch to PixelTopology
    //RectangularPixelTopology const * topology;
    PixelTopology const * topology;

    LocalVector drift;
  };
  
 private:
   typedef  __gnu_cxx::hash_map< unsigned int, Param> Params;
  
  Params m_Params;
  


};

#endif


