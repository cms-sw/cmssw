#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEBase_H
#define RecoLocalTracker_SiPixelRecHits_PixelCPEBase_H 1

//-----------------------------------------------------------------------------
// \class        PixelCPEBase
// \description  Base class for pixel CPE's, with all the common code.
// Perform the position and error evaluation of pixel hits using 
// the Det angle to estimate the track impact angle.
// Move geomCorrection to the concrete class. d.k. 06/06.
//-----------------------------------------------------------------------------

#include <utility>
#include <vector>
#include "TMath.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/EtaCorrection.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"

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

class MagneticField;
class PixelCPEBase : public PixelClusterParameterEstimator {
 public:
  // PixelCPEBase( const DetUnit& det );
  PixelCPEBase(edm::ParameterSet const& conf, const MagneticField * mag = 0);
    
  //--------------------------------------------------------------------------
  // Obtain the angles from the position of the DetUnit.
  // LocalValues is typedef for pair<LocalPoint,LocalError> 
  //--------------------------------------------------------------------------
  inline LocalValues localParameters( const SiPixelCluster & cl, 
				      const GeomDetUnit    & det ) const 
  {
    nRecHitsTotal_++ ;
    setTheDet( det );
    computeAnglesFromDetPosition(cl, det);
    return std::make_pair( localPosition(cl,det), localError(cl,det) );
  }

  //--------------------------------------------------------------------------
  // In principle we could use the track too to obtain alpha and beta.
  //--------------------------------------------------------------------------
  inline LocalValues localParameters( const SiPixelCluster & cl,
				      const GeomDetUnit    & det, 
				      const LocalTrajectoryParameters & ltp) const 
  {
    nRecHitsTotal_++ ;
    setTheDet( det );
    computeAnglesFromTrajectory(cl, det, ltp);
    return std::make_pair( localPosition(cl,det), localError(cl,det) );
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
		setTheDet( det );
		return std::make_pair( localPosition(cl,det), localError(cl,det) );
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
  inline float cotAlphaFromCluster() const { return cotAlphaFromCluster_; }
  inline float cotBetaFromCluster()  const { return cotBetaFromCluster_; }
  inline float probabilityX()        const { return probabilityX_; }
  inline float probabilityY()        const { return probabilityY_; }
  inline float qBin()                const { return qBin_ ; }

 protected:
  //--- All methods and data members are protected to facilitate (for now)
  //--- access from derived classes.

  typedef GloballyPositioned<double> Frame;

  //---------------------------------------------------------------------------
  //  Data members
  //---------------------------------------------------------------------------
  //--- Detector-level quantities
  mutable const PixelGeomDetUnit * theDet;
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

  // [Petar, 5/18/07] 
  // Add estimates of cot(alpha) and cot(beta) from the
  // cluster length.  This can be used by:
  // a) the seed cleaning
  // b) any possible crude "quality" flag based on (dis)agreement between
  //    W_pred and W (from charge lenght)
  // c) an alternative 2nd pass CPE which reads charge per unit length (k_3D) from
  //    the DB but then needs angle estimates to switch to 
  mutable float cotAlphaFromCluster_;
  mutable float cotBetaFromCluster_;

  //--- Probability
  mutable float probabilityX_ ; 
  mutable float probabilityY_ ; 
  mutable float qBin_ ;

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

  //--- Counters
  mutable int    nRecHitsTotal_ ;
  mutable int    nRecHitsUsedEdge_ ;


  //--- Global quantities
  mutable float theTanLorentzAnglePerTesla;   // tan(Lorentz angle)/Tesla
  int     theVerboseLevel;                    // algorithm's verbosity

  mutable const   MagneticField * magfield_;          // magnetic field
  
  bool  alpha2Order;                          // switch on/off E.B effect.


  //---------------------------------------------------------------------------
  //  Methods.
  //---------------------------------------------------------------------------
  void       setTheDet( const GeomDetUnit & det ) const ;
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

};

#endif
