#ifndef DataFormats_SiPixelRecHit_h
#define DataFormats_SiPixelRecHit_h 1

//---------------------------------------------------------------------------
//!  \class SiPixelRecHit
//!  \brief Pixel Reconstructed Hit
//!
//!  A pixel hit is a 2D position and error in a given
//!  pixel sensor. It contains a persistent reference edm::Ref to the
//!  pixel cluster. 
//!
//!  \author porting from ORCA: Petar Maksimovic (JHU), 
//!          DetSetVector and persistent references: V.Chiochia (Uni Zurich)
//---------------------------------------------------------------------------

//! Our base class
#include "DataFormats/TrackerRecHit2D/interface/BaseSiTrackerRecHit2DLocalPos.h"
//! Quality word packing
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitQuality.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Ref.h"



class SiPixelRecHit : public  BaseSiTrackerRecHit2DLocalPos {
public:

  typedef edm::Ref<edm::DetSetVector<SiPixelCluster>, SiPixelCluster > ClusterRef;

  SiPixelRecHit(): BaseSiTrackerRecHit2DLocalPos (),cluster_() {}

  ~SiPixelRecHit() {}
  
  SiPixelRecHit( const LocalPoint&, const LocalError&,
		 const DetId&, 
		 edm::Ref< edm::DetSetVector<SiPixelCluster>, SiPixelCluster> const&  cluster);  

  virtual SiPixelRecHit * clone() const {return new SiPixelRecHit( * this); }
  
  edm::Ref<edm::DetSetVector<SiPixelCluster>, 
	   SiPixelCluster>  const& cluster() const { return cluster_;}

  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;

#if 1
  //--------------------------------------------------------------------------
  //--- Accessors of other auxiliary quantities
  //--- Added Oct 07 by Petar for 18x.

private:
  SiPixelRecHitQuality::QualWordType  qualWord_ ;   // unsigned int 32-bit wide
  //

public:
  //--- Allow direct access to the packed quality information.
  SiPixelRecHitQuality::QualWordType rawQualityWord() const { return qualWord_; }
  

  inline float cotAlphaFromCluster() const     {
    return SiPixelRecHitQuality::thePacking.cotAlphaFromCluster( qualWord_ );
  }
  inline float cotBetaFromCluster() const     {
    return SiPixelRecHitQuality::thePacking.cotBetaFromCluster( qualWord_ );
  }
  //--- Template fit probability, in X and Y directions
  inline float probabilityX() const     {
    return SiPixelRecHitQuality::thePacking.probabilityX( qualWord_ );
  }
  inline float probabilityY() const     {
    return SiPixelRecHitQuality::thePacking.probabilityY( qualWord_ );
  }
  //--- Charge `bin' (0,1,2,3 ~ charge, qBin==4 is unphysical, qBin=5,6,7 = unused)
  inline int qBin() const     {
    return SiPixelRecHitQuality::thePacking.qBin( qualWord_ );
  }
  //--- Quality flags (true or false):
  //--- cluster is on the edge of the module, or straddles a dead ROC
  inline bool isOnEdge() const     {
    return SiPixelRecHitQuality::thePacking.isOnEdge( qualWord_ );
  }
  //--- cluster contains bad pixels, or straddles bad column or two-column
  inline bool hasBadPixels() const     {
    return SiPixelRecHitQuality::thePacking.hasBadPixels( qualWord_ );
  }
  //--- the cluster spans two ROCS (and thus contains big pixels)
  inline bool spansTwoROCs() const     {
    return SiPixelRecHitQuality::thePacking.spansTwoROCs( qualWord_ );
  }










  // Add estimates of cot(alpha) and cot(beta) from the
  // cluster length.  This can be used by:
  // a) the seed cleaning
  // b) any possible crude "quality" flag based on (dis)agreement between
  //    W_pred and W (from charge lenght)
  // c) an alternative 2nd pass CPE which reads charge per unit length (k_3D) from
  //    the DB but then needs angle estimates to switch to

  //--- cot(alpha) obtained from the sizes along X and Y (= thickness/(size-1))
#endif


#if 0  
  inline void  setCotAlphaFromCluster(float x) const { cotAlphaFromCluster_ = x; }
  inline void  setCotBetaFromCluster(float x)  const { cotBetaFromCluster_ = x; }
  inline void  setProbabilityX(float x)        const { probabilityX_ = x; }
  inline void  setProbabilityY(float x)        const { probabilityY_ = x; }
  inline void  setQBin(float x)                const { qBin_  = x; }
#endif



private:

  edm::Ref<edm::DetSetVector<SiPixelCluster>, SiPixelCluster > cluster_;

};

// Comparison operators
inline bool operator<( const SiPixelRecHit& one, const SiPixelRecHit& other) {
  if ( one.geographicalId() < other.geographicalId() ) {
    return true;
  } else {
    return false;
  }
}

#endif
