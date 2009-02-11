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

  typedef edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster > ClusterRef;

  SiPixelRecHit(): BaseSiTrackerRecHit2DLocalPos(), qualWord_(0), cluster_()  {}

  ~SiPixelRecHit() {}
  
  SiPixelRecHit( const LocalPoint&, const LocalError&,
		 const DetId&, 
		 ClusterRef const&  cluster);  

  virtual SiPixelRecHit * clone() const {return new SiPixelRecHit( * this); }
  
  ClusterRef const& cluster() const { return cluster_;}
  void setClusterRef(const ClusterRef &ref) { cluster_  = ref; }

  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;


  //--------------------------------------------------------------------------
  //--- Accessors of other auxiliary quantities
  //--- Added Oct 07 by Petar for 18x.
private:
  // *************************************************************************
  //
  SiPixelRecHitQuality::QualWordType  qualWord_ ;   // unsigned int 32-bit wide
  //
  // *************************************************************************

public:
  //--- Allow direct access to the packed quality information.
  SiPixelRecHitQuality::QualWordType rawQualityWord() const { return qualWord_; }
  

  //--- Template fit probability, in X and Y directions
  inline float probabilityX() const     {
    return SiPixelRecHitQuality::thePacking.probabilityX( qualWord_ );
  }
  inline float probabilityY() const     {
    return SiPixelRecHitQuality::thePacking.probabilityY( qualWord_ );
  }

  // Add estimates of cot(alpha) and cot(beta) from the
  // cluster length.  This can be used by:
  // a) the seed cleaning
  // b) any possible crude "quality" flag based on (dis)agreement between
  //    W_pred and W (from charge lenght)
  // c) an alternative 2nd pass CPE which reads charge per unit length (k_3D) from
  //    the DB but then needs angle estimates to switch to
  //--- cot(alpha) obtained from the sizes along X and Y (= thickness/(size-1))
  inline float cotAlphaFromCluster() const     {
    return SiPixelRecHitQuality::thePacking.cotAlphaFromCluster( qualWord_ );
  }
  inline float cotBetaFromCluster() const     {
    return SiPixelRecHitQuality::thePacking.cotBetaFromCluster( qualWord_ );
  }

  //--- Charge `bin' (values 0, 1, 2, 3) according to Morris's template
  //--- code. qBin==4 is unphysical, qBin=5,6,7 are yet unused)
  //
  inline int qBin() const     {
    return SiPixelRecHitQuality::thePacking.qBin( qualWord_ );
  }

  //--- Quality flags (true or false):

  //--- The cluster is on the edge of the module, or straddles a dead ROC
  inline bool isOnEdge() const     {
    return SiPixelRecHitQuality::thePacking.isOnEdge( qualWord_ );
  }
  //--- The cluster contains bad pixels, or straddles bad column or two-column
  inline bool hasBadPixels() const     {
    return SiPixelRecHitQuality::thePacking.hasBadPixels( qualWord_ );
  }
  //--- The cluster spans two ROCS (and thus contains big pixels)
  inline bool spansTwoROCs() const     {
    return SiPixelRecHitQuality::thePacking.spansTwoROCs( qualWord_ );
  }

  
  //--- Setters for the above
  inline void setCotAlphaFromCluster( float cotalpha ) {
    SiPixelRecHitQuality::thePacking.setCotAlphaFromCluster( cotalpha, qualWord_ );
  }
  inline void setCotBetaFromCluster ( float cotbeta ) {
    SiPixelRecHitQuality::thePacking.setCotBetaFromCluster( cotbeta, qualWord_ );
  }
  inline void setProbabilityX( float prob ) {
    SiPixelRecHitQuality::thePacking.setProbabilityX( prob, qualWord_ );
  }
  inline void setProbabilityY( float prob ) {
    SiPixelRecHitQuality::thePacking.setProbabilityY( prob, qualWord_ );
  }  
  inline void setQBin( int qbin ) {
    SiPixelRecHitQuality::thePacking.setQBin( qbin, qualWord_ );
  }
  inline void setIsOnEdge( bool flag ) {
    SiPixelRecHitQuality::thePacking.setIsOnEdge( flag, qualWord_ );
  }
  inline void setHasBadPixels( bool flag ) {
    SiPixelRecHitQuality::thePacking.setHasBadPixels( flag, qualWord_ );
  }
  inline void setSpansTwoROCs( bool flag ) {
    SiPixelRecHitQuality::thePacking.setSpansTwoROCs( flag, qualWord_ );
  }


private:

  SiPixelClusterRefNew cluster_;

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
