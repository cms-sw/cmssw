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
#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"
//! Quality word packing
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitQuality.h"

#include "TkCloner.h"

class SiPixelRecHit final : public TrackerSingleRecHit {
  
public:
  
  typedef edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster > ClusterRef;
  
  SiPixelRecHit(){}
  
  ~SiPixelRecHit() override{}

  SiPixelRecHit( const LocalPoint& pos , const LocalError& err, 
		 SiPixelRecHitQuality::QualWordType qual,
		 GeomDet const & idet,
		 ClusterRef const&  clus) : 
    TrackerSingleRecHit(pos,err,idet, clus){
    qualWord_=qual; }

  bool isPixel() const override { return true;}

  
  SiPixelRecHit * clone() const override {return new SiPixelRecHit( * this); }
#ifndef __GCCXML__
  RecHitPointer cloneSH() const override { return std::make_shared<SiPixelRecHit>(*this);}
#endif

  
  ClusterRef cluster()  const { return cluster_pixel(); }

  void setClusterRef(ClusterRef const & ref)  {setClusterPixelRef(ref);}

  int dimension() const override {return 2;}
  void getKfComponents( KfComponentsHolder & holder ) const override { getKfComponents2D(holder); }
  
  
  bool canImproveWithTrack() const override {return true;}
private:
  // double dispatch
  SiPixelRecHit * clone_(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const override {
    return cloner(*this,tsos).release();
  }
#ifndef __GCCXML__
   RecHitPointer cloneSH_(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const override {
    return cloner.makeShared(*this,tsos);
  }
#endif  
  
public:
  //--- The overall probability.  flags is the 32-bit-packed set of flags that
  //--- our own concrete implementation of clusterProbability() uses to direct
  //--- the computation based on the information stored in the quality word
  //--- (and which was computed by the CPE).  The default of flags==0 returns
  //--- probabilityY() only (as that's the safest thing to do).
  //--- Flags are static and kept in the transient rec hit.
  using BaseTrackerRecHit::clusterProbability;
  float clusterProbability(unsigned int flags = 0) const;
  
  
  //--- Allow direct access to the packed quality information.
  inline SiPixelRecHitQuality::QualWordType rawQualityWord() const { 
    return qualWord_ ; 
  }

 
  //--- Template fit probability, in X and Y directions
  //--- These are obsolete and will be taken care of in the quality code
  inline float probabilityX() const     {
    return SiPixelRecHitQuality::thePacking.probabilityX( qualWord_ );
  }
  inline float probabilityY() const     {
    return SiPixelRecHitQuality::thePacking.probabilityY( qualWord_ );
  }

  //--- Template fit probability, in X and Y direction combined and in charge
  inline float probabilityXY() const     {
    return SiPixelRecHitQuality::thePacking.probabilityXY( qualWord_ );
  }
  inline float probabilityQ() const     {
    return SiPixelRecHitQuality::thePacking.probabilityQ( qualWord_ );
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

  //--- Quality flag for whether the probability is filled
  inline bool hasFilledProb() const {
    return SiPixelRecHitQuality::thePacking.hasFilledProb( qualWord_ );
  }

};

#endif
