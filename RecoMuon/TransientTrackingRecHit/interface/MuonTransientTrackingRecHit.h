#ifndef MuonTransientTrackingRecHit_h
#define MuonTransientTrackingRecHit_h

/** \class MuonTransientTrackingRecHit
 *
 *  A TransientTrackingRecHit for muons.
 *
 *
 *   \author   C. Liu            Purdue University
 *
 *   \modified by C. Calabria    INFN & Universita  Bari
 */


#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"


class MuonTransientTrackingRecHit final : public GenericTransientTrackingRecHit{
public:
#if defined( __GXX_EXPERIMENTAL_CXX0X__)
   using MuonRecHitPointer = std::shared_ptr<MuonTransientTrackingRecHit>;
   using ConstMuonRecHitPointer = std::shared_ptr<MuonTransientTrackingRecHit const>;

#else
   typedef MuonTransientTrackingRecHit *           MuonRecHitPointer;
   typedef MuonTransientTrackingRecHit const *     ConstMuonRecHitPointer;
#endif

//  typedef ReferenceCountingPointer<MuonTransientTrackingRecHit>      MuonRecHitPointer;
//  typedef ConstReferenceCountingPointer<MuonTransientTrackingRecHit> ConstMuonRecHitPointer;
  typedef std::vector<MuonRecHitPointer>                             MuonRecHitContainer;
  typedef std::vector<ConstMuonRecHitPointer>                        ConstMuonRecHitContainer;
  
  virtual ~MuonTransientTrackingRecHit(){}

  /// Direction in 3D for segments, otherwise (0,0,0)
  virtual LocalVector localDirection() const;

  /// Direction in 3D for segments, otherwise (0,0,0)
  virtual GlobalVector globalDirection() const;

  /// Error on the local direction
  virtual LocalError localDirectionError() const;

  /// Error on the global direction
  virtual GlobalError globalDirectionError() const;
 
  virtual AlgebraicSymMatrix parametersError() const  override;

  /// Chi square of the fit for segments, else 0
  virtual double chi2() const;

  /// Degrees of freedom for segments, else 0
  virtual int degreesOfFreedom() const;

  /// if this rec hit is a DT rec hit 
  bool isDT() const;

  /// if this rec hit is a CSC rec hit 
  bool isCSC() const;

  /// if this rec hit is a GEM rec hit 
  bool isGEM() const; 
 
  /// if this rec hit is a RPC rec hit
  bool isRPC() const;

  /// return the sub components of this transient rechit
  virtual ConstRecHitContainer transientHits() const override;

  /// FIXME virtual ConstMuonRecHitContainer specificTransientHits() const;

  static RecHitPointer build( const GeomDet * geom, const TrackingRecHit* rh) {
    return RecHitPointer( new MuonTransientTrackingRecHit(geom, rh));
  }

  static MuonRecHitPointer specificBuild(const GeomDet * geom, const TrackingRecHit* rh) {
    return MuonRecHitPointer(new MuonTransientTrackingRecHit(geom, rh));
  }

  void invalidateHit();

 private:

  friend class kkkwwwxxxyyyzzz; //just to avoid the compiler warning...

  /// Construct from a TrackingRecHit and its GeomDet
  MuonTransientTrackingRecHit(const GeomDet * geom, const TrackingRecHit * rh);

  /// Copy ctor
  MuonTransientTrackingRecHit(const MuonTransientTrackingRecHit & other );

  virtual MuonTransientTrackingRecHit* clone() const  override {
    return new MuonTransientTrackingRecHit(*this);
  }

};
#endif

