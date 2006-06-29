#ifndef MuonTransientTrackingRecHit_h
#define MuonTransientTrackingRecHit_h

/** \class MuonTransientTrackingRecHit
 *
 *  A TransientTrackingRecHit for muons.
 *
 *  $Date: $
 *  $Revision: $
 */


#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"


class MuonTransientTrackingRecHit: public GenericTransientTrackingRecHit{
public:

  /// Construct from a TrackingRecHit and its GeomDet
  MuonTransientTrackingRecHit(const GeomDet * geom, const TrackingRecHit * rh);

  /// Copy ctor
  MuonTransientTrackingRecHit( const MuonTransientTrackingRecHit & other );
  
  virtual ~MuonTransientTrackingRecHit(){}

  virtual MuonTransientTrackingRecHit* clone() const {
    return new MuonTransientTrackingRecHit(*this);
  }

  /// Direction in 3D for segments, otherwise (0,0,0)
  virtual LocalVector localDirection() const;

  /// Direction in 3D for segments, otherwise (0,0,0)
  virtual GlobalVector globalDirection() const;

  /// Error on the local direction
  virtual LocalError localDirectionError() const;

  /// Error on the global direction
  virtual GlobalError globalDirectionError() const;
 
  /// Chi square of the fit for segments, else 0
  virtual double chi2() const;

  /// Degrees of freedom for segments, else 0
  virtual int degreesOfFreedom() const;

  /// if this rec hit is a DT rec hit 
  bool isDT() const;

  /// if this rec hit is a CSC rec hit 
  bool isCSC() const;
 
  /// if this rec hit is a RPC rec hit
  bool isRPC() const;

  /// return the sub components of this transient rechit
  edm::OwnVector<const TransientTrackingRecHit> transientHits() const;
  
 private:
};
#endif

