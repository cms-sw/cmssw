#ifndef DataFormats_GEMRecHit_H
#define DataFormats_GEMRecHit_H

/** \class GEMRecHit
 *
 *  RecHit for GEM 
 *
 *  \author M. Maggi -- INFN Bari 
 */

#include "DataFormats/TrackingRecHit/interface/RecHit2DLocalPos.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"


class GEMRecHit : public RecHit2DLocalPos {
 public:

  GEMRecHit(const GEMDetId& gemId,
	    int bx);

  /// Default constructor
  GEMRecHit();

  /// Constructor from a local position, gemId and digi time.
  /// The 3-dimensional local error is defined as
  /// resolution (the cell resolution) for the coordinate being measured
  /// and 0 for the two other coordinates
  GEMRecHit(const GEMDetId& gemId,
	    int bx,
	    const LocalPoint& pos);
  

  /// Constructor from a local position and error, gemId and bx.
  GEMRecHit(const GEMDetId& gemId,
	    int bx,
	    const LocalPoint& pos,
	    const LocalError& err);
  

  /// Constructor from a local position and error, gemId, bx, frist strip of cluster and cluster size.
  GEMRecHit(const GEMDetId& gemId,
	    int bx,
	    int firstStrip,
	    int clustSize,
	    const LocalPoint& pos,
	    const LocalError& err);
  
  /// Destructor
  ~GEMRecHit() override;


  /// Return the 3-dimensional local position
  LocalPoint localPosition() const override {
    return theLocalPosition;
  }


  /// Return the 3-dimensional error on the local position
  LocalError localPositionError() const override {
    return theLocalError;
  }


  GEMRecHit* clone() const override;

  
  /// Access to component RecHits.
  /// No components rechits: it returns a null vector
  std::vector<const TrackingRecHit*> recHits() const override;


  /// Non-const access to component RecHits.
  /// No components rechits: it returns a null vector
  std::vector<TrackingRecHit*> recHits() override;


  /// Set local position 
  void setPosition(LocalPoint pos) {
    theLocalPosition = pos;
  }

  
  /// Set local position error
  void setError(LocalError err) {
    theLocalError = err;
  }


  /// Set the local position and its error
  void setPositionAndError(LocalPoint pos, LocalError err) {
    theLocalPosition = pos;
    theLocalError = err;
  }
  

  /// Return the gemId
  GEMDetId gemId() const {
    return theGEMId;
  }
 
  int BunchX() const {
    return theBx;
  }

  int firstClusterStrip() const {
    return theFirstStrip;
  }

  int clusterSize() const {
    return theClusterSize;
  }

  /// Comparison operator, based on the gemId and the digi time
  bool operator==(const GEMRecHit& hit) const;

 private:
  GEMDetId theGEMId;
  int theBx;
  int theFirstStrip;
  int theClusterSize;
  // Position and error in the Local Ref. Frame of the GEMLayer
  LocalPoint theLocalPosition;
  LocalError theLocalError;

};
#endif

/// The ostream operator
std::ostream& operator<<(std::ostream& os, const GEMRecHit& hit);
