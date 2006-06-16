#ifndef DataFormats_RPCRecHit_H
#define DataFormats_RPCRecHit_H

/** \class RPCRecHit
 *
 *  RecHit for RPC 
 *
 *  $Date: 2006/06/15 16:36:24 $
 *  $Revision: 1.3 $
 *  \author M. Maggi -- INFN Bari 
 */

#include "DataFormats/TrackingRecHit/interface/RecHit1D.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"


class RPCRecHit : public RecHit1D {
 public:

  RPCRecHit(const RPCDetId& rpcId,
	    int bx);

  RPCRecHit(const RPCDetId& rpcId,
	    int bx,int multiplicity);
  /// Default constructor
  RPCRecHit();

  /// Constructor from a local position, rpcId and digi time.
  /// The 3-dimensional local error is defined as
  /// resolution (the cell resolution) for the coordinate being measured
  /// and 0 for the two other coordinates
  RPCRecHit(const RPCDetId& rpcId,
	    int bx,int multiplicity,
	    const LocalPoint& pos);
  

  /// Constructor from a local position and error, rpcId and bx.
  RPCRecHit(const RPCDetId& rpcId,
	    int bx,int multiplicity,
	    const LocalPoint& pos,
	    const LocalError& err);
  

  /// Constructor from a local position and error, rpcId, bx and cluster size.
  RPCRecHit(const RPCDetId& rpcId,
	    int bx,
	    int clustSize,
	    const LocalPoint& pos,
	    const LocalError& err);
  
  /// Destructor
  virtual ~RPCRecHit();


  /// Return the 3-dimensional local position
  virtual LocalPoint localPosition() const {
    return theLocalPosition;
  }


  /// Return the 3-dimensional error on the local position
  virtual LocalError localPositionError() const {
    return theLocalError;
  }

  /// Return the detId of the Det 
  virtual DetId geographicalId() const;


  virtual RPCRecHit* clone() const;

  
  /// Access to component RecHits.
  /// No components rechits: it returns a null vector
  virtual std::vector<const TrackingRecHit*> recHits() const;


  /// Non-const access to component RecHits.
  /// No components rechits: it returns a null vector
  virtual std::vector<TrackingRecHit*> recHits();


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
  

  /// Return the rpcId
  RPCDetId rpcId() const {
    return theRPCId;
  }
 
  int BunchX() const {
    return theBx;
  }

  int clusterSize() const {
    return theClusterSize;
  }

  /// Comparison operator, based on the rpcId and the digi time
  bool operator==(const RPCRecHit& hit) const;

 private:
  RPCDetId theRPCId;
  int theBx;
  int theClusterSize;
  // Position and error in the Local Ref. Frame of the RPCLayer
  LocalPoint theLocalPosition;
  LocalError theLocalError;

};
#endif

/// The ostream operator
std::ostream& operator<<(std::ostream& os, const RPCRecHit& hit);
