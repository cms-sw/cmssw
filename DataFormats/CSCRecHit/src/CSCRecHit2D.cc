#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>

CSCRecHit2D::CSCRecHit2D() :
  theDetId( 0 ),
  theLocalPosition(0.,0.), theLocalError(0.,0.,0.),
  theChaCo( ChannelContainer() ),     
  theChi2( -1. ), theProb( 0. )
{}

CSCRecHit2D::CSCRecHit2D( const DetId& id, 
               const LocalPoint& pos, const LocalError& err, 
	       const ChannelContainer& channels,
	       float chi2, float prob ) :
  theDetId( id ), 
  theLocalPosition( pos ), theLocalError( err ),
  theChaCo( channels ),
  theChi2(chi2), theProb(prob)
{}

CSCRecHit2D::~CSCRecHit2D() {}

//GlobalPoint CSCRecHit2D::globalPosition() const {
//  return det().toGlobal(localPosition());
//}

//bool CSCRecHit2D::nearby(const CSCRecHit2D& other, float maxDeltaRPhi) {
//  //  float dphi = deltaPhi(phi(), other.phi());
//  float dy = abs(localPosition().x() - other.localPosition().x());
//  return dy < maxDeltaRPhi;
//}

//bool CSCRecHit2D::nearby(float otherX, float maxDeltaRPhi) {
//  //  float dphi = deltaPhi(phi(), otherPhi);
//  float dy = abs(localPosition().x() - otherX);
//  return dy < maxDeltaRPhi;
//}

