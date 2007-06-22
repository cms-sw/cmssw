#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <iostream>

CSCRecHit2D::CSCRecHit2D() :
  theDetId(),
  theLocalPosition(0.,0.), 
  theLocalError(0.,0.,0.),
  theChaCo( ChannelContainer() ),
  theADCs( ADCContainer() ),
  theWireGroups( ChannelContainer() ),
  theTpeak( -999. ),  
  theChi2( -1. ), 
  theProb( 0. )
{}

CSCRecHit2D::CSCRecHit2D( const CSCDetId& id, 
                          const LocalPoint& pos, 
                          const LocalError& err, 
	                  const ChannelContainer& channels, 
                          const ADCContainer& adcs,
                          const ChannelContainer& wgroups,
                          const float tpeak, 
	                  float chi2, 
                          float prob ) :
  theDetId( id ), 
  theLocalPosition( pos ), 
  theLocalError( err ),
  theChaCo( channels ),
  theADCs( adcs ),
  theWireGroups( wgroups ),
  theTpeak( tpeak ),
  theChi2( chi2 ), 
  theProb( prob )
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

std::ostream& operator<<(std::ostream& os, const CSCRecHit2D& rh) {
  os << "CSCRecHit2D: local x = " << rh.localPosition().x() << " +/- " << sqrt( rh.localPositionError().xx() ) <<
    " y = " << rh.localPosition().y() << " +/- " << sqrt( rh.localPositionError().yy() ) <<
    " chi2 = " << rh.chi2() << " prob = " << rh.prob();
  return os;
}
