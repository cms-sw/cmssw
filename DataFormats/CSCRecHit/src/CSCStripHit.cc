#include <DataFormats/CSCRecHit/interface/CSCStripHit.h>
#include <iostream>

CSCStripHit::CSCStripHit() :
  theDetId(),
  theHitHalfStripPosition(),
  theHitTmax()     
{}

CSCStripHit::CSCStripHit( const CSCDetId& id, const float& halfStripPos, const int& tmax) :
  theDetId( id ), 
  theHitHalfStripPosition( halfStripPos ),
  theHitTmax( tmax )
{}

CSCStripHit::~CSCStripHit() {}


