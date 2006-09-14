#include <DataFormats/CSCRecHit/interface/CSCWireHit.h>
#include <iostream>

CSCWireHit::CSCWireHit() :
  theDetId(),
  theWireHitPosition(),
  theWireHitTmax()
{}

CSCWireHit::CSCWireHit( const CSCDetId& id, const float& wHitPos, const int& tmax) :
  theDetId( id ), 
  theWireHitPosition( wHitPos ),
  theWireHitTmax ( tmax )
{}

CSCWireHit::~CSCWireHit() {}


