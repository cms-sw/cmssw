#include <DataFormats/CSCRecHit/interface/CSCWireHit.h>
#include <iostream>

CSCWireHit::CSCWireHit() :
  theDetId(),
  theHitWirePosition(),
  theHitTmax()
{}

CSCWireHit::CSCWireHit( const CSCDetId& id, const int& wgroup, const int& tmax) :
  theDetId( id ), 
  theHitWirePosition( wgroup ),
  theHitTmax ( tmax )
{}

CSCWireHit::~CSCWireHit() {}


