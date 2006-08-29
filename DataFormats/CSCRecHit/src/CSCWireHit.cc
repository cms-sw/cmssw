#include <DataFormats/CSCRecHit/interface/CSCWireHit.h>
#include <iostream>

CSCWireHit::CSCWireHit() :
  theDetId(),
  theHitWirePosition(),
  theHitTmax()
{}

CSCWireHit::CSCWireHit( const CSCDetId& id, const float& wgrouppos, const int& tmax) :
  theDetId( id ), 
  theHitWirePosition( wgrouppos ),
  theHitTmax ( tmax )
{}

CSCWireHit::~CSCWireHit() {}


