#include <DataFormats/CSCRecHit/interface/CSCWireHit.h>
#include <iostream>

CSCWireHit::CSCWireHit() :
  theDetId(),
  theWireHitPosition()     
{}

CSCWireHit::CSCWireHit( const CSCDetId& id, const int& wire_group) :
  theDetId( id ), 
  theWireHitPosition( wire_group )
{}

CSCWireHit::~CSCWireHit() {}


