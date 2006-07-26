#include <DataFormats/CSCRecHit/interface/CSCStripHit.h>
#include <iostream>

CSCStripHit::CSCStripHit() :
  theDetId(),
  theHalfStripHitPosition()     
{}

CSCStripHit::CSCStripHit( const CSCDetId& id, const float& halfStripPos) :
  theDetId( id ), 
  theHalfStripHitPosition( halfStripPos )
{}

CSCStripHit::~CSCStripHit() {}


