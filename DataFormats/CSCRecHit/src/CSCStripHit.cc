#include <DataFormats/CSCRecHit/interface/CSCStripHit.h>
#include <iostream>

CSCStripHit::CSCStripHit() :
  theDetId(),
  theStripHitPosition()     
{}

CSCStripHit::CSCStripHit( const CSCDetId& id, const float& strip_pos) :
  theDetId( id ), 
  theStripHitPosition( strip_pos )
{}

CSCStripHit::~CSCStripHit() {}


