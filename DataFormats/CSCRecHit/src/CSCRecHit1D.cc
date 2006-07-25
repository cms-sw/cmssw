#include <DataFormats/CSCRecHit/interface/CSCRecHit1D.h>
#include <iostream>

CSCRecHit1D::CSCRecHit1D() :
  theDetId(),
  theChannel()     
{}

CSCRecHit1D::CSCRecHit1D( const CSCDetId& id, const float& channel) :
  theDetId( id ), 
  theChannel( channel )
{}

CSCRecHit1D::~CSCRecHit1D() {}


