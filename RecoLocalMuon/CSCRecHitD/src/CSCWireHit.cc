#include <RecoLocalMuon/CSCRecHitD/src/CSCWireHit.h>
#include <iostream>

CSCWireHit::CSCWireHit() :
  theDetId(),
  theWireHitPosition(),
  theWgroups(),
  theWireHitTmax()
{}

CSCWireHit::CSCWireHit( const CSCDetId& id, 
                        const float& wHitPos, 
                        ChannelContainer& wgroups, 
                        const int& tmax ) :
  theDetId( id ), 
  theWireHitPosition( wHitPos ),
  theWgroups( wgroups ),
  theWireHitTmax ( tmax )
{}

CSCWireHit::~CSCWireHit() {}


