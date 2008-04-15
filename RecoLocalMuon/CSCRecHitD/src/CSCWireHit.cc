#include <RecoLocalMuon/CSCRecHitD/src/CSCWireHit.h>
#include <iostream>

CSCWireHit::CSCWireHit() :
  theDetId(),
  theWireHitPosition(),
  theWgroups(),
  theWireHitTmax(),
  isDeadWGAround()
{}

CSCWireHit::CSCWireHit( const CSCDetId& id, 
                        const float& wHitPos, 
                        ChannelContainer& wgroups, 
                        const int& tmax,
                        const bool& isNearDeadWG ) :
  theDetId( id ), 
  theWireHitPosition( wHitPos ),
  theWgroups( wgroups ),
  theWireHitTmax ( tmax ),
  isDeadWGAround( isNearDeadWG )
{}

CSCWireHit::~CSCWireHit() {}


