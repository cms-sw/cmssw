#include <RecoLocalMuon/CSCRecHitD/src/CSCStripHit.h>
#include <iostream>

CSCStripHit::CSCStripHit() :
  theDetId(),
  theStripHitPosition(),
  theStripHitTmax(),     
  theStrips(),
  theStripHitADCs()
{}

CSCStripHit::CSCStripHit( const CSCDetId& id, 
                          const float& sHitPos, 
                          const int& tmax, 
                          const ChannelContainer& strips, 
                          const StripHitADCContainer& s_adc ) :
  theDetId( id ), 
  theStripHitPosition( sHitPos ),
  theStripHitTmax( tmax ),
  theStrips( strips ),
  theStripHitADCs( s_adc )
{}

CSCStripHit::~CSCStripHit() {}


