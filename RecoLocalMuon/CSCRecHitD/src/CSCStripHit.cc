#include <RecoLocalMuon/CSCRecHitD/src/CSCStripHit.h>
#include <iostream>

CSCStripHit::CSCStripHit() :
  theDetId(),
  theStripHitPosition(),
  theStripHitTmax(),     
  theStrips(),
  theStripHitADCs(),
  theConsecutiveStrips(),
  theClosestMaximum()
{}

CSCStripHit::CSCStripHit( const CSCDetId& id, 
                          const float& sHitPos, 
                          const int& tmax, 
                          const ChannelContainer& strips, 
                          const StripHitADCContainer& s_adc,
			  const int& numberOfConsecutiveStrips,
                          const int& closestMaximum) :
  theDetId( id ), 
  theStripHitPosition( sHitPos ),
  theStripHitTmax( tmax ),
  theStrips( strips ),
  theStripHitADCs( s_adc ),
  theConsecutiveStrips(numberOfConsecutiveStrips),
  theClosestMaximum(closestMaximum)
{}

CSCStripHit::~CSCStripHit() {}


