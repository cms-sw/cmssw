#include <RecoLocalMuon/CSCRecHitD/src/CSCStripHit.h>
#include <iostream>

CSCStripHit::CSCStripHit() :
  theDetId(),
  theStripHitPosition(),
  theStripHitTmax(),     
  theStrips(),
  theStripHitADCs(),
  theStripHitRawADCs(),
  theConsecutiveStrips(),
  theClosestMaximum(),
  isDeadStripAround()
{}

CSCStripHit::CSCStripHit( const CSCDetId& id, 
                          const float& sHitPos, 
                          const int& tmax, 
                          const ChannelContainer& strips, 
                          const StripHitADCContainer& s_adc,
                          const StripHitADCContainer& s_adcRaw,
			  const int& numberOfConsecutiveStrips,
                          const int& closestMaximum,
                          const bool& isNearDeadStrip) :
  theDetId( id ), 
  theStripHitPosition( sHitPos ),
  theStripHitTmax( tmax ),
  theStrips( strips ),
  theStripHitADCs( s_adc ),
  theStripHitRawADCs( s_adcRaw ),
  theConsecutiveStrips(numberOfConsecutiveStrips),
  theClosestMaximum(closestMaximum),
  isDeadStripAround(isNearDeadStrip)
{}

CSCStripHit::~CSCStripHit() {}


