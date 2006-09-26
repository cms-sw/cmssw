#include <DataFormats/CSCRecHit/interface/CSCStripHit.h>
#include <iostream>

CSCStripHit::CSCStripHit() :
  theDetId(),
  theStripHitPosition(),
  theStripHitTmax(),     
  theStripHitClusterSize(),
  theStripHitADCs()
{}

CSCStripHit::CSCStripHit( const CSCDetId& id, const float& sHitPos, const int& tmax, 
                          const int& clusterSize, const StripHitADCContainer& s_adc) :
  theDetId( id ), 
  theStripHitPosition( sHitPos ),
  theStripHitTmax( tmax ),
  theStripHitClusterSize( clusterSize ),
  theStripHitADCs( s_adc )
{}

CSCStripHit::~CSCStripHit() {}


