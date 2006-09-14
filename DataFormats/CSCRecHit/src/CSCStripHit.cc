#include <DataFormats/CSCRecHit/interface/CSCStripHit.h>
#include <iostream>

CSCStripHit::CSCStripHit() :
  theDetId(),
  theStripHitPosition(),
  theStripHitTmax(),     
  theStripHitTpeak(),     
  theStripHitClusterSize(),
  theStripHitADCs()
{}

CSCStripHit::CSCStripHit( const CSCDetId& id, const float& sHitPos, const int& tmax, 
                          const float& tpeak, const int& clusterSize, const StripHitADCContainer& s_adc) :
  theDetId( id ), 
  theStripHitPosition( sHitPos ),
  theStripHitTmax( tmax ),
  theStripHitTpeak( tpeak ),
  theStripHitClusterSize( clusterSize ),
  theStripHitADCs( s_adc )
{}

CSCStripHit::~CSCStripHit() {}


