#include "DataFormats/PatCandidates/interface/Conversion.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace pat;

Conversion::Conversion( int index):
vtxProb_(0.0),
lxy_(0.0),
nHitsMax_(0)
{
  index_ = index;
}

void Conversion::setVtxProb( double vtxProb ){
  vtxProb_ = vtxProb;
}

void Conversion::setLxy( double lxy ){
  lxy_ = lxy;
}

void Conversion::setNHitsMax( int nHitsMax ){
  nHitsMax_ = nHitsMax;
}

