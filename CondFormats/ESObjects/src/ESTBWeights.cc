#include "CondFormats/ESObjects/interface/ESTBWeights.h"
//
// defualt ctor creates vectors of length EBDataFrame::MAXSAMPLES==10
//
ESTBWeights::ESTBWeights() {

}

ESTBWeights::~ESTBWeights() {
}

void
ESTBWeights::setValue(const ESStripGroupId& groupId,
                        const ESWeightSet& weight) {
  map_.insert( std::make_pair(groupId ,weight) );
}


