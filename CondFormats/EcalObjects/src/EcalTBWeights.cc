/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: $
 **/

#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
//
// defualt ctor creates vectors of length EBDataFrame::MAXSAMPLES==10
//
EcalTBWeights::EcalTBWeights() {

}

EcalTBWeights::~EcalTBWeights() {
}

void
EcalTBWeights::setValue(const EcalXtalGroupId& groupId,
                        const EcalTDCId& tdcId,
                        const EcalWeightSet& weight) {
  setValue( std::make_pair<EcalXtalGroupId,EcalTDCId>(groupId,tdcId), weight);
}

void
EcalTBWeights::setValue(const std::pair<EcalXtalGroupId,EcalTDCId >& keyPair, const EcalWeightSet& weight) {
  map_.insert( std::make_pair(keyPair,weight) );
}

