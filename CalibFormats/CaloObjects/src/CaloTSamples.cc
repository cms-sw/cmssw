#include "CalibFormats/CaloObjects/interface/CaloTSamplesBase.icc"
#include "CalibFormats/CaloObjects/interface/CaloTSamples.icc"
#include "CondFormats/EcalObjects/interface/EcalConstants.h"

template class CaloTSamplesBase<float>;

template class CaloTSamples<float, ecalPh2::sampleSize>;
template class CaloTSamples<float, 3>;
template class CaloTSamples<float, 10>;
