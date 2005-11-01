#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
template std::map< int, EcalPedestals::Item >::iterator;
template std::map< int, EcalPedestals::Item >::const_iterator;

#include "CondFormats/EcalObjects/interface/EcalWeightRecAlgoWeights.h"
#include "CondFormats/EcalObjects/interface/EcalWeight.h"
namespace {
  namespace {
    std::vector< std::vector<EcalWeight> > vecOfVec0;
    std::vector<EcalWeight>  vec0;
  }
}
template  std::vector<EcalWeight>::iterator;
template  std::vector< std::vector<EcalWeight> >::iterator;
