/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: $
 **/

#include "CondFormats/EcalObjects/interface/EcalWeightRecAlgoWeights.h"
#include "CondFormats/EcalObjects/interface/EcalWeight.h"
//
// defualt ctor creates vectors of length EBDataFrame::MAXSAMPLES==10
//
EcalWeightRecAlgoWeights::EcalWeightRecAlgoWeights() {

// comment the following until POOL is fixed and released. The fix is currently in the HEAD of POOL
/**
  wgtBeforeSwitch_.push_back( std::vector<EcalWeight>(10,10.) );
  wgtBeforeSwitch_.push_back( std::vector<EcalWeight>(10,20.) );
  wgtBeforeSwitch_.push_back( std::vector<EcalWeight>(10,30.) );

  wgtAfterSwitch_.push_back( std::vector<EcalWeight>(10,110.) );
  wgtAfterSwitch_.push_back( std::vector<EcalWeight>(10,120.) );
  wgtAfterSwitch_.push_back( std::vector<EcalWeight>(10,130.) );

  wgtChi2_.push_back( std::vector<EcalWeight>(10,100.) );
  wgtChi2_.push_back( std::vector<EcalWeight>(10,200.) );
  wgtChi2_.push_back( std::vector<EcalWeight>(10,300.) );
  wgtChi2_.push_back( std::vector<EcalWeight>(10,400.) );
  wgtChi2_.push_back( std::vector<EcalWeight>(10,500.) );
  wgtChi2_.push_back( std::vector<EcalWeight>(10,600.) );
  wgtChi2_.push_back( std::vector<EcalWeight>(10,700.) );
  wgtChi2_.push_back( std::vector<EcalWeight>(10,800.) );
  wgtChi2_.push_back( std::vector<EcalWeight>(10,900.) );
  wgtChi2_.push_back( std::vector<EcalWeight>(10,1000.) );
**/
}

EcalWeightRecAlgoWeights::~EcalWeightRecAlgoWeights() {
}
