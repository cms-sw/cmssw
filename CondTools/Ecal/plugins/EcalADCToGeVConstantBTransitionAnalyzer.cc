#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondTools/RunInfo/interface/BTransitionAnalyzer.h"
#include <sstream>

class EcalADCToGeVConstantBTransitionAnalyzer: public cond::BTransitionAnalyzer<EcalADCToGeVConstant, EcalADCToGeVConstantRcd> {
public:
  EcalADCToGeVConstantBTransitionAnalyzer( edm::ParameterSet const & pset ):
    cond::BTransitionAnalyzer<EcalADCToGeVConstant, EcalADCToGeVConstantRcd>( pset )  {}
  bool equalPayloads( edm::ESHandle<EcalADCToGeVConstant> const & payloadHandle, edm::ESHandle<EcalADCToGeVConstant> const & payloadRefHandle ) {
    bool areEquals = false;
    std::ostringstream os;
    os << "[" << "EcalADCToGeVConstantBTransitionAnalyzer::" << __func__ << "]: " << "Payload extracted starting from magnetic field value: ";
    payloadHandle->print( os );
    os << "\nReference payload from the target tag: ";
    payloadRefHandle->print( os );
    edm::LogInfo( "EcalADCToGeVConstantBTransitionAnalyzer" ) << os.str();
    if( payloadHandle->getEBValue() == payloadRefHandle->getEBValue() &&
        payloadHandle->getEEValue() == payloadRefHandle->getEEValue() ) areEquals = true;
    return areEquals;
  }
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EcalADCToGeVConstantBTransitionAnalyzer);
