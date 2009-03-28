#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterLocalContCorrectionBaseClass.h"

#include "CondFormats/DataRecord/interface/EcalClusterLocalContCorrParametersRcd.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

EcalClusterLocalContCorrectionBaseClass::EcalClusterLocalContCorrectionBaseClass()
{}

EcalClusterLocalContCorrectionBaseClass::~EcalClusterLocalContCorrectionBaseClass()
{}

void
EcalClusterLocalContCorrectionBaseClass::init( const edm::EventSetup& es )
{
        es.get<EcalClusterLocalContCorrParametersRcd>().get( esParams_ );
        params_ = esParams_.product();
	es_ = &es; //needed to access the ECAL geometry
}

void
EcalClusterLocalContCorrectionBaseClass::checkInit() const
{
        if ( ! params_ ) {
                // non-initialized function parameters: throw exception
                throw cms::Exception("EcalClusterLocalContCorrectionBaseClass::checkInit()") 
                        << "Trying to access an uninitialized crack correction function.\n"
                        "Please call `init( edm::EventSetup &)' before any use of the function.\n";
        }
}
