#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterEnergyCorrectionBaseClass.h"

#include "CondFormats/DataRecord/interface/EcalClusterEnergyCorrectionParametersRcd.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

EcalClusterEnergyCorrectionBaseClass::EcalClusterEnergyCorrectionBaseClass()
{}

EcalClusterEnergyCorrectionBaseClass::~EcalClusterEnergyCorrectionBaseClass()
{}

void
EcalClusterEnergyCorrectionBaseClass::init( const edm::EventSetup& es )
{
        es.get<EcalClusterEnergyCorrectionParametersRcd>().get( esParams_ );
        params_ = esParams_.product();
}

void
EcalClusterEnergyCorrectionBaseClass::checkInit() const
{
        if ( ! params_ ) {
                // non-initialized function parameters: throw exception
                throw cms::Exception("EcalClusterEnergyCorrectionBaseClass::checkInit()") 
                        << "Trying to access an uninitialized crack correction function.\n"
                        "Please call `init( edm::EventSetup &)' before any use of the function.\n";
        }
}
