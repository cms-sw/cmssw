#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterEnergyUncertaintyBaseClass.h"

#include "CondFormats/DataRecord/interface/EcalClusterEnergyUncertaintyParametersRcd.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

EcalClusterEnergyUncertaintyBaseClass::EcalClusterEnergyUncertaintyBaseClass()
{}

EcalClusterEnergyUncertaintyBaseClass::~EcalClusterEnergyUncertaintyBaseClass()
{}

void
EcalClusterEnergyUncertaintyBaseClass::init( const edm::EventSetup& es )
{
        es.get<EcalClusterEnergyUncertaintyParametersRcd>().get( esParams_ );
        params_ = esParams_.product();
}

void
EcalClusterEnergyUncertaintyBaseClass::checkInit() const
{
        if ( ! params_ ) {
                // non-initialized function parameters: throw exception
                throw cms::Exception("EcalClusterEnergyUncertaintyBaseClass::checkInit()") 
                        << "Trying to access an uninitialized crack correction function.\n"
                        "Please call `init( edm::EventSetup &)' before any use of the function.\n";
        }
}
