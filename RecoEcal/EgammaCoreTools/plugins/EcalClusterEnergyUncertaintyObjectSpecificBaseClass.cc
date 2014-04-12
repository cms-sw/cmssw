#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterEnergyUncertaintyObjectSpecificBaseClass.h"

//#include "CondFormats/DataRecord/interface/EcalClusterEnergyUncertaintyObjectSpecificParametersRcd.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

EcalClusterEnergyUncertaintyObjectSpecificBaseClass::EcalClusterEnergyUncertaintyObjectSpecificBaseClass()
{}

EcalClusterEnergyUncertaintyObjectSpecificBaseClass::~EcalClusterEnergyUncertaintyObjectSpecificBaseClass()
{}

void
EcalClusterEnergyUncertaintyObjectSpecificBaseClass::init( const edm::EventSetup& es )
{
  //es.get<EcalClusterEnergyUncertaintyParametersRcd>().get( esParams_ );
  //params_ = esParams_.product();
}

void
EcalClusterEnergyUncertaintyObjectSpecificBaseClass::checkInit() const
{
  /*
        if ( ! params_ ) {
                // non-initialized function parameters: throw exception
                throw cms::Exception("EcalClusterEnergyUncertaintyObjectSpecificBaseClass::checkInit()") 
                        << "Trying to access an uninitialized crack correction function.\n"
                        "Please call `init( edm::EventSetup &)' before any use of the function.\n";
        }
  */
}
