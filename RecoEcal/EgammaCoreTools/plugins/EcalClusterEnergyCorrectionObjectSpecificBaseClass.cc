#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterEnergyCorrectionObjectSpecificBaseClass.h"

#include "CondFormats/DataRecord/interface/EcalClusterEnergyCorrectionObjectSpecificParametersRcd.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

EcalClusterEnergyCorrectionObjectSpecificBaseClass::EcalClusterEnergyCorrectionObjectSpecificBaseClass()
{}

EcalClusterEnergyCorrectionObjectSpecificBaseClass::~EcalClusterEnergyCorrectionObjectSpecificBaseClass()
{}

void
EcalClusterEnergyCorrectionObjectSpecificBaseClass::init( const edm::EventSetup& es )
{
  es.get<EcalClusterEnergyCorrectionObjectSpecificParametersRcd>().get( esParams_ );
  params_ = esParams_.product();
}

void
EcalClusterEnergyCorrectionObjectSpecificBaseClass::checkInit() const
{
  
        if ( ! params_ ) {
                // non-initialized function parameters: throw exception
                throw cms::Exception("EcalClusterEnergyCorrectionObjectSpecificBaseClass::checkInit()") 
                        << "Trying to access an uninitialized correction function.\n"
                        "Please call `init( edm::EventSetup &)' before any use of the function.\n";
        }
   
}
