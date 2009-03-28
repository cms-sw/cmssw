#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterCrackCorrectionBaseClass.h"

#include "CondFormats/DataRecord/interface/EcalClusterCrackCorrParametersRcd.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

EcalClusterCrackCorrectionBaseClass::EcalClusterCrackCorrectionBaseClass() :
        params_(0)
{}

EcalClusterCrackCorrectionBaseClass::~EcalClusterCrackCorrectionBaseClass()
{}

void
EcalClusterCrackCorrectionBaseClass::init( const edm::EventSetup& es )
{
        es.get<EcalClusterCrackCorrParametersRcd>().get( esParams_ );
        params_ = esParams_.product();
	es_ = &es; //needed to access the ECAL geometry

        //// check if parameters are retrieved correctly
        //EcalClusterCrackCorrParameters::const_iterator it;
        //std::cout << "[[EcalClusterCrackCorrectionBaseClass::init]] " 
        //        << params_->size() << " parameters:";
        //for ( it = params_->begin(); it != params_->end(); ++it ) {
        //        std::cout << " " << *it;
        //}
        //std::cout << "\n";
}

void
EcalClusterCrackCorrectionBaseClass::checkInit() const
{
        if ( ! params_ ) {
                // non initialized function parameters: throw exception
                throw cms::Exception("EcalClusterCrackCorrectionBaseClass::checkInit()") 
                        << "Trying to access an uninitialized crack correction function.\n"
                        "Please call `init( edm::EventSetup &)' before any use of the function.\n";
        }
}
