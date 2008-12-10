#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterCrackCorrection.h"


float EcalClusterCrackCorrection::getValue( const reco::BasicCluster & basicCluster, const EcalRecHitCollection & recHit) const
{
        checkInit();
        // private member params_ = EcalClusterCrackCorrectionParameters
        // (see in CondFormats/EcalObjects/interface)
        EcalClusterCrackCorrParameters::const_iterator it;
        std::cout << "[[EcalClusterCrackCorrectionBaseClass::getValue]] " 
                << params_->size() << " parameters:";
        for ( it = params_->begin(); it != params_->end(); ++it ) {
                std::cout << " " << *it;
        }
        std::cout << "\n";
        return 1;
}

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"
DEFINE_EDM_PLUGIN( EcalClusterFunctionFactory, EcalClusterCrackCorrection, "EcalClusterCrackCorrection");
