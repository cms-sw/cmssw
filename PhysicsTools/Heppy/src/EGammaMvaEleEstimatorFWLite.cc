
#include "PhysicsTools/Heppy/interface/EGammaMvaEleEstimatorFWLite.h"
#include "EgammaAnalysis/ElectronTools/interface/EGammaMvaEleEstimator.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

EGammaMvaEleEstimatorFWLite::EGammaMvaEleEstimatorFWLite() :
    estimator_(0)
{
}

EGammaMvaEleEstimatorFWLite::~EGammaMvaEleEstimatorFWLite()
{
    delete estimator_;
}

void EGammaMvaEleEstimatorFWLite::initialize( std::string methodName,
        MVAType type,
        bool useBinnedVersion,
        std::vector<std::string> weightsfiles ) 
{
    EGammaMvaEleEstimator::MVAType pogType;
    switch(type) {
        case EGammaMvaEleEstimatorFWLite::kTrig: pogType = EGammaMvaEleEstimator::kTrig; break;
        case EGammaMvaEleEstimatorFWLite::kTrigNoIP: pogType = EGammaMvaEleEstimator::kTrigNoIP; break;
        case EGammaMvaEleEstimatorFWLite::kNonTrig: pogType = EGammaMvaEleEstimator::kNonTrig; break;
        default:
            return;
    }
    estimator_ = new EGammaMvaEleEstimator();
    std::vector<std::string> weightspaths;
    for (const std::string &s : weightsfiles) {
        weightspaths.push_back( edm::FileInPath(s).fullPath() );
    }
    estimator_->initialize(methodName, pogType, useBinnedVersion, weightspaths);
}

float EGammaMvaEleEstimatorFWLite::mvaValue(const pat::Electron& ele,
                const reco::Vertex& vertex,
                double rho,
                bool full5x5,
                bool printDebug)
{
  return -1.;
//FIXME
//    return estimator_->mvaValue(ele,vertex,rho,full5x5,printDebug);
}

