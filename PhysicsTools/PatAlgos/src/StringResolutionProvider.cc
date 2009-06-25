#include "PhysicsTools/PatAlgos/interface/StringResolutionProvider.h"

#include "DataFormats/PatCandidates/interface/ParametrizationHelper.h"

#include <Math/Functions.h>

#include <map>

StringResolutionProvider::StringResolutionProvider(const edm::ParameterSet &iConfig) :
    constraints_() 
{ 
    using namespace std;
    typedef pat::CandKinResolution::Parametrization Parametrization;

    std::vector<double> cons = iConfig.getParameter< std::vector<double> > ("constraints");
    constraints_.insert(constraints_.end(), cons.begin(), cons.end());

    std::string parametrization(iConfig.getParameter< std::string > ("parametrization") );
    parametrization_ = pat::helper::ParametrizationHelper::fromString(parametrization);
 
    dimension_ = pat::helper::ParametrizationHelper::dimension(parametrization_); 
    std::vector< std::string > vstr = iConfig.getParameter < std::vector< std::string > >("resolutions");
    if (vstr.size() != static_cast<size_t>(dimension_)) {
        throw cms::Exception("StringResolutionProvider") << "Parameterization " << parametrization.c_str() << " needs " << dimension_ << " functions.";
    }
    for (int i = 0; i < dimension_; ++i) {
        resols_[i] = std::auto_ptr<Function>(new Function(vstr[i])); //std::auto_ptr<Function>( new Function(vstr[i]) );
    }
}

StringResolutionProvider::~StringResolutionProvider() { }
 
pat::CandKinResolution
StringResolutionProvider::getResolution(const reco::Candidate &c) const { 
    //AlgebraicVector4 vec = pat::helper::ParametrizationHelper::parametersFromP4(parametrization_, c.polarP4()); // polar is generally better
    //for (int i = dimension_; i < 4; ++i) { vec[i] = constraints_[i-dimension_]; }
    std::vector<pat::CandKinResolution::Scalar> covariances(dimension_);
    for (int i = 0; i < dimension_; ++i) {
        covariances[i] = ROOT::Math::Square(resols_[i]->operator()(c));
    }
    return pat::CandKinResolution(parametrization_, covariances, constraints_);
}
