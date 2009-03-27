#include <vector>
#include <string>

#include "DataFormats/PatCandidates/interface/Jet.h"

#include "PhysicsTools/PatUtils/interface/bJetSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"

// BTag selector implements the most primitive object selector and filter
struct bJetFilterSelector {

    bJetFilterSelector(const edm::ParameterSet & config) : bTagger_(config) {
        operatingPoint_ = config.getParameter<std::string>("operatingPoint");
        tagger_ = config.getParameter<std::string>("tagger");
    }

    template<typename T>
    bool operator()(const T & t) const {
        return bTagger_.IsbTag(t, operatingPoint_, tagger_);
    }

private:

    std::string operatingPoint_, tagger_;
    bJetSelector bTagger_;

};


// PArameter adapterfor BTagSelector is trivial, it is mostly to keep compatibility in the FW.
namespace reco {
namespace modules {
template<>
struct ParameterAdapter<bJetFilterSelector> {
    static bJetFilterSelector make(const edm::ParameterSet & config) {
        return bJetFilterSelector(config);
    }
};
}
}


#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace pat {
typedef SingleObjectSelector<std::vector<Jet>, bJetFilterSelector> bJetFilter;
DEFINE_FWK_MODULE(bJetFilter);
}
