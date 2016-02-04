#include "PhysicsTools/PatAlgos/interface/BaseIsolator.h"
#include <sstream>
#include <iomanip>

using pat::helper::BaseIsolator;

BaseIsolator::BaseIsolator(const edm::ParameterSet &conf, bool withCut) :
    input_(conf.getParameter<edm::InputTag>("src")),
    cut_(withCut ? conf.getParameter<double>("cut") : -2.0),
    try_(0), fail_(0)
{
}

void 
BaseIsolator::print(std::ostream &out) const {
    using namespace std;
    out << description() << " < " << cut_ << ": try " << try_ << ", fail " << fail_;
}

