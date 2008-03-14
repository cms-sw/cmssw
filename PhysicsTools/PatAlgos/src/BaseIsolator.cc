#include "PhysicsTools/PatAlgos/interface/BaseIsolator.h"
#include <sstream>
#include <iomanip>

using pat::helper::BaseIsolator;

BaseIsolator::BaseIsolator(const edm::ParameterSet &conf) :
    input_(conf.getParameter<edm::InputTag>("src")),
    cut_(conf.getParameter<double>("cut")),
    try_(0), fail_(0)
{
}

void 
BaseIsolator::print(std::ostream &out) const {
    using namespace std;
    out << description() << " < " << cut_ << ": try " << try_ << ", fail " << fail_;
}

