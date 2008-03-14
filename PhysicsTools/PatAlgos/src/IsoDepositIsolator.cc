#include "PhysicsTools/PatAlgos/interface/IsoDepositIsolator.h"
#include <sstream>

using pat::helper::IsoDepositIsolator;
using pat::helper::BaseIsolator;
#include <sstream>

IsoDepositIsolator::IsoDepositIsolator(const edm::ParameterSet &conf) :
    BaseIsolator(conf), deltaR_(conf.getParameter<double>("deltaR"))
{
}

void
IsoDepositIsolator::beginEvent(const edm::Event &event) {
    event.getByLabel(input_, handle_);
}

void
IsoDepositIsolator::endEvent() {
    handle_.clear();
}

std::string 
IsoDepositIsolator::description() const {
    using namespace std;
    ostringstream oss;
    oss << input_.encode() << "(dR=" << deltaR_ <<")";
    return oss.str();
}
