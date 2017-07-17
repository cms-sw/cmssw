#include "PhysicsTools/PatAlgos/interface/SimpleIsolator.h"
#include <sstream>

using pat::helper::SimpleIsolator;
using pat::helper::BaseIsolator;


SimpleIsolator::SimpleIsolator(const edm::ParameterSet &conf, edm::ConsumesCollector & iC, bool withCut) :
    BaseIsolator(conf, iC, withCut)
{
}

void
SimpleIsolator::beginEvent(const edm::Event &event, const edm::EventSetup &eventSetup) {
    event.getByToken(inputDoubleToken_, handle_);
}

void
SimpleIsolator::endEvent() {
    handle_.clear();
}

