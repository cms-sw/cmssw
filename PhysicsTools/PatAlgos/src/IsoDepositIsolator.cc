#include "PhysicsTools/PatAlgos/interface/IsoDepositIsolator.h"
#include <sstream>

#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositVetoFactory.h"

#include <boost/regex.hpp>

using pat::helper::IsoDepositIsolator;
using pat::helper::BaseIsolator;
using namespace reco::isodeposit;

IsoDepositIsolator::IsoDepositIsolator(const edm::ParameterSet &conf, edm::ConsumesCollector & iC, bool withCut) :
    BaseIsolator(conf,iC,withCut), deltaR_(conf.getParameter<double>("deltaR")),
    mode_(Sum), skipDefaultVeto_(false),
    inputIsoDepositToken_(iC.consumes<Isolation>(input_))
{
    if (conf.exists("mode")) {
        std::string mode = conf.getParameter<std::string>("mode");
        if (mode == "sum") mode_ = Sum;
        else if (mode == "sumRelative") mode_ = SumRelative;
        else if (mode == "max") mode_ = Max;
        else if (mode == "maxRelative") mode_ = MaxRelative;
        else if (mode == "sum2") mode_ = Sum2;
        else if (mode == "sum2Relative") mode_ = Sum2Relative;
        else if (mode == "count") mode_ = Count;
        else throw cms::Exception("Not Implemented") << "Mode '" << mode << "' not implemented. " <<
                "Supported modes are 'sum', 'sumRelative', 'max', 'maxRelative', 'sum2', 'sum2Relative', 'count'." <<
                "New methods can be easily implemented if requested.";
    }

    if (conf.exists("veto")) {
        vetos_.push_back(new ConeVeto(Direction(), conf.getParameter<double>("veto")));
    }
    if (conf.exists("threshold")) {
        vetos_.push_back(new ThresholdVeto(conf.getParameter<double>("threshold")));
    }
    if (conf.exists("skipDefaultVeto")) {
        skipDefaultVeto_ = conf.getParameter<bool>("skipDefaultVeto");
    }

    if (conf.exists("vetos")) { // expert configuration
        if (!vetos_.empty())
            throw cms::Exception("Configuration") << "You can't both configure this module with 'veto'/'threshold' AND with 'vetos'!";
        if (!conf.exists("skipDefaultVeto"))
            throw cms::Exception("Configuration") << "When using the expert configuration variable 'vetos' you must specify the value for 'skipDefaultVeto' too.";

        typedef std::vector<std::string> vstring;
        vstring vetos = conf.getParameter< vstring >("vetos");
        reco::isodeposit::EventDependentAbsVeto *evdep = 0;
        for (vstring::const_iterator it = vetos.begin(), ed = vetos.end(); it != ed; ++it) {
              vetos_.push_back( IsoDepositVetoFactory::make( it->c_str(), evdep, iC ) );
              if (evdep != 0) evdepVetos_.push_back(evdep);
        }
    }

}

IsoDepositIsolator::~IsoDepositIsolator() {
    for (AbsVetos::iterator it = vetos_.begin(), ed = vetos_.end(); it != ed; ++it) {
        delete *it;
    }
}

void
IsoDepositIsolator::beginEvent(const edm::Event &event, const edm::EventSetup &eventSetup) {
    event.getByToken(inputIsoDepositToken_, handle_);
    for (EventDependentAbsVetos::iterator it = evdepVetos_.begin(), ed = evdepVetos_.end(); it != ed; ++it) {
        (*it)->setEvent(event,eventSetup);
    }
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

float
IsoDepositIsolator::getValue(const edm::ProductID &id, size_t index) const {
    const reco::IsoDeposit &dep = handle_->get(id, index);

    double eta = dep.eta(), phi = dep.phi(); // better to center on the deposit direction that could be, e.g., the impact point at calo
    for (AbsVetos::const_iterator it = vetos_.begin(), ed = vetos_.end(); it != ed; ++it) {
        (const_cast<AbsVeto *>(*it))->centerOn(eta, phi); // I need the const_cast to be able to 'move' the veto
    }
    switch (mode_) {
        case Count:        return dep.countWithin(deltaR_, vetos_, skipDefaultVeto_);
        case Sum:          return dep.sumWithin(deltaR_, vetos_, skipDefaultVeto_);
        case SumRelative:  return dep.sumWithin(deltaR_, vetos_, skipDefaultVeto_) / dep.candEnergy() ;
        case Sum2:         return dep.sum2Within(deltaR_, vetos_, skipDefaultVeto_);
        case Sum2Relative: return dep.sum2Within(deltaR_, vetos_, skipDefaultVeto_) / (dep.candEnergy() * dep.candEnergy()) ;
        case Max:          return dep.maxWithin(deltaR_, vetos_, skipDefaultVeto_);
        case MaxRelative:  return dep.maxWithin(deltaR_, vetos_, skipDefaultVeto_) / dep.candEnergy() ;
    }
    throw cms::Exception("Logic error") << "Should not happen at " << __FILE__ << ", line " << __LINE__; // avoid gcc warning
}

