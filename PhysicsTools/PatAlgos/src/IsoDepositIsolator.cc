#include "PhysicsTools/PatAlgos/interface/IsoDepositIsolator.h"
#include <sstream>

#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"
#include <boost/regex.hpp>

using pat::helper::IsoDepositIsolator;
using pat::helper::BaseIsolator;
using namespace reco::isodeposit;

IsoDepositIsolator::IsoDepositIsolator(const edm::ParameterSet &conf, bool withCut) :
    BaseIsolator(conf,withCut), deltaR_(conf.getParameter<double>("deltaR")), 
    mode_(Sum), skipDefaultVeto_(false)
{
    if (conf.exists("mode")) {
        std::string mode = conf.getParameter<std::string>("mode");
        if (mode == "sum") mode_ = Sum;
        else if (mode == "sumRelative") mode_ = SumRelative;
        //else if (mode == "max") mode_ = Max;                  // TODO: on request only
        //else if (mode == "maxRelative") mode_ = MaxRelative;  // TODO: on request only
        else if (mode == "count") mode_ = Count;
        else throw cms::Exception("Not Implemented") << "Mode '" << mode << "' not implemented. " <<
                "Supported modes are 'sum', 'sumRelative', 'count'." <<
                //"Supported modes are 'sum', 'sumRelative', 'max', 'maxRelative', 'count'." << // TODO: on request only
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
        for (vstring::const_iterator it = vetos.begin(), ed = vetos.end(); it != ed; ++it) {
            static boost::regex threshold("Threshold\\((\\d+\\.\\d+)\\)"),
                cone("ConeVeto\\((\\d+\\.\\d+)\\)"),
                angleCone("AngleCone\\((\\d+\\.\\d+)\\)"),
                angleVeto("AngleVeto\\((\\d+\\.\\d+)\\)"),
                number("^(\\d+\\.?|\\d*\\.\\d*)$");
            boost::cmatch match;
            if (regex_match(it->c_str(), match, threshold)) {
                vetos_.push_back(new ThresholdVeto(atof(match[1].first)));
            } else if (regex_match(it->c_str(), match, cone)) {
                vetos_.push_back(new ConeVeto(Direction(), atof(match[1].first)));
            } else if (regex_match(it->c_str(), match, number)) {
                vetos_.push_back(new ConeVeto(Direction(), atof(match[1].first)));
            } else if (regex_match(it->c_str(), match, angleCone)) {
                vetos_.push_back(new AngleCone(Direction(), atof(match[1].first)));
            } else if (regex_match(it->c_str(), match, angleVeto)) {
                vetos_.push_back(new AngleConeVeto(Direction(), atof(match[1].first)));
            } else {
                throw cms::Exception("Not Implemented") << "Veto " << it->c_str() << " not implemented yet...";
            }
        }
    }

}

IsoDepositIsolator::~IsoDepositIsolator() {
    for (AbsVetos::iterator it = vetos_.begin(), ed = vetos_.end(); it != ed; ++it) {
        delete *it;
    }
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

float
IsoDepositIsolator::getValue(const edm::ProductID &id, size_t index) const {
    const reco::IsoDeposit &dep = handle_->get(id, index);

    double eta = dep.eta(), phi = dep.phi(); // better to center on the deposit direction that could be, e.g., the impact point at calo
    for (AbsVetos::const_iterator it = vetos_.begin(), ed = vetos_.end(); it != ed; ++it) {
        (const_cast<AbsVeto *>(*it))->centerOn(eta, phi); // I need the const_cast to be able to 'move' the veto
    }
    switch (mode_) {
        case Sum:         return dep.depositWithin(deltaR_, vetos_, skipDefaultVeto_);
        case SumRelative: return dep.depositWithin(deltaR_, vetos_, skipDefaultVeto_) / dep.candEnergy() ;
        case Count:       return dep.depositAndCountWithin(deltaR_, vetos_, skipDefaultVeto_).second ;
    }
    throw cms::Exception("Logic error") << "Should not happen at " << __FILE__ << ", line " << __LINE__; // avoid gcc warning
}
