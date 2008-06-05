#include "PhysicsTools/PatAlgos/interface/IsoDepositIsolator.h"
#include <sstream>

#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"
#include <boost/regex.hpp>

using pat::helper::IsoDepositIsolator;
using pat::helper::BaseIsolator;
using namespace reco::isodeposit;

class SwitchingEcalVeto : public AbsVeto {
    public:
        // creates SwitchingEcalVeto from another AbsVeto (which becomes owned by this veto) 
        SwitchingEcalVeto(AbsVeto *veto, bool isBarrel) :
            veto_(veto), barrel_(isBarrel) {}
        virtual bool veto(double eta, double phi, float value) const {
            return on_ ? veto_->veto(eta,phi,value) : false;
        }
        virtual void centerOn(double eta, double phi) {
            if ( (fabs(eta) < 1.479) == (barrel_) ) {
                on_ = true;
                veto_->centerOn(eta,phi);
            } else {
                on_ = false;
            }
        }
    private:
        std::auto_ptr<AbsVeto> veto_;
        bool barrel_, on_;   
};

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
              vetos_.push_back( makeVeto( it->c_str() ) );
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

AbsVeto *
IsoDepositIsolator::makeVeto(const char *string) const {
    static boost::regex 
        ecalSwitch("^Ecal(Barrel|Endcaps):(.*)"),
        threshold("Threshold\\((\\d+\\.\\d+)\\)"),
        cone("ConeVeto\\((\\d+\\.\\d+)\\)"),
        angleCone("AngleCone\\((\\d+\\.\\d+)\\)"),
        angleVeto("AngleVeto\\((\\d+\\.\\d+)\\)"),
        rectangularEtaPhiVeto("RectangularEtaPhiVeto\\(([+-]?\\d+\\.\\d+),([+-]?\\d+\\.\\d+),([+-]?\\d+\\.\\d+),([+-]?\\d+\\.\\d+)\\)"),
        number("^(\\d+\\.?|\\d*\\.\\d*)$");
    boost::cmatch match;
    
    if (regex_match(string, match, ecalSwitch)) {
        return new SwitchingEcalVeto(makeVeto(match[2].first), (match[1].first == "Barrel") );
    } else if (regex_match(string, match, threshold)) {
        return new ThresholdVeto(atof(match[1].first));
    } else if (regex_match(string, match, cone)) {
        return new ConeVeto(Direction(), atof(match[1].first));
    } else if (regex_match(string, match, number)) {
        return new ConeVeto(Direction(), atof(match[1].first));
    } else if (regex_match(string, match, angleCone)) {
        return new AngleCone(Direction(), atof(match[1].first));
    } else if (regex_match(string, match, angleVeto)) {
        return new AngleConeVeto(Direction(), atof(match[1].first));
    } else if (regex_match(string, match, rectangularEtaPhiVeto)) {
        return new RectangularEtaPhiVeto(reco::isodeposit::Direction(), 
                    atof(match[1].first), atof(match[2].first), 
                    atof(match[3].first), atof(match[4].first));
    } else {
        throw cms::Exception("Not Implemented") << "Veto " << string << " not implemented yet...";
    }
}
