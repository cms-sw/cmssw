#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/SimpleIsolator.h"
#include "PhysicsTools/PatAlgos/interface/IsoDepositIsolator.h"
#include "DataFormats/PatCandidates/interface/Flags.h"
#include <sstream>

using namespace pat::helper;


MultiIsolator::MultiIsolator(const edm::ParameterSet &conf) {
    using pat::Flags;
    if (conf.exists("tracker")) addIsolator(conf.getParameter<edm::ParameterSet>("tracker"), Flags::Isolation::Tracker);
    if (conf.exists("ecal"))    addIsolator(conf.getParameter<edm::ParameterSet>("ecal"),    Flags::Isolation::ECal);
    if (conf.exists("hcal"))    addIsolator(conf.getParameter<edm::ParameterSet>("hcal"),    Flags::Isolation::HCal);
    if (conf.exists("calo"))    addIsolator(conf.getParameter<edm::ParameterSet>("calo"),    Flags::Isolation::Calo);
    if (conf.exists("calo") && (conf.exists("ecal") || conf.exists("hcal"))) {
        throw cms::Exception("Configuration") << 
            "MultiIsolator: you can't specify both 'calo' isolation and 'ecal'/'hcal', " <<
            "as the 'calo' isolation flag is just the logical OR of 'ecal' and 'hcal'.\n";                            
    }
    if (conf.exists("user")) {
        std::vector<edm::ParameterSet> psets = conf.getParameter<std::vector<edm::ParameterSet> >("user");
        if (psets.size() > 5) {
            throw cms::Exception("Configuration") << 
                "MultiIsolator: you can specify at most 5 user isolation collections.\n";            
        }
        uint32_t bit = Flags::Isolation::User1;
        for (std::vector<edm::ParameterSet>::const_iterator it = psets.begin(), ed = psets.end(); it != ed; ++it, bit <<= 1) {
            addIsolator(*it, bit);
        }
    }
}


void 
MultiIsolator::addIsolator(BaseIsolator *iso, uint32_t mask) {
    isolators_.push_back(iso);
    masks_.push_back(mask); 
}

BaseIsolator * 
MultiIsolator::make(const edm::ParameterSet &conf) {
    if (conf.empty()) return 0;
    if (conf.exists("placeholder") && conf.getParameter<bool>("placeholder")) return 0;
    if (conf.exists("deltaR")) {
        return new IsoDepositIsolator(conf);
    } else {
        return new SimpleIsolator(conf);
    }
}


void 
MultiIsolator::addIsolator(const edm::ParameterSet &conf, uint32_t mask) {
    BaseIsolator * iso = make(conf);
    if (iso) addIsolator(iso, mask);
}


void
MultiIsolator::beginEvent(const edm::Event &event) {
    for (boost::ptr_vector<BaseIsolator>::iterator it = isolators_.begin(), ed = isolators_.end(); it != ed; ++it) {
        it->beginEvent(event);
    }
}

void
MultiIsolator::endEvent() {
    for (boost::ptr_vector<BaseIsolator>::iterator it = isolators_.begin(), ed = isolators_.end(); it != ed; ++it) {
        it->endEvent();
    }
}

void
MultiIsolator::print(std::ostream &out) const {
    for (boost::ptr_vector<BaseIsolator>::const_iterator it = isolators_.begin(), ed = isolators_.end(); it != ed; ++it) {
        out << " * ";
        it->print(out); 
        out << ": Flag " << pat::Flags::bitToString( masks_[it - isolators_.begin()] ) << "\n";
    }    
    out << "\n";
}

std::string 
MultiIsolator::printSummary() const {
    std::ostringstream isoSumm;
    print(isoSumm);
    return isoSumm.str();
}

