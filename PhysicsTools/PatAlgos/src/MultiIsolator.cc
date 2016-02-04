#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/SimpleIsolator.h"
#include "PhysicsTools/PatAlgos/interface/IsoDepositIsolator.h"
#include "DataFormats/PatCandidates/interface/Flags.h"
#include <sstream>

using namespace pat::helper;

MultiIsolator::MultiIsolator(const edm::ParameterSet &conf, bool cuts) {
    using pat::Flags;
    if (conf.exists("tracker")) addIsolator(conf.getParameter<edm::ParameterSet>("tracker"), cuts, Flags::Isolation::Tracker, pat::TrackIso);
    if (conf.exists("ecal"))    addIsolator(conf.getParameter<edm::ParameterSet>("ecal"),    cuts, Flags::Isolation::ECal, pat::EcalIso);
    if (conf.exists("hcal"))    addIsolator(conf.getParameter<edm::ParameterSet>("hcal"),    cuts, Flags::Isolation::HCal, pat::HcalIso);
    if (conf.exists("calo"))    addIsolator(conf.getParameter<edm::ParameterSet>("calo"),    cuts, Flags::Isolation::Calo, pat::CaloIso);
    if (conf.exists("calo") && (conf.exists("ecal") || conf.exists("hcal"))) {
        throw cms::Exception("Configuration") << 
            "MultiIsolator: you can't specify both 'calo' isolation and 'ecal'/'hcal', " <<
            "as the 'calo' isolation flag is just the logical OR of 'ecal' and 'hcal'.\n";                            
    }
    if (conf.exists("pfAllParticles"))  addIsolator(conf.getParameter<edm::ParameterSet>("pfAllParticles"), cuts,Flags::Isolation::Calo, pat::PfAllParticleIso); 	 
    if (conf.exists("pfChargedHadron")) addIsolator(conf.getParameter<edm::ParameterSet>("pfChargedHadron"), cuts,Flags::Isolation::Calo, pat::PfChargedHadronIso); 	 
    if (conf.exists("pfNeutralHadron")) addIsolator(conf.getParameter<edm::ParameterSet>("pfNeutralHadron"), cuts,Flags::Isolation::Calo, pat::PfNeutralHadronIso); 	 
    if (conf.exists("pfGamma"))         addIsolator(conf.getParameter<edm::ParameterSet>("pfGamma"), cuts,Flags::Isolation::Calo, pat::PfGammaIso);
    if (conf.exists("user")) {
   
        std::vector<edm::ParameterSet> psets = conf.getParameter<std::vector<edm::ParameterSet> >("user");
        if (psets.size() > 5) {
            throw cms::Exception("Configuration") << 
                "MultiIsolator: you can specify at most 5 user isolation collections.\n";            
        }
        uint32_t bit = Flags::Isolation::User1;
        for (std::vector<edm::ParameterSet>::const_iterator it = psets.begin(), ed = psets.end(); it != ed; ++it, bit <<= 1) {
            addIsolator(*it, cuts, bit, pat::IsolationKeys(pat::UserBaseIso + (it - psets.begin())));
        }
    }
}


void 
MultiIsolator::addIsolator(BaseIsolator *iso, uint32_t mask, pat::IsolationKeys key) {
    isolators_.push_back(iso);
    masks_.push_back(mask); 
    keys_.push_back(key);
}

BaseIsolator * 
MultiIsolator::make(const edm::ParameterSet &conf, bool withCut) {
    if (conf.empty()) return 0;
    if (conf.exists("placeholder") && conf.getParameter<bool>("placeholder")) return 0;
    if (conf.exists("deltaR")) {
        return new IsoDepositIsolator(conf, withCut);
    } else {
        return new SimpleIsolator(conf, withCut);
    }
}


void 
MultiIsolator::addIsolator(const edm::ParameterSet &conf, bool withCut, uint32_t mask, pat::IsolationKeys key) {
   BaseIsolator * iso = make(conf, withCut);
    if (iso) addIsolator(iso, mask, key);
}


void
MultiIsolator::beginEvent(const edm::Event &event, const edm::EventSetup &eventSetup) {
    for (boost::ptr_vector<BaseIsolator>::iterator it = isolators_.begin(), ed = isolators_.end(); it != ed; ++it) {
        it->beginEvent(event, eventSetup);
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

