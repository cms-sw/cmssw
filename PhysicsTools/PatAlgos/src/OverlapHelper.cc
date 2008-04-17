#include "PhysicsTools/PatAlgos/interface/OverlapHelper.h"
#include "FWCore/MessageService/interface/MessageLogger.h"
#include "DataFormats/PatCandidates/interface/Flags.h"

using pat::helper::OverlapHelper;

pat::helper::OverlapHelper::Worker::Worker(const edm::ParameterSet &pset) : 
    tag_(pset.getParameter<edm::InputTag>("collection")),
    finder_(pset.getParameter<double>("deltaR")),
    cut_()
{
    if (pset.exists("cut")) {
        std::string cut = pset.getParameter<std::string>("cut");
        if (!cut.empty()) {
            typedef StringCutObjectSelector<reco::Candidate> CutObj;
            cut_ = boost::shared_ptr<CutObj>( new CutObj(cut) );
        }
    }
    if (pset.exists("flags")) {
        std::vector<std::string> flags = pset.getParameter<std::vector<std::string> >("flags");
        if (!flags.empty()) {
            flags_ = boost::shared_ptr<pat::SelectorByFlags>( new pat::SelectorByFlags(flags) );
        }
    }
}


void 
pat::helper::OverlapHelper::addWorker(const edm::ParameterSet &pset, uint32_t mask) {
    if (!pset.empty()) addWorker(Worker(pset), mask);
}
void
pat::helper::OverlapHelper::addWorker(const OverlapHelper::Worker w, uint32_t mask) 
{
    if (mask == 0) mask = ((1 << workers_.size()) << pat::Flags::Overlap::Shift);
    masks_.push_back(mask);
    workers_.push_back(w);
}


pat::helper::OverlapHelper::OverlapHelper(const std::vector<edm::ParameterSet> &psets) 
{
    typedef std::vector<edm::ParameterSet>::const_iterator VPI;
    workers_.reserve(psets.size());
    for (VPI it = psets.begin(), ed = psets.end(); it != ed; ++it) {
        addWorker(Worker(*it));
    }
}

pat::helper::OverlapHelper::OverlapHelper(const edm::ParameterSet &pset) 
{
    using pat::Flags;
    if (pset.exists("jets"))      addWorker(pset.getParameter<edm::ParameterSet>("jets"),      Flags::Overlap::Jets);
    if (pset.exists("electrons")) addWorker(pset.getParameter<edm::ParameterSet>("electrons"), Flags::Overlap::Electrons);
    if (pset.exists("muons"))     addWorker(pset.getParameter<edm::ParameterSet>("muons"),     Flags::Overlap::Muons);
    if (pset.exists("taus"))      addWorker(pset.getParameter<edm::ParameterSet>("taus"),      Flags::Overlap::Taus);
    if (pset.exists("photons"))   addWorker(pset.getParameter<edm::ParameterSet>("photons"),   Flags::Overlap::Photons);
    if (pset.exists("user")) {
        std::vector<edm::ParameterSet> psets = pset.getParameter<std::vector<edm::ParameterSet> >("user");
        typedef std::vector<edm::ParameterSet>::const_iterator VPI;
        if (psets.size() > 0) addWorker(psets[0], Flags::Overlap::User1);
        if (psets.size() > 1) addWorker(psets[1], Flags::Overlap::User2);
        if (psets.size() > 2) addWorker(psets[2], Flags::Overlap::User3);
        if (psets.size() > 3) {
            throw cms::Exception("Configuration") << 
                                 "OverlapHelper: " <<
                                 "Only up to 3 additional overlap collections can be specified.\n" << 
                                 "If you need more, usurp one of the standard ones.\n";
        }
    }
}
