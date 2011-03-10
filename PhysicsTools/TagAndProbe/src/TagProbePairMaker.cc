#include "PhysicsTools/TagAndProbe/interface/TagProbePairMaker.h"
#include "DataFormats/Candidate/interface/Candidate.h"

tnp::TagProbePairMaker::TagProbePairMaker(const edm::ParameterSet &iConfig) :
    src_(iConfig.getParameter<edm::InputTag>("tagProbePairs"))
{
    std::string arbitration = iConfig.getParameter<std::string>("arbitration");
    if (arbitration == "None") {
        arbitration_ = None;
    } else if (arbitration == "OneProbe") {
        arbitration_ = OneProbe;
    } else if (arbitration == "BestMass") {
        arbitration_ = BestMass;
        arbitrationMass_ = iConfig.getParameter<double>("massForArbitration");
    } else throw cms::Exception("Configuration") << "TagProbePairMakerOnTheFly: the only currently allowed values for 'arbitration' are 'None', 'OneProbe', 'BestMass'\n";
}


tnp::TagProbePairs
tnp::TagProbePairMaker::run(const edm::Event &iEvent) const 
{
    // declare output
    tnp::TagProbePairs pairs;
    
    // read from event
    edm::Handle<reco::CandidateView> src;
    iEvent.getByLabel(src_, src);

    // convert
    for (reco::CandidateView::const_iterator it = src->begin(), ed = src->end(); it != ed; ++it) {
        const reco::Candidate & mother = *it;
        if (mother.numberOfDaughters() != 2) throw cms::Exception("CorruptData") << "Tag&Probe pair with " << mother.numberOfDaughters() << " daughters\n";
        pairs.push_back(tnp::TagProbePair(mother.daughter(0)->masterClone(), mother.daughter(1)->masterClone(), 
                                          src->refAt(it - src->begin()), mother.mass()));
    }

    if ((arbitration_ != None) && (pairs.size() > 1)) {
        // might need to clean up
        arbitrate(pairs);
    }

    // return
    return pairs;
}

void 
tnp::TagProbePairMaker::arbitrate(TagProbePairs &pairs) const 
{
    size_t nclean = pairs.size();
    for (TagProbePairs::iterator it = pairs.begin(), ed = pairs.end(); it != ed; ++it) {
        if (it->tag.isNull()) continue; // skip already invalidated pairs
        bool invalidateThis = false;
        for (TagProbePairs::iterator it2 = it + 1; it2 != ed; ++it2) {   // it+1 <= ed, otherwise we don't arrive here
            if (it->tag == it2->tag) {
                if (arbitration_ == OneProbe) {
                    // invalidate this one
                    it2->tag = reco::CandidateBaseRef(); --nclean;
                    // remember to invalidate also the other one (but after finishing to check for duplicates)
                    invalidateThis = true;
                } else if (arbitration_ == BestMass) {
                    // but the best one in the first  iterator
                    if (fabs(it2->mass-arbitrationMass_) < fabs(it->mass-arbitrationMass_)) {
                        std::swap(*it, *it2);
                    }
                    // and invalidate it2
                    it2->tag = reco::CandidateBaseRef(); --nclean;
                }
            }
        }
        if (invalidateThis) { it->tag = reco::CandidateBaseRef(); --nclean; }
    }

    if (nclean == 0) {
        pairs.clear();
    } else if (nclean < pairs.size()) {
        TagProbePairs cleaned; cleaned.reserve(nclean);
        for (TagProbePairs::iterator it = pairs.begin(), ed = pairs.end(); it != ed; ++it) {
            if (it->tag.isNonnull()) cleaned.push_back(*it);
        }
        pairs.swap(cleaned);
    }
}
