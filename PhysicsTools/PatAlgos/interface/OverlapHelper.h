#ifndef PhysicsTools_PatAlgos_OverlapHelper_h
#define PhysicsTools_PatAlgos_OverlapHelper_h
/**
    \class OverlapHelper "PhysicsTools/PatAlgos/interface/OverlapHelper.h"
    \brief Helper class for removing overlaps.
     Helper class for removing overlaps: takes into account reading parameters from cfg & collections from event
*/

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "PhysicsTools/PatUtils/interface/SimpleOverlapFinder.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include <memory>

namespace pat { namespace helper {

class OverlapHelper {
    public:
        /// A vector whose i-th element is: -1 if everything was ok; n>=0 if there was an overlap with the n-th collection (starting from zero)
        typedef std::vector<uint32_t> Result;

        OverlapHelper() { }
        OverlapHelper(const std::vector<edm::ParameterSet> &psets) ;

        /// True if it has a non null configuration
        bool enabled() const  { return !tests_.empty(); }

        /// A vector of the results of overlap testing for element in "in", encoded in bits. 0 = no overlaps.
        /// Each bit is set to 1 if there is an overlap with the corresponding collection
        /// e.g. if this is configured to isolate against electrons and taus, 1 = overlaps with electron, 2 = with tau, 3 = with both, 0 = safe.
        std::auto_ptr<Result> test( const edm::Event &iEvent, const std::vector<const reco::Candidate *> &in  ) const ;

        /// same as the other test() method, but with templaed input for convenience
        template <typename Iterator>
        std::auto_ptr<Result> test( const edm::Event &iEvent, const Iterator &begin, const Iterator &end  ) const ;
    protected:
        struct Test {
            edm::InputTag src;
            pat::SimpleOverlapFinder finder;
            Test(edm::InputTag aTag, pat::SimpleOverlapFinder aFinder) : src(aTag), finder(aFinder) {}
        } ; // struct Test
        std::vector<Test> tests_;

} ; // class

template <typename Iterator>
std::auto_ptr<OverlapHelper::Result>
OverlapHelper::test(const edm::Event &iEvent, const Iterator &begin, const Iterator &end) const {
    std::vector<const reco::Candidate *> tmp;
    for (Iterator it = begin; it != end; ++it) {
        tmp.push_back(&*it);
    }   
    return test(iEvent, tmp);
}

} } // namespaces

#endif

