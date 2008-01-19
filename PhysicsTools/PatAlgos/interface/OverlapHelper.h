#ifndef PhysicsTools_PatAlgos_OverlapHelper_h
#define PhysicsTools_PatAlgos_OverlapHelper_h
/**
    \class OverlapHelper "PhysicsTools/PatAlgos/interface/OverlapHelper.h"
    \brief Helper class for removing overlaps. 
    Can only check overlaps between already existing* collections.
    Full framework only. 
*/

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

//#include "PhysicsTools/PatUtils/interface/SimpleOverlapFinder.h"
#include "PhysicsTools/PatUtils/interface/GenericOverlapFinder.h"


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"


namespace pat { 
    namespace helper {


class OverlapHelper {
    public:
        typedef pat::GenericOverlapFinder<pat::OverlapByDeltaR> SimpleOverlapFinder;

        /// A vector of the results of overlap testing for element in the collection to isolate, encoded in bits. 
        /// 0 = no overlaps.
        /// Each bit is set to 1 if there is an overlap with the corresponding collection.
        typedef std::vector<uint32_t> Result;

        OverlapHelper() { }
        OverlapHelper(const std::vector<edm::ParameterSet> &psets) ;

        /// True if it has a non null configuration
        bool enabled() const  { return !workers_.empty(); }

        /// A vector of the results of overlap testing for element in "in", encoded in bits. 0 = no overlaps.
        /// Each bit is set to 1 if there is an overlap with the corresponding collection
        template<typename Collection>
        std::auto_ptr<Result> test( const edm::Event &iEvent, const Collection &items ) const ;

        // == Generic worker class ==
        class Worker {
            public:
                Worker(const edm::ParameterSet &pset) : 
                    tag_(pset.getParameter<edm::InputTag>("collection")),
                    finder_(pset.getParameter<double>("deltaR")) { }
                template<typename Collection>
                std::auto_ptr<pat::OverlapList> 
                    run( const edm::Event &iEvent, const Collection &items) const ;
            protected:
                edm::InputTag tag_;
                SimpleOverlapFinder finder_;
        }; // worker
    protected:
        std::vector<Worker> workers_;
} ; // class


} } // namespaces

template<typename Collection>
std::auto_ptr<pat::OverlapList>
pat::helper::OverlapHelper::Worker::run(const edm::Event &iEvent, const Collection &items) const {
        using namespace edm;
        Handle< View<reco::Candidate> > handle;
        iEvent.getByLabel(tag_, handle);
        return finder_.find(items, *handle);
}

template<typename Collection>
std::auto_ptr<pat::helper::OverlapHelper::Result>
pat::helper::OverlapHelper::test( const edm::Event &iEvent, const Collection &in  ) const {
    using namespace std;

    size_t size = in.size();
    auto_ptr<Result> ret(new Result(size, 0));
    Result &result = *ret; // pointers to vectors are more ugly work with than refs to vectors

    int testBit = 1;
    for (vector<Worker>::const_iterator itw = workers_.begin(), edw = workers_.end(); 
                itw != edw; ++itw, testBit <<= 1) {
        auto_ptr<pat::OverlapList> overlaps = itw->run(iEvent, in);
        for (pat::OverlapList::const_iterator ito = overlaps->begin(), edo = overlaps->end(); 
                    ito != edo; ++ito) {
            result[ito->first] += testBit;
        }
    }

    return ret;
}

#endif

