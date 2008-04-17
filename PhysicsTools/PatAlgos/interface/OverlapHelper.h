#ifndef PhysicsTools_PatAlgos_OverlapHelper_h
#define PhysicsTools_PatAlgos_OverlapHelper_h
/**
    \class pat::helper::OverlapHelper "PhysicsTools/PatAlgos/interface/OverlapHelper.h"
    \brief Helper class for removing overlaps. 
    Can only check overlaps between already existing* collections.
    Full framework only. 
*/

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/Utilities/interface/StringCutObjectSelector.h"

//#include "PhysicsTools/PatUtils/interface/SimpleOverlapFinder.h"
#include "PhysicsTools/PatUtils/interface/GenericOverlapFinder.h"

#include "PhysicsTools/PatUtils/interface/PatSelectorByFlags.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include <boost/shared_ptr.hpp>

namespace pat { 
    namespace helper {

    struct OverlapWithPointerByDeltaR {
        public:
            OverlapWithPointerByDeltaR(double deltaR) :  scale_(1.0/(deltaR*deltaR)) {}
            template<typename T1, typename T2>
            double operator()(const T1 &t1, const T2 &t2) const {
                return deltaR2(t1,*t2) * scale_;
            }
        private:
            double scale_;
    };
class OverlapHelper {
    public:
        typedef pat::GenericOverlapFinder<pat::helper::OverlapWithPointerByDeltaR> SimpleOverlapFinder;

        // == Generic worker class ==
        class Worker {
            public:
                Worker(const edm::ParameterSet &pset) ; 
                template<typename Collection>
                std::auto_ptr<pat::OverlapList> 
                    run( const edm::Event &iEvent, const Collection &items) const ;
            protected:
                edm::InputTag tag_;
                SimpleOverlapFinder finder_;
                boost::shared_ptr<StringCutObjectSelector<reco::Candidate> > cut_;
                boost::shared_ptr<pat::SelectorByFlags > flags_;
                mutable std::vector<const reco::Candidate *> tmp_;
        }; // worker
 
        /// A vector of the results of overlap testing for element in the collection to isolate, encoded in bits. 
        /// 0 = no overlaps.
        /// Each bit is set to 1 if there is an overlap with the corresponding collection.
        typedef std::vector<uint32_t> Result;

        OverlapHelper() { }
        OverlapHelper(const std::vector<edm::ParameterSet> &psets) ;
        OverlapHelper(const edm::ParameterSet &pset) ;

        void addWorker(const edm::ParameterSet &pset, uint32_t mask=0) ;
        void addWorker(Worker w, uint32_t mask=0) ;

        /// True if it has a non null configuration
        bool enabled() const  { return !workers_.empty(); }

        /// A vector of the results of overlap testing for element in "in", encoded in bits. 0 = no overlaps.
        /// Each bit is set to 1 if there is an overlap with the corresponding collection
        template<typename Collection>
        std::auto_ptr<Result> test( const edm::Event &iEvent, const Collection &items ) const ;

   protected:
        std::vector<Worker>   workers_;
        std::vector<uint32_t> masks_;
} ; // class


} } // namespaces

template<typename Collection>
std::auto_ptr<pat::OverlapList>
pat::helper::OverlapHelper::Worker::run(const edm::Event &iEvent, const Collection &items) const {
        using namespace edm;
        Handle< View<reco::Candidate> > handle;
        iEvent.getByLabel(tag_, handle);
        tmp_.clear(); // just in case
        for (View<reco::Candidate>::const_iterator it = handle->begin(), ed = handle->end(); it != ed; ++it) {
            if (cut_.get()   && ! (*cut_  )(*it) ) continue;
            if (flags_.get() && ! (*flags_)(*it) ) continue;
            tmp_.push_back( & *it );
        }
        std::auto_ptr<pat::OverlapList> ret = finder_.find(items, tmp_);
        tmp_.clear(); 
        return ret;
}

template<typename Collection>
std::auto_ptr<pat::helper::OverlapHelper::Result>
pat::helper::OverlapHelper::test( const edm::Event &iEvent, const Collection &in  ) const {
    using namespace std;

    size_t size = in.size();
    auto_ptr<Result> ret(new Result(size, 0));
    Result &result = *ret; // pointers to vectors are more ugly work with than refs to vectors

    vector<uint32_t>::const_iterator itm = masks_.begin();
    for (vector<Worker>::const_iterator itw = workers_.begin(), edw = workers_.end(); 
                itw != edw; ++itw, ++itm) {
        auto_ptr<pat::OverlapList> overlaps = itw->run(iEvent, in);
        for (pat::OverlapList::const_iterator ito = overlaps->begin(), edo = overlaps->end(); 
                    ito != edo; ++ito) {
            result[ito->first] |= *itm;
        }
    }

    return ret;
}

#endif

