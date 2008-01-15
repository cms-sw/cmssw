#ifndef PhysicsTools_PatAlgos_CleanerHelper_h
#define PhysicsTools_PatAlgos_CleanerHelper_h
/**
    \class CleanerHelper "PhysicsTools/PatAlgos/interface/CleanerHelper.h"
    \brief Helper class for cleaning with Pt sorting and backreferences
*/

#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

namespace pat { namespace helper {

template<typename T, typename T2=T, typename Collection=std::vector<T2> >
class CleanerHelper {
    public:
        CleanerHelper(const edm::InputTag &src, const std::string &instanceName="") ;
        ~CleanerHelper() { }

        /// read data from the event and setup internal containers
        void newEvent(edm::Event &iEvent);

        /// put data in the event
        void done();

        /// a reference to a view of the the source collection
        const edm::View<T> & view() const { return *sourceView_; }

        /// size of the source collection
        size_t size() const { return sourceView_->size(); }

        /// a reference to the view 
        const T & operator[](size_t i) const { return (*sourceView_)[i]; }

        /// says that item idx is selected and possibly specifies a new value for it
        void addItem(size_t idx, const T2 &value) ;
        //void addItem(size_t idx, 
        //        const T2 &value = dynamic_cast<const T2 &>((*sourceView_)[idx])) ;

        /// Pt sort
        typedef size_t first_argument_type;
        typedef size_t second_argument_type;
        bool operator()( const size_t & t1, const size_t & t2 ) const {
            return (*sourceView_)[t1].pt() > (*sourceView_)[t2].pt();
        }

    private:
        // ---- member functions ----
        void cleanup() ;
        // ---- datamembers ----
        // fixed
        edm::InputTag src_;
        std::string label_;
        // new at every event
        edm::Event *event_;
        edm::Handle< edm::View<T> > sourceView_;
        edm::RefProd<Collection> outRefProd_;
        std::vector<reco::CandidateBaseRef> originalRefs_;
        Collection selected_;
        // temporary but we keep them to avoid re allocating  memory
        std::vector<size_t> indices_;
        std::vector<reco::CandidateBaseRef> sortedRefs_; 
}; // class


template<typename T, typename T2, typename Collection>
CleanerHelper<T,T2,Collection>::CleanerHelper(const edm::InputTag &src, const std::string &instanceName) :
    src_(src),
    label_(instanceName) 
{ 
}


template<typename T, typename T2, typename Collection>
void CleanerHelper<T,T2,Collection>::newEvent(edm::Event &iEvent) 
{
    cleanup(); // just in case

    event_ = & iEvent;

    edm::Handle< edm::View<T> > sourceView_;
    event_->getByLabel(src_, sourceView_);

    outRefProd_ = event_->getRefBeforePut<Collection>(label_);
}

template<typename T, typename T2, typename Collection>
void CleanerHelper<T,T2,Collection>::addItem(size_t idx, const T2 &value) 
{
    edm::RefToBase<T> backRef(sourceView_, idx);
    selected_.push_back(value);
}

template<typename T, typename T2, typename Collection>
void CleanerHelper<T,T2,Collection>::done() {
    if (event_ == 0) throw cms::Exception("CleanerHelper") << 
        "You're calling done() without calling newEvent() before";

    std::auto_ptr<reco::CandRefValueMap> backRefs(new reco::CandRefValueMap());
    reco::CandRefValueMap::Filler backRefFiller(*backRefs);

    // step 1: make list of indices
    size_t nselected = selected_.size();
    indices_.resize(nselected);
    for (size_t i = 0; i < nselected; ++i) indices_[i] = i;

    // step 2: sort the list of indices. I am the comparator
    std::sort(indices_.begin(), indices_.end(), *this);

    // step 3: use sorted indices
    std::auto_ptr<Collection> sorted(new Collection(nselected));
    sortedRefs_.resize(nselected);
    for (size_t i = 0; i < nselected; ++i) {
        (*sorted)[i]     = selected_[indices_[i]];
        sortedRefs_[i]   = originalRefs_[indices_[i]];
    }

    // fill in backrefs
    backRefFiller.insert(outRefProd_, sortedRefs_.begin(), sortedRefs_.end());
    backRefFiller.fill();

    // put objects in Event
    event_->put(sorted);
    event_->put(backRefs);

    cleanup();
}

template<typename T, typename T2, typename Collection>
void CleanerHelper<T,T2,Collection>::cleanup() {
    selected_.clear();
    originalRefs_.clear();
    sortedRefs_.clear();

    event_ = 0;
}



} } // namespaces

#endif
