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

template<typename T>
struct NullSorter {
        bool operator()( const T & t1, const T & t2 ) const {
            return (&t1) < (&t2);
        }
}; 

template<typename T, typename T2=T, typename Collection = std::vector<T2>, typename Comparator = NullSorter<T2> >
class CleanerHelper {
    public:
        CleanerHelper() { } // needed for EDM Modules
        CleanerHelper(const edm::InputTag &src, const std::string &instanceName="") ;
        ~CleanerHelper() { }

        /// read data from the event and setup internal containers
        void newEvent(edm::Event &iEvent);

        /// Put data in the event and clear transient containers.
        /// Only items with mark == 0 (which is the default) will be saved.
        void done();

        /// a reference to a view of the the source collection
        const edm::View<T> & source() const { return *sourceView_; }

        /// size of the source collection
        size_t srcSize() const { return sourceView_->size(); }

        /// a reference to an item in the source collection
        const T & srcAt(size_t i) const { return (*sourceView_)[i]; }
        /// a reference to an item in the source collection
        edm::RefToBase<T> srcRefAt(size_t i) const { return sourceView_->refAt(i); }

        /// A const reference to the selected items. 
        /// Must be const, to avoid people doing push_back or remove by hand and breaking the refs.
        const Collection & selected() const { return selected_; }

        /// Size of the collection of selected items
        size_t size() const { return selected_.size(); }
        /// A reference to the i-th selected item (const)
        const T2 & operator[](size_t i) const { return selected_[i]; }
        /// A reference to the i-th selected item. Non const, you are allowed to modify the item.
        T2 & operator[](size_t i) { return selected_[i]; }

        /// Mark an item (given its index in the selected collection).
        /// At the end only items with mark == 0 will be saved
        void setMark(size_t selIdx, uint32_t mark) { marks_[selIdx] = mark; }
        /// Get the mark of an item (given its index in the selected collection)
        /// At the end only items with mark == 0 will be saved
        uint32_t mark(size_t selIdx) { return marks_[selIdx]; }

        /// Inform the helper that item with index sourceIdx in the source collection is selected,
        /// and it's value in the new collection is "value" (that can just be original object if you only select)
        /// It also allows to set a transient marks on the items (in the end only items with mark == 0 will be saved)
        /// Returns the index in the collection of selected items.
        size_t addItem(size_t sourceIdx, const T2 &value, uint32_t mark=0) ;

        /// Pt sort
        typedef size_t first_argument_type;
        typedef size_t second_argument_type;
        bool operator()( const size_t & t1, const size_t & t2 ) const {
           return comp_(selected_[t1], selected_[t2]);
        }

    private:
        // ---- member functions ----
        void cleanup() ;
        // ---- datamembers ----
        // fixed
        edm::InputTag src_;
        std::string label_;
        Comparator comp_;
        // new at every event
        edm::Event *event_;
        edm::Handle< edm::View<T> > sourceView_;
        std::vector<reco::CandidateBaseRef> originalRefs_;
        Collection selected_;
        std::vector<uint32_t> marks_;
        // temporary but we keep them to avoid re allocating  memory
        std::vector<size_t> indices_;
        std::vector<reco::CandidateBaseRef> sortedRefs_; 
}; // class


template<typename T, typename T2, typename Collection, typename Comparator>
CleanerHelper<T,T2,Collection,Comparator>::CleanerHelper(const edm::InputTag &src, const std::string &instanceName) :
    src_(src),
    label_(instanceName) 
{ 
}


template<typename T, typename T2, typename Collection, typename Comparator>
void CleanerHelper<T,T2,Collection,Comparator>::newEvent(edm::Event &iEvent) 
{
    cleanup(); // just in case

    event_ = & iEvent;

    event_->getByLabel(src_, sourceView_);
}

template<typename T, typename T2, typename Collection, typename Comparator>
size_t CleanerHelper<T,T2,Collection,Comparator>::addItem(size_t idx, const T2 &value, const uint32_t mark) 
{
    selected_.push_back(value);
    marks_.push_back(mark);

    edm::RefToBase<T> backRef(sourceView_, idx); // That's all I can get from a View

    // === OPTION 1 ===
    //
    // originalRefs_.push_back(reco::CandidateBaseRef(backRef));   // <== DOES NOT WORK
    // 
    //   NOTE: the above apparently creates problems with dictionaries as it look like it requires a dictionary for
    //      edm::reftobase::Holder<Candidate, edm::RefToBase<T> >
    //   which is usually not provided (if T is a final type like CaloJet usually there is not even the dict for RefToBase<T>!)
    //   if the RefToBase constructor from RefToBase gets fixed as in
    //     https://hypernews.cern.ch/HyperNews/CMS/get/edmFramework/1308.html 
    //   then it should work

    // === OPTION 2 ===
    //
    // edm::Ref< std::vector<T> > plainRef = backRef.template castTo< edm::Ref< std::vector<T> > >();
    // originalRefs_.push_back(reco::CandidateBaseRef(plainRef));    
    //  
    //   NOTE:: this works if I'm reading a std::vector<T>, but I don't think it works in the general case

    // === OPTION 3 ===
    boost::shared_ptr<edm::reftobase::RefHolderBase> holderBase(backRef.holder().release());
    originalRefs_.push_back(reco::CandidateBaseRef(holderBase));
    //
    //   NOTE: this should force the conversion into and IndirectHolder,
    //     and avoid dictionary problems even if we can't fix RefToBase directly.
    //
    // May the Source be with you, Luke

    return selected_.size() - 1;
}

template<typename T, typename T2, typename Collection, typename Comparator>
void CleanerHelper<T,T2,Collection,Comparator>::done() {
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
    std::auto_ptr<Collection> sorted(new Collection()); sorted->reserve(nselected);
    sortedRefs_.reserve(nselected);
    for (size_t i = 0; i < nselected; ++i) {
        size_t idx = indices_[i];
        if (marks_[idx] != 0) continue; // skip marked items
        sorted->push_back( selected_[idx] );
        sortedRefs_.push_back( originalRefs_[idx] );
    }

    // FIRST put the collection
    edm::OrphanHandle<Collection> newRefProd = event_->put(sorted);

    // THEN fill the map and put it in the event
    // fill in backrefs
    backRefFiller.insert(newRefProd, sortedRefs_.begin(), sortedRefs_.end());
    backRefFiller.fill();

    // put objects in Event
    event_->put(backRefs);

    cleanup();
}

template<typename T, typename T2, typename Collection, typename Comparator>
void CleanerHelper<T,T2,Collection,Comparator>::cleanup() {
    selected_.clear();
    marks_.clear();
    originalRefs_.clear();
    sortedRefs_.clear();

    event_ = 0;
}



} } // namespaces

#endif
