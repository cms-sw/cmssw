#ifndef PhysicsTools_PatAlgos_CleanerHelper_h
#define PhysicsTools_PatAlgos_CleanerHelper_h
/**
    \class pat::helper::CleanerHelper "PhysicsTools/PatAlgos/plugins/CleanerHelper.h"
    \brief Helper class for cleaning with Pt sorting and backreferences
*/

#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/Flags.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/MessageService/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

// let's try to catch some errors at compile time
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace pat { namespace helper {

template<typename T>
struct NullSorter {
        bool operator()( const T & t1, const T & t2 ) const {
            return (&t1) < (&t2);
        }
}; 

template<typename T, typename T2=T, typename Collection = std::vector<T2>, typename Comparator = NullSorter<T2> >
class CleanerHelper {
    // make sure our template arguments T and T2 inherits from Candidate. The extra () is needed!.
    BOOST_STATIC_ASSERT( (boost::is_base_of<reco::Candidate,T>::value) );
    BOOST_STATIC_ASSERT( (boost::is_base_of<reco::Candidate,T2>::value) );
    public:
        typedef CleanerHelper<T,T2,Collection,Comparator> cleaner_type;
        typedef Collection                                collection_type;

        CleanerHelper() { } // needed for EDM Modules
        CleanerHelper(const edm::InputTag &src, const std::string &instanceName="") ;
        ~CleanerHelper() { }

        /// configures the output collections
        void configure(const edm::ParameterSet &conf) ;

        /// calls all the produces<type>(label) methods needed for it's outputs
        void registerProducts(edm::ProducerBase &producer) ;
    
        /// read data from the event and setup internal containers
        void newEvent(edm::Event &iEvent);

        /// Put data in the event and clear transient containers.
        /// Only items with mark == 0 (which is the default) will be saved.
        void done();

        /// To be called optionally at the end of the job, will print summary information
        /// if bookeeping was enabled
        void endJob();


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
        void setMark(size_t selIdx, uint32_t mark) { marks_[selIdx] = mark; if (markItems_) reallyMarkItem(selIdx); }

        /// Turn on additional bit flags on an item. Bits which are already 1 won't be affected.
        /// At the end only items with mark == 0 will be saved
        void addMark(size_t selIdx, uint32_t mark) { marks_[selIdx] |= mark; if (markItems_) reallyMarkItem(selIdx); }

        /// Get the mark of an item (given its index in the selected collection)
        /// At the end only items with mark == 0 will be saved
        uint32_t mark(size_t selIdx) const { return marks_[selIdx]; }

        /// Inform the helper that item with index sourceIdx in the source collection is selected,
        /// and it's value in the new collection is "value" (that can just be original object if you only select)
        /// It also allows to set a transient marks on the items (in the end only items with mark == 0 will be saved)
        /// Returns the index in the collection of selected items.
        size_t addItem(size_t sourceIdx, const T2 &value, uint32_t mark=0) ;

        /// Returns the index of the last selected item, in case you forgot it
        /// Note that if there are no items it will return -1!
        int    lastItemSelIndex() const { return selected_.size() - 1; }

        /// Tell the module also to save "all" items, with a given label,
        /// The collection will contain also the items with mark != 0
        /// If the output type is a subclass of reco::Candidate, and if setStatusMark is true,
        /// the output items will have as status what was set as mark.
        void setSaveAll(const std::string &label="all") {
             saveAll_ = true; labelAll_ = label;
        }
        /// Tell the module also to save the non accepted item, with a given label.
        /// even those for which mark != 0
        void setSaveRejected(const std::string &label="rejected") {
             saveRejected_ = true; labelRejected_ = label;
        }

        /// Sets which bits can be ignored when deciding if the items is good or not.
        /// By default, all bits are used (mask=0), so that if any bit in the mark is 1
        /// the item will not be written to disk.
        /// If a bit is set to "1" in this mask, items will be considered good whatever 
        /// the value of that bit in the status mark
        void setBitsToIgnore (uint32_t mask) { bitsToIgnore_ = mask; }

        /// Mask of bits which are ignored when determining if the item is accepted or not
        uint32_t bitsToIgnore() const { return bitsToIgnore_; }

        /// Tells this helper to call "setStatus(int status)" method on each object
        /// using the 'mark' as status.
        /// This can be done only if the output type T2 is a subclass of Particle
        void setStatusMark(bool setIt=true) ;

        /// Pt sort
        typedef size_t first_argument_type;
        typedef size_t second_argument_type;
        bool operator()( const size_t & t1, const size_t & t2 ) const {
           return comp_(selected_[t1], selected_[t2]);
        }

        class FilteredCollection {
            public:
                typedef T2 value_type;
                typedef typename boost::indirect_iterator<typename Collection::const_iterator> const_iterator;
                explicit FilteredCollection(const cleaner_type &cleaner)  { fill_(cleaner, cleaner.bitsToIgnore()); }
                explicit FilteredCollection(const cleaner_type &cleaner, 
                                            uint32_t bitsToIgnore)        { fill_(cleaner, bitsToIgnore); }
                size_t size() const { return accepted_.size(); }
                const T2 & operator[]     (size_t i) const { return * accepted_[i]; }
                size_t     originalIndexOf(size_t i) const { return oldIndices_[i]; }
                const_iterator begin() const { return const_iterator(accepted_.begin()); }
                const_iterator end()   const { return const_iterator(accepted_.end()  ); }
            private: 
                std::vector<const value_type *> accepted_; 
                std::vector<size_t>             oldIndices_;
                void fill_(const cleaner_type &cleaner, uint32_t mask) {
                    for (size_t i = 0, n = cleaner.size(); i < n; ++i) {
                        if ((cleaner.mark(i) & ~mask) == 0) {
                            accepted_.push_back(& cleaner[i]);
                            oldIndices_.push_back(i);
                        }
                    }
                }
        };
        friend class FilteredCollection;

        /// Returns a "view" of the items which have no mandatory 'bad flags' on
        /// This "view" is static, it won't change if you change bits 
        /// To convert between indices here and indices in all list, use originalIndexOf method.
        FilteredCollection accepted() const { return FilteredCollection(*this); }       
 

        ///Turn on bookkeeping of summary information bit per bit
        void requestSummary() { makeSummary_ = true; } 

        ///Print summary information bit per bit. 
        std::string printSummary() ;
    private:
        // ---- member functions ----
        void cleanup() ;
        void reallyMarkItem(int idx) ;
        // ---- datamembers ----
        // fixed
        edm::InputTag src_;
        std::string label_;
        std::string moduleLabel_;
        Comparator comp_;
        // toggle if I want to save all the data
        bool saveRejected_, saveAll_;
        std::string  labelRejected_, labelAll_;
        // toggle if the output data is marked
        bool markItems_;
        // new at every event
        edm::Event *event_;
        edm::Handle< edm::View<T> > sourceView_;
        std::vector<reco::CandidateBaseRef> originalRefs_;
        Collection selected_;
        std::vector<uint32_t> marks_;
        uint32_t bitsToIgnore_;
        // to print summary information at the end
        bool makeSummary_;
        enum Constants_ { NumberOfBits_ = 32, Tested_ = 0, Failed_ = 1 };
        uint64_t countsTotal_[2], countsForEachBit_[NumberOfBits_][2];
        void addToSummary(uint32_t mask, bool ok);
}; // class


template<typename T, typename T2, typename Collection, typename Comparator>
CleanerHelper<T,T2,Collection,Comparator>::CleanerHelper(const edm::InputTag &src, const std::string &instanceName) :
    src_(src),
    label_(instanceName),moduleLabel_(),
    saveRejected_(false),saveAll_(false),markItems_(false),
    makeSummary_(false)
{ 
    memset( countsForEachBit_, 0, NumberOfBits_ * 2 * sizeof(uint64_t));
    memset( countsTotal_     , 0, 2 * sizeof(uint64_t));
}

template<typename T, typename T2, typename Collection, typename Comparator>
void CleanerHelper<T,T2,Collection,Comparator>::configure(const edm::ParameterSet &conf) {
    moduleLabel_ = conf.getParameter<std::string>("@module_label");
    if ( conf.exists("saveAll") ) {
        std::string saveAll = conf.getParameter<std::string>("saveAll");
        if (!saveAll.empty()) setSaveAll( saveAll );
    }
    if ( conf.exists("saveRejected") ) {
        std::string saveRejected = conf.getParameter<std::string>("saveRejected");
        if (!saveRejected.empty()) setSaveRejected( saveRejected );
    }

    if (conf.exists("markItems")) setStatusMark ( conf.getParameter<bool>("markItems") );

    makeSummary_ = conf.getUntrackedParameter<bool>("wantSummary", false);

    if (conf.exists("bitsToIgnore")) {
        std::string match = "bitsToIgnore";
        std::vector<std::string> names;
        bool found = false; 
        // test for bool
        names = conf.getParameterNamesForType<uint32_t>();
        if (find(names.begin(), names.end(), match) != names.end()) {
            setBitsToIgnore(conf.getParameter<uint32_t>(match));
            found = true;
        } else {
            names = conf.getParameterNamesForType<std::vector<uint32_t> >();
            if (find(names.begin(), names.end(), match) != names.end()) {
                std::vector<uint32_t> rawbits = conf.getParameter<std::vector<uint32_t> >(match);
                uint32_t mask = 0;
                for (std::vector<uint32_t>::const_iterator it = rawbits.begin(); it != rawbits.end(); ++it)  mask |= (1 << *it);
                setBitsToIgnore(mask);
                found = true;
            } else {
                names = conf.getParameterNamesForType<std::vector<std::string> >();
                if (find(names.begin(), names.end(), match) != names.end()) {
                    uint32_t mask = pat::Flags::get( conf.getParameter<std::vector<std::string> >(match) );
                    setBitsToIgnore(mask);
                    found = true;
                }
            }
        } 
        if (!found) {
            throw cms::Exception("Configuration") << 
                "CleanerHelper: parameter 'bitsToIgnore' must be any one of the following:\n" <<
                " - an uint32 mask, with bits to ignore set to 1\n" <<
                " - an vuint32 list of the numbers of bits to set to 1 (bits numbers start from 0)\n" <<
                " - an vstring list of the names of the bits or bit sets to ignore (as defined in pat::Flags)\n";
        }
    }

    if ((saveAll_ || saveRejected_) && !markItems_) {
        typename edm::LogWarning("CleanerHelper") << "You have set 'saveAll' and/or 'saveRejected' but you're not marking items. " <<
                                                     "This is a bit inconsistent, but for the moment is allowed.";
    }
}

template<typename T, typename T2, typename Collection, typename Comparator>
void CleanerHelper<T,T2,Collection,Comparator>::registerProducts(edm::ProducerBase &producer) {
    producer.produces<Collection>(label_).setBranchAlias(moduleLabel_+label_);
    producer.produces<reco::CandRefValueMap>(label_);
    if (saveRejected_) {
        producer.produces<Collection>(labelRejected_).setBranchAlias(moduleLabel_+labelRejected_);
        producer.produces<reco::CandRefValueMap>(labelRejected_); 
    }
    if (saveAll_) {
        producer.produces<Collection>(labelAll_).setBranchAlias(moduleLabel_+labelAll_);
        producer.produces<reco::CandRefValueMap>(labelAll_); 
    }
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
void CleanerHelper<T,T2,Collection,Comparator>::reallyMarkItem(int idx) {
    // vvvv DOES NOT WORK
    //   struct NotMarker { static void mark(T2 &t, int mark) { assert(false);     } };
    //   struct Marker    { static void mark(T2 &t, int mark) { t.setStatus(mark); } };
    //   typedef typename boost::mpl::if_c<typename boost::mpl::is_base_of<typename reco::Particle,T2>, Marker, NotMarker>::type MaybeMarker;
    //   MaybeMarker::mark(selected_[idx], marks_[idx]);
    // ^^^^^
    // vvvvv try again with templates... but it still does not work
    //                                   I should put it in global scope
    //   using reco::Particle;
    //   template<typename X> struct Marker           { static void mark(X &t, int mark)        { assert(false);     } };
    //   template<>           struct Marker<Particle> { static void mark(Particle &t, int mark) { t.setStatus(mark); } };
    //   Marker<T2>::mark(selected_[idx], marks_[idx]);
    // ^^^^
    // so we just assume that T2 inherits from Particle, period.
    selected_[idx].setStatus(marks_[idx]);
}

template<typename T, typename T2, typename Collection, typename Comparator>
void CleanerHelper<T,T2,Collection,Comparator>::done() {
    if (event_ == 0) throw cms::Exception("CleanerHelper") << 
        "You're calling done() without calling newEvent() before";

    std::auto_ptr<reco::CandRefValueMap> backRefs(new reco::CandRefValueMap());
    std::auto_ptr<reco::CandRefValueMap> backRefsRejected, backRefsAll;
    if (saveRejected_) backRefsRejected = std::auto_ptr<reco::CandRefValueMap>(new reco::CandRefValueMap());
    if (saveAll_)      backRefsAll      = std::auto_ptr<reco::CandRefValueMap>(new reco::CandRefValueMap());

    // step 1: make list of indices
    size_t nselected = selected_.size();
    std::vector<size_t> indices(nselected);
    for (size_t i = 0; i < nselected; ++i) indices[i] = i;

    // step 2: sort the list of indices. I am the comparator
    std::sort(indices.begin(), indices.end(), *this);

    // step 3: use sorted indices
    std::auto_ptr<Collection> sorted(new Collection()); sorted->reserve(nselected);
    std::vector<reco::CandidateBaseRef> sortedRefs;     sortedRefs.reserve(nselected); 

    std::auto_ptr<Collection> sortedRejected, sortedAll;
    std::vector<reco::CandidateBaseRef> sortedRefsRejected, sortedRefsAll;
    if (saveRejected_) { sortedRejected = std::auto_ptr<Collection>(new Collection());  }
    if (saveAll_     ) { sortedAll      = std::auto_ptr<Collection>(new Collection());  }

    for (size_t i = 0; i < nselected; ++i) {
        size_t idx = indices[i];

        bool ok = ( ( marks_[idx] & (~bitsToIgnore_) ) == 0);
        if (makeSummary_) { addToSummary(marks_[idx], ok); }

        if (ok) { // save only unmarked items
            sorted->push_back( selected_[idx] );
            sortedRefs.push_back( originalRefs_[idx] );
        } else if (saveRejected_) {
            sortedRejected->push_back( selected_[idx] );
            sortedRefsRejected.push_back( originalRefs_[idx] );
        }
        if (saveAll_) {
            sortedAll->push_back( selected_[idx] );
            sortedRefsAll.push_back( originalRefs_[idx] );
        }
    }

    // FIRST put the collection(s)
    edm::OrphanHandle<Collection> newRefProd = event_->put(sorted, label_);
    edm::OrphanHandle<Collection> newRefProdRejected, newRefProdAll;
    if (saveRejected_) newRefProdRejected = event_->put(sortedRejected, labelRejected_);
    if (saveAll_     ) newRefProdAll      = event_->put(sortedAll     , labelAll_     );

    // THEN fill the map and put it in the event
    // fill in backrefs
    reco::CandRefValueMap::Filler backRefFiller(*backRefs);
    backRefFiller.insert(newRefProd, sortedRefs.begin(), sortedRefs.end());
    backRefFiller.fill();
    event_->put(backRefs, label_);
    if (saveRejected_) {
        reco::CandRefValueMap::Filler backRefFiller(*backRefsRejected);
        backRefFiller.insert(newRefProdRejected, sortedRefsRejected.begin(), sortedRefsRejected.end());
        backRefFiller.fill();
        event_->put(backRefsRejected, labelRejected_);
    }
    if (saveAll_) {
        reco::CandRefValueMap::Filler backRefFiller(*backRefsAll);
        backRefFiller.insert(newRefProdAll, sortedRefsAll.begin(), sortedRefsAll.end());
        backRefFiller.fill();
        event_->put(backRefsAll, labelAll_);
    }

    cleanup();
}

template<typename T, typename T2, typename Collection, typename Comparator>
void CleanerHelper<T,T2,Collection,Comparator>::cleanup() {
    selected_.clear();
    marks_.clear();
    originalRefs_.clear();

    event_ = 0;
}

template<typename T, typename T2, typename Collection, typename Comparator>
void CleanerHelper<T,T2,Collection,Comparator>::setStatusMark(bool setIt) {
    markItems_ = setIt;
    if (markItems_) {
        // I check that T2 derives from reco::Particle
        // this would be needed if we relaxed the constraint that T2 must inherit from Candidate
        //if (! boost::is_base_of<reco::Particle,T2>::value ) {
        //    throw cms::Exception("Wrong Type") << "You can't use 'setStatusMark' " << 
        //        " when producting objects of type " << typeid(T2).name() << 
        //        " because they don't inherit from reco::Particle.\n" <<
        //        " hint: use c++filt demangle the type name in this error message.\n";
        //}
    }      
}


template<typename T, typename T2, typename Collection, typename Comparator>
void CleanerHelper<T,T2,Collection,Comparator>::addToSummary(uint32_t bits, bool ok) {
    uint8_t bit; uint32_t mask;
    for (bit = 0, mask = 1; bit < NumberOfBits_; ++bit, mask <<= 1) {
        countsForEachBit_[bit][Tested_]++;
        if (mask & bits) countsForEachBit_[bit][Failed_]++;
    }
    countsTotal_[Tested_]++; if (!ok) countsTotal_[Failed_]++;
}

template<typename T, typename T2, typename Collection, typename Comparator>
std::string CleanerHelper<T,T2,Collection,Comparator>::printSummary() {
    using namespace std;
    ostringstream out;
    out << "   * Summary info: try " << countsTotal_[Tested_] << ", fail " << countsTotal_[Failed_] << "\n";
    for (uint8_t bit = 0; bit < NumberOfBits_; ++bit) {
        out << "   * Bit " << setw(3) << int(bit) << // the cast is required: uint8_t is a char, so it prints as char if we don't convert it
                ": try "   << setw(7) << countsForEachBit_[bit][Tested_] << 
                ", fail "  << setw(7) << countsForEachBit_[bit][Failed_] << 
                " (label " << left << setw(20) << pat::Flags::bitToString(1 << bit) << right << ")\n";
    }
    out << "\n";
    return out.str();
}

template<typename T, typename T2, typename Collection, typename Comparator>
void CleanerHelper<T,T2,Collection,Comparator>::endJob() {
}

} } // namespaces

#endif
