#ifndef DataFormats_Common_AssociativeIterator_h
#define DataFormats_Common_AssociativeIterator_h
/**
 * \class AssociativeIterator<KeyRefType, AssociativeCollection>
 *
 * \author Giovanni Petrucciani, SNS Pisa
 *
 * Allows iteraton on a "new style" associative container (ValueMap, Association)
 * as a collection of std::pair<key, value>.
 *
 * The key is built on the fly using a helper (ItemGetter) that produces keys given ProductID and index
 * At the moment such a helper is available only in full framework (EdmEventItemGetter<RefType>)
 * 
 * KeyRefType can be Ref<C>, RefToBase<T>, Ptr<T>
 * AssociativeCollection can be ValueMap<V>, Association<C>
 *
 * Example usage is as follows:
 *   Handle<ValueMap<double> > hDiscriminators;
 *   iEvent.getByLabel(..., hDiscriminators);
 *   AssociativeIterator<RefToBase<Jet>, ValueMap<double> > itBTags(*hDiscriminators, EdmEventItemGetter(iEvent)), endBTags = itBTags.end();
 *   for ( ; itBTags != endBTags; ++itBTags ) {
 *      cout << " Jet PT = " << itBTags->first->pt() << ", disc = " << itBTags->second << endl;
 *   }
 * or, for edm::Association
 *   Handle<Association<GenParticleCollection> > hMCMatch;
 *   iEvent.getByLabel(..., hMCMatch);
 *   AssociativeIterator<RefToBase<Candidate>, Association<GenParticleCollection> > itMC(*hMCMatch, EdmEventItemGetter(iEvent)), endMC = itMC.end();
 *   for ( ; itMC != endMC; ++itMC ) {
 *      cout << " Particle with PT = " << itMC->first->pt() ;
 *      if (itMC->second.isNull()) { 
 *          cout << " UNMATCHED." << endl;
 *      } else {
 *          cout << " matched. MC PT = " << itMC->second->pt() << endl;
 *      }
 *   }
 *
 *
 *
 */

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/EDProductGetter.h"

namespace edm {
    struct Event;
    template <class T> struct View;
    template <class T> struct Handle;
    template <class T> struct Association;
    template <class T> struct RefToBase;
    template <class T> struct Ptr;
    template <class C, class T, class F> struct Ref;
}

namespace edm {

    // Helper classes to convert one ref type to another.
    // Now it's able to convert anything to itself, and RefToBase to anything else
    // This won't be needed if we used Ptr
    namespace helper {
        template<typename RefFrom, typename RefTo>
        struct RefConverter {
            static RefTo convert(const RefFrom &ref) { return RefTo(ref); }
        };
        template<typename T>
        struct RefConverter<RefToBase<T>, Ptr<T> > {
            static Ptr<T> convert(const RefToBase<T> &ref) { return Ptr<T>(ref.id(), ref.isAvailable() ? ref.get() : 0, ref.key()); }
        };
        template<typename T, typename C, typename V, typename F>
        struct RefConverter<RefToBase<T>, Ref<C,V,F> > {
            static Ref<C,V,F> convert(const RefToBase<T> &ref) { return ref.template castTo<Ref<C,V,F> >(); }
        };
    }

    /// Helper class that fetches some type of Ref given ProductID and index, using the edm::Event
    //  the implementation uses View, and works for RefType = Ref, RefToBase and Ptr
    template<typename RefType>
    class EdmEventItemGetter {
        public: 
            typedef typename RefType::value_type element_type;
            EdmEventItemGetter(const edm::Event &iEvent) : iEvent_(iEvent) { }
            ~EdmEventItemGetter() { }

            RefType get(const ProductID &id, size_t idx) const {
                typedef typename edm::RefToBase<element_type> BaseRefType; // could also use Ptr, but then I can't do Ptr->RefToBase
                if (id_ != id) {
                    id_ = id;
                    iEvent_.get(id_, view_);
                }
                BaseRefType ref = view_->refAt(idx);
                typedef typename helper::RefConverter<BaseRefType, RefType> conv; 
                return conv::convert(ref);
            }
        private:
            mutable Handle<View<element_type> > view_;
            mutable ProductID id_;
            const edm::Event &iEvent_;
    };

    // unfortunately it's not possible to define value_type of an Association<C> correctly
    // so we need yet another template trick
    namespace helper {
        template<typename AC> 
        struct AssociativeCollectionValueType {
            typedef typename AC::value_type type;
        };

        template<typename C>
        struct AssociativeCollectionValueType< Association<C> > {
            typedef typename Association<C>::reference_type type;
        };
    }

template<typename KeyRefType, typename AssociativeCollection, 
            typename ItemGetter = EdmEventItemGetter<KeyRefType> >
class AssociativeIterator {
    public:
        typedef KeyRefType                                  key_type;
        typedef typename KeyRefType::value_type             key_val_type;
        typedef typename helper::AssociativeCollectionValueType<AssociativeCollection>::type  val_type;
        typedef typename std::pair<key_type, val_type>      value_type;

        typedef AssociativeIterator<KeyRefType,AssociativeCollection,ItemGetter>   self_type;

        /// Create the associative iterator, pointing at the beginning of the collection
        AssociativeIterator(const AssociativeCollection &map, const ItemGetter &getter) ;

        self_type & operator++() ;
        self_type & operator--() ;
        self_type & nextProductID() ;
        // self_type & skipTo(const ProductID &id, size_t offs = 0) ; // to be implemented one day

        const value_type & operator*()  const { return *(this->get()); }
        const value_type * operator->() const { return  (this->get()); }
        const value_type * get()        const { chkPair(); return & pair_; }

        const key_type   & key() const { chkPair(); return pair_.first; }
        const val_type   & val() const { return map_.get(idx_);         }
        const ProductID  & id()  const { return ioi_->first; }
        
        operator bool() const { return idx_ < map_.size(); }
        self_type end() const ;
    
        bool operator==(const self_type &other) const { return other.idx_ == idx_; }
        bool operator!=(const self_type &other) const { return other.idx_ != idx_; }
        bool operator<( const self_type &other) const { return other.idx_  < idx_; }

    private:
        typedef typename AssociativeCollection::id_offset_vector  id_offset_vector;
        typedef typename id_offset_vector::const_iterator         id_offset_iterator;
        const AssociativeCollection & map_;
        id_offset_iterator   ioi_, ioi2_;
        size_t               idx_; 

        ItemGetter           getter_;

        mutable bool         pairOk_;
        mutable value_type   pair_;
                
        void chkPair() const ;

    };

    template<typename KeyRefType, typename AC, typename IG>
    AssociativeIterator<KeyRefType,AC,IG>::AssociativeIterator(const AC &map, const IG &getter) :
        map_(map), ioi_(map_.ids().begin()), ioi2_(ioi_+1), idx_(0), 
        getter_(getter),
        pairOk_(false)
    {
    }

    template<typename KeyRefType, typename AC, typename IG>
    AssociativeIterator<KeyRefType,AC,IG> & AssociativeIterator<KeyRefType,AC,IG>::operator++() {
        pairOk_ = false;
        idx_++;
        if (ioi2_ < map_.ids().end()) {
            if (ioi2_->second == idx_) {
                ++ioi_; ++ioi2_;
            }
        }
        return *this;
    }

    template<typename KeyRefType, typename AC, typename IG>
    AssociativeIterator<KeyRefType,AC,IG> & AssociativeIterator<KeyRefType,AC,IG>::operator--() {
        pairOk_ = false;
        idx_--;
        if (ioi_->second < idx_) {
            --ioi_; --ioi2_;
        }
        return *this;

    }

    template<typename KeyRefType, typename AC, typename IG>
    AssociativeIterator<KeyRefType,AC,IG> & AssociativeIterator<KeyRefType,AC,IG>::nextProductID() {
        pairOk_ = false;
        ioi_++; ioi2_++;
        if (ioi_ == map_.ids().end()) {
            idx_ = map_.size();
        } else {
            idx_ = ioi_->second;
        }
    }

    /*
    template<typename KeyRefType, typename AC, typename IG>
    AssociativeIterator<KeyRefType,AC,IG> & AssociativeIterator<KeyRefType,AC,IG>::skipTo(const ProductID &id, size_t offs) {
        pairOk_ = false;
        throw Exception(errors::UnimplementedFeature);
    }
    */

    template<typename KeyRefType, typename AC, typename IG>
    AssociativeIterator<KeyRefType,AC,IG> AssociativeIterator<KeyRefType,AC,IG>::end() const {
        self_type ret(map_, getter_);
        ret.ioi_  = map_.ids().end();
        ret.ioi2_ = ret.ioi_ + 1;
        ret.idx_  = map_.size(); 
        return ret;
    }
   
    template<typename KeyRefType, typename AC, typename IG>
    void AssociativeIterator<KeyRefType,AC,IG>::chkPair() const {
        if (pairOk_) return;
        pair_.first = getter_.get(id(), idx_ - ioi_->second);
        pair_.second = map_.get(idx_);
        pairOk_ = true;
    }

}

#endif

