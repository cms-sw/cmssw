#ifndef DataFormats_Common_MultiAssociation_h
#define DataFormats_Common_MultiAssociation_h
/* \class MultiAssociation
 *
 * \author Giovanni Petrucciani, SNS Pisa and CERN PH-CMG
 * 
 * One-to-Many variant of edm::Association<C> / edm::ValueMap<T>, 
 *
 * Given a key, it will return a range of iterators within a collection (fast access), or collection by value (slow access mode)
 *
 * The range of iterators is handled through boost::sub_range, so it should feel like a collection:
 *   1) it has a '::const_iterator', a 'begin()' and an 'end()'
 *   2) it has a 'size()' and an 'empty()'
 *   3) it has a 'front()', 'back()'
 *   4) it works as an array  (i.e. you can use range[index] to pick an element)
 *   5) if your MultiAssociation is not const, you can modify the values associated to each key
 *      (but you can't push_back new values for a given key)
 * ( details at http://www.boost.org/doc/libs/1_37_0/libs/range/doc/utility_class.html#sub_range )
 *
 * The collection can be a RefVector<C> (to work a la edm::Association<C>), a std::vector<T> (to work a la ValueMap<T>), a PtrVector<T>...
 * The collection must be compatible with sub_range and support a few other features. Usually you need:
 *  - that it has a default constructor
 *  - that it has a const_iterator, and "begin() const" returns such const_iterator. 
 *  - that it has an iterator, and "begin()" returns such iterator (note: it doesn't have to be writable, it can be const as well) 
 *  - that it has a swap method
 *  - that 'begin() + offset' is legal (and fast, otherwise this thing is will be awfully slow)
 *  - that you can push_back on a C the * of a C::const_iterator. Namely this should be legal
 *    <code>
 *       C::const_iterator it = ..something..
 *       C someOtherC = ...
 *       someOtherC.push_back(*it);
 *    </code>
 *
 * It can be filled through a FastFiller or a LazyFiller. 
 * FastFiller is probably faster, but has many constraints:
 * - you must fill only one product id at time
 * - you must fill items in strict key order
 * - there is no way to abort a filling operation
 * Try LazyFiller first, unless you're doing some sort of batch task that satisfies the FastFiller requirements
 *
 * It stores:
 *  - for each collection of keys: a product id and an offset  
 *  - for each key (even those not mapped to any value): one offset
 *  - for each value: one value
 * With respect to ValueMap / Association, there is one extra int32 for each key (but we don't store null values)
 *
 * Its backbone is given by edm::helper::IndexRangeAssociation, that maps keys to ranges, and is not templated.
 *
 */

#include <vector>
#include <map>
#include <memory>
#include <boost/utility.hpp>
#include <boost/range.hpp>
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"

namespace edm {
  namespace helper {
      /// Base class to map to items one a range within a target collection. 
      /// All items in a target collection must be mapped to one and only one key.
      /// The target collection will normally be a RefVector (also PtrVector and std::vector could work)
      /// To be used by MultiAssociation. Any other usage beyond MultiAssociation is not supported.
      /** This class holds:
           - ref_offsets_: A non decreasing list of offsets in the target collection, with one item for each key *plus one end item*.
                           A range is given by a consecutive pair of offsets.
                           While filling the map, some items can be '-1', but not the last one.
           - id_offsets_ : A list of pairs, mapping a product id to its associated range in ref_offsets_
          This class can be filled through a FastFiller, that requires to fill keys and offsets in strictly increasing order, and without gaps. 
          Only one FastFiller can exist at time, and each FastFiller fills only for a given key collection. */ 
      class IndexRangeAssociation {
          public:
              typedef std::pair<unsigned int, unsigned int> range;
            
              IndexRangeAssociation() : isFilling_(false) {}

              /// Get a range of indices for this key. RefKey can be edm::Ref, edm::Ptr, edm::RefToBase
              /// And end value of -1 means 'until the end of the other collection, AFAICT'
              template<typename RefKey>
              range operator[](const RefKey &r) const {  
                  return get(r.id(), r.key()); 
              }
              
              /// Get a range of indices for this product id and key (non-template version)
              /// And end value of -1 means 'until the end of the other collection, AFAICT'
              range get(const edm::ProductID & id, unsigned int t) const ;

              /// True if this IndexRangeAssociation has info for this product id
              bool  contains(ProductID id) const ;

              /// Size of this collection (number of keys)
              unsigned int size() const { return ref_offsets_.empty() ? 0 : ref_offsets_.size() - 1; }
            
              /// True if it's empty (no keys)
              bool empty() const { return ref_offsets_.empty(); }

              /// FastFiller for the IndexRangeAssociation:
              /// It requires to fill items in strict key order.
              /// You can have a single FastFiller for a given map at time
              /// You can't access the map for this collection while filling it
              class FastFiller : boost::noncopyable {
                  public:
                      /// Make a filler for a collection with a given product id and size
                      FastFiller(IndexRangeAssociation &assoc, ProductID id, unsigned int size) ;

                      /// When the FastFiller goes out of scope, it unlocks the map so you can make a new one
                      ~FastFiller() ;

                      /// Sets the starting offset for this key.
                      template<typename RefKey> 
                      void insert(const RefKey &r, unsigned int startingOffset, unsigned int size) {
                         insert(r.id(), r.key(), startingOffset, size);
                      }

                      /// Sets the starting offset for this key (non-templated variant)
                      void insert(edm::ProductID id, unsigned int key, unsigned int startingOffset, unsigned int size) ;
                  private:
                      IndexRangeAssociation & assoc_;
                      const ProductID         id_;
                      unsigned int start_, end_; // indices into assoc_.ref_offsets_ (end_ points to the end marker, so it's valid)
                      /// last key used to fill (to check that the new key must be strictly greater than lastKey_)
                      int lastKey_; 
              }; // FastFiller
              friend class FastFiller;

              void swap(IndexRangeAssociation &other) ;

              static void throwUnexpectedProductID(ProductID found, ProductID expected, const char *where) ;
          private:
              typedef std::pair<edm::ProductID,unsigned int> id_off_pair;
              typedef std::vector<id_off_pair> id_offset_vector; // sorted by product id
              typedef std::vector<int> offset_vector; 
              id_offset_vector id_offsets_;  
              offset_vector    ref_offsets_; 

              bool isFilling_; // transient, to check no two fillers exist at the same time
              struct IDComparator { 
                bool operator()(const id_off_pair &p, const edm::ProductID &id) const {  return p.first < id; }
              }; // IDComparator

              
    };

    // Free swap function
    inline void swap(IndexRangeAssociation& lhs, IndexRangeAssociation& rhs) { lhs.swap(rhs); }

  } // Helper Namespace
 
  template<typename C>
  class MultiAssociation {
  public:
    typedef C Collection;
    typedef boost::sub_range<const Collection> const_range;
    typedef boost::sub_range<Collection>       range;

    MultiAssociation() {}

    /// Get a range of values for this key (fast)
    template<typename RefKey> 
    const_range operator[](const RefKey &r) const { 
        return get(r.id(), r.key()); 
    }
    // ---- and the non-const sister
    /// Get a range of values for this key (fast)
    template<typename RefKey> 
    range operator[](const RefKey &r) { 
        return get(r.id(), r.key()); 
    }

    /// Get a copy of the values for this key (slow!)
    template<typename RefKey> 
    Collection getValues(const RefKey &r) const { 
        return getValues(r.id(), r.key()); 
    }

    /// True if there are keys from this product id
    bool contains(const edm::ProductID &id) const { return indices_.contains(id); }

    /// Get a range of values for this product id and index (fast)
    const_range get(const edm::ProductID &id, unsigned int t) const ;
    // ---- and the non-const sister
    /// Get a range of values for this product id and index (fast)
    range get(const edm::ProductID &id, unsigned int t) ;

    /// Get a copy of the values for this product id and index (slow!)
    Collection getValues(const edm::ProductID &id, unsigned int t) const ;
    
    void swap(MultiAssociation &other) {
        indices_.swap(other.indices_);
        data_.swap(other.data_);
    }

    /// Returns the number of values
    unsigned int dataSize()    const { return data_.size();    }

    /// Returns the number of keys
    unsigned int size() const { return indices_.size(); }

    /// Returns true if there are no keys
    bool empty() const { return indices_.empty(); }

    /// FastFiller for the MultiAssociation. 
    /// It is fast, but it requires to fill items in strict key order.
    /// You can have a single FastFiller for a given map at time
    /// You can't access the map for this collection while filling it
    class FastFiller {
        public:
            template<typename HandleType>
            FastFiller(MultiAssociation &assoc, const HandleType &handle) : 
                assoc_(assoc), indexFiller_(new IndexFiller(assoc_.indices_, handle.id(), handle->size())) {}

            FastFiller(MultiAssociation &assoc, edm::ProductID id, unsigned int size) : 
                assoc_(assoc), indexFiller_(new IndexFiller(assoc_.indices_, id, size)) {}

            ~FastFiller() {}
        
            /// Sets the Collection values associated to this key, making copies of those in refs
            template<typename KeyRef>
            void setValues(const KeyRef &k, const Collection &refs) { setValues(k.id(), k.key(), refs); }

            /// Sets the Collection values associated to this key, making copies of those in refs
            void setValues(const edm::ProductID &id, unsigned int key, const Collection &refs);
        private:
            MultiAssociation &  assoc_;
            typedef edm::helper::IndexRangeAssociation::FastFiller IndexFiller;
            std::shared_ptr<IndexFiller> indexFiller_;

    }; // FastFiller
    friend class FastFiller;

    template<typename HandleType>
    FastFiller fastFiller(const HandleType &handle) { return FastFiller(*this, handle); }

    /// LazyFiller for the MultiAssociation. 
    /// It is slower than FastFiller, as it keeps a copy of the input before filling it (unless you use swapValues)
    /// Anyway it has no constraints on the order of the keys, or the number of fillers
    /// It can also be copied around by value without cloning its keys (e.g to put it into a std::vector)
    /// If you set fillOnExit to 'true', it will fill the MultiAssociation automatically when going out of scope
    class LazyFiller {
        public:
            template<typename HandleType>
            LazyFiller(MultiAssociation &assoc, const HandleType &handle, bool fillOnExit=false) :
                assoc_(assoc), 
                id_(handle.id()), size_(handle->size()), 
                tempValues_(new TempValues()), fillOnExit_(fillOnExit) {}
            ~LazyFiller() noexcept(false) { if (fillOnExit_) fill(); } 

            /// Does the real filling. Until this is called, the map is not modified at all.
            /// Calling this twice won't have any effect (but you can't modify a LazyFiller 
            /// after calling 'fill()')
            /// Implementation note: inside, it just makes a FastFiller and uses it.
            void fill() noexcept(false);

            /// If set to true, the LazyFiller wil call 'fill()' when it goes out of scope
            void setFillOnExit(bool fillOnExit) { fillOnExit_ = fillOnExit; }

            /// Sets the Collection values associated to this key, making copies of those in refs
            template<typename KeyRef>
            void setValues(const KeyRef &k, const Collection &refs) ;

            /// Swaps the Collection values associated to this key with the ones in 'ref'
            /// This is expected to be faster than 'setValues'.
            template<typename KeyRef>
            void swapValues(const KeyRef &k, Collection &refs) ;
        private:
            typedef std::map<unsigned int, Collection> TempValues;
            MultiAssociation & assoc_;
            ProductID id_; 
            unsigned int size_;
            std::shared_ptr<TempValues> tempValues_;
            bool fillOnExit_;
    }; // LazyFiller
    friend class LazyFiller;

    template<typename HandleType>
    LazyFiller lazyFiller(const HandleType &h, bool fillOnExit=false) { return LazyFiller(*this, h, fillOnExit); }

    //Used by ROOT storage
    CMS_CLASS_VERSION(10)

  private:
    typedef helper::IndexRangeAssociation Indices;
    Indices    indices_;
    Collection data_;

}; // class

  // Free swap function
  template <typename C>
  inline void swap(MultiAssociation<C>& lhs, MultiAssociation<C>& rhs) { lhs.swap(rhs); }

  //============= IMPLEMENTATION OF THE METHODS =============
  template<typename C>
  typename MultiAssociation<C>::const_range
  MultiAssociation<C>::get(const edm::ProductID & id, unsigned int key) const {
    Indices::range idxrange = indices_.get(id,key);
    return const_range(data_.begin()+idxrange.first, data_.begin()+idxrange.second);
  }

  template<typename C>
  typename MultiAssociation<C>::range
  MultiAssociation<C>::get(const edm::ProductID & id, unsigned int key) {
    Indices::range idxrange = indices_.get(id,key);
    return range(data_.begin()+idxrange.first, data_.begin()+idxrange.second);
  }


  template<typename C>
  typename MultiAssociation<C>::Collection
  MultiAssociation<C>::getValues(const edm::ProductID & id, unsigned int key) const {
     Collection ret;
     const_range values = get(id,key);
     for (typename const_range::const_iterator it = values.begin(), ed = values.end(); it != ed; ++it) {
        ret.push_back(*it);
     }
     return ret;
  }

  template<typename C>
  void MultiAssociation<C>::FastFiller::setValues(const edm::ProductID & id, unsigned int key, const Collection &vals) {
     indexFiller_->insert(id, key, assoc_.data_.size(), vals.size());
     for (typename Collection::const_iterator it = vals.begin(), ed = vals.end(); it != ed; ++it) {
        assoc_.data_.push_back(*it);
     }
  } 
  
  template<typename C>
  template<typename KeyRef>
  void MultiAssociation<C>::LazyFiller::setValues(const KeyRef &k, const Collection &vals) {
     if (k.id() != id_) Indices::throwUnexpectedProductID(k.id(),id_,"LazyFiller::insert");  
     (*tempValues_)[k.key()] = vals;
  }

  template<typename C>
  template<typename KeyRef>
  void MultiAssociation<C>::LazyFiller::swapValues(const KeyRef &k, Collection &vals) {
     if (k.id() != id_) Indices::throwUnexpectedProductID(k.id(),id_,"LazyFiller::insert");  
     vals.swap((*tempValues_)[k.key()]);
  }


  template<typename C>
  void MultiAssociation<C>::LazyFiller::fill() {
     if (id_ != ProductID()) { // protection against double filling
        typename MultiAssociation<C>::FastFiller filler(assoc_, id_, size_);
        for (typename TempValues::const_iterator it = tempValues_->begin(), ed = tempValues_->end(); it != ed; ++it) {
           filler.setValues(id_, it->first, it->second);                
        }
        id_ = ProductID();     // protection against double filling
     }
  }

 
} // namespace

#endif
