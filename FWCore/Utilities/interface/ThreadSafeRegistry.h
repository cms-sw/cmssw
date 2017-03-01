#ifndef FWCore_Utilities_ThreadSafeRegistry_h
#define FWCore_Utilities_ThreadSafeRegistry_h

#include <map>
#include <vector>
#include <ostream>
#include <mutex>

// ----------------------------------------------------------------------

/// A ThreadSafeRegistry is used to keep track of the instances of
/// some type 'value_typed'.  These objects are each associated with
/// a given 'key_type' object, which must be retrievable from the
/// value_type object, and which must uniquely identify the
/// value_type's value.
///
/// This class is sufficiently thread-safe to be usable in a
/// thread-safe manner. Don't let  the name mislead you  into thinking
/// it provides more guarantee than that!
///
/// If 'm' is of type 'value_type const&', the expression
///
///    key_type k = m.id();
///
///  must be legal, and must return the unique key associated with
///  the value of 'm'.
// ----------------------------------------------------------------------

#pragma GCC visibility push(default)
namespace edm {
  namespace detail {

    template <typename KEY, typename T>
    class ThreadSafeRegistry {
    public:
      typedef KEY   key_type;
      typedef T     value_type;
      typedef typename std::map<key_type, value_type> collection_type;
      typedef typename collection_type::size_type      size_type;

      typedef typename std::vector<value_type> vector_type;

      static ThreadSafeRegistry* instance();

      /// Retrieve the value_type object with the given key.
      /// If we return 'true', then 'result' carries the
      /// value_type object.
      /// If we return 'false, no matching key was found, and
      /// the value of 'result' is undefined.
      bool getMapped(key_type const& k, value_type& result) const;
      
      /** Retrieve a pointer to the value_type object with the given key.
       If there is no object associated with the given key 0 is returned.
       */
      value_type const* getMapped(key_type const& k) const;

      /// Insert the given value_type object into the
      /// registry. If there was already a value_type object
      /// with the same key, we don't change it. This should be OK,
      /// since it should have the same contents if the key is the
      /// same.  Return 'true' if we really added the new
      /// value_type object, and 'false' if the
      /// value_type object was already present.
      bool insertMapped(value_type const& v);

      /// put the value_type objects in the given collection
      /// into the registry.
      void insertCollection(collection_type const& c);
      void insertCollection(vector_type const& c);

      /// Return true if there are no contained value_type objects.
      bool empty() const;

      /// Return true if there are any contained value_type objects.
      bool notEmpty() const;

      /// Return the number of contained value_type objects.
      size_type size() const;

      /// Print the contents of this registry to the given ostream.
      void print(std::ostream& os) const;

    private:
      ThreadSafeRegistry();
      ~ThreadSafeRegistry();

      // The following two are not implemented.
      ThreadSafeRegistry(ThreadSafeRegistry<KEY,T> const&); 
    
      ThreadSafeRegistry<KEY,T>& 
      operator= (ThreadSafeRegistry<KEY,T> const&);

      mutable std::mutex mutex_;
      collection_type data_;
    };

    template <typename KEY, typename T, typename E>
    inline
    std::ostream&
    operator<< (std::ostream& os, ThreadSafeRegistry<KEY,T> const& reg) {
      reg.print(os);
      return os;
    }

    template <typename KEY, typename T>
    void 
    ThreadSafeRegistry<KEY,T>::insertCollection(collection_type const& c) {
      for (auto const& item: c ) {
	insertMapped(item.second);
      }
    }

    template <typename KEY, typename T>
    void 
    ThreadSafeRegistry<KEY,T>::insertCollection(vector_type const& c) {
      for (auto const& item: c) {
	insertMapped(item);
      }
    }

    template <typename KEY, typename T>
    inline
    bool
    ThreadSafeRegistry<KEY,T>::empty() const {
      std::lock_guard<std::mutex> guard(mutex_);
      return data_.empty();
    }
    
    template <typename KEY, typename T>
    inline
    bool
    ThreadSafeRegistry<KEY,T>::notEmpty() const {
      return !empty();
    }

    template <typename KEY, typename T>
    inline
    typename ThreadSafeRegistry<KEY,T>::size_type
    ThreadSafeRegistry<KEY,T>::size() const {
      std::lock_guard<std::mutex> guard(mutex_);
      return data_.size();
    }

    template <typename KEY, typename T>
    void
    ThreadSafeRegistry<KEY,T>::print(std::ostream& os) const {
      std::lock_guard<std::mutex> guard(mutex_);
      os << "Registry with " << size() << " entries\n";
      for (auto const& item: data_) {
	  os << item.first << " " << item.second << '\n';
      }
    }

    template <typename KEY, typename T> 
    ThreadSafeRegistry<KEY,T>::ThreadSafeRegistry() : 
      data_()
    { }


    template <typename KEY, typename T> 
    ThreadSafeRegistry<KEY,T>::~ThreadSafeRegistry() 
    { }

  } // namespace detail
} // namespace edm
#pragma GCC visibility pop
#endif //  FWCore_Utilities_ThreadSafeRegistry_h
