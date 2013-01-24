#ifndef FWCore_Utilities_ThreadSafeIndexedRegistry_h
#define FWCore_Utilities_ThreadSafeIndexedRegistry_h

#include <vector>
#include "boost/thread.hpp"

// ----------------------------------------------------------------------

/// A ThreadSafeIndexedRegistry is used to keep track of the instances of
/// some type 'value_typed', stored in a vector.
///
/// This class is sufficiently thread-safe to be usable in a
/// thread-safe manner. Don't let  the name mislead you  into thinking
/// it provides more guarantee than that!
///
// ----------------------------------------------------------------------

#pragma GCC visibility push(default)
namespace edm {
  namespace detail {
    struct Empty { };

    template <typename T, typename E=Empty>
    class ThreadSafeIndexedRegistry {
    public:
      typedef T     value_type;
      typedef E     extra_type;
      typedef typename std::vector<value_type> collection_type;
      typedef typename collection_type::size_type      size_type;

      typedef typename collection_type::const_iterator const_iterator;

      static ThreadSafeIndexedRegistry* instance();

      /// Retrieve the value_type object with the given index.
      /// If we return 'true', then 'result' carries the
      /// value_type object.
      /// If we return 'false, no matching index was found, and
      /// the value of 'result' is undefined.
      void getMapped(size_type index, value_type& result) const;

      /// put the given value_type object into the
      /// registry.
      bool insertMapped(value_type const& v);

      /// put the value_type objects in the given collection
      /// into the registry.
      void insertCollection(collection_type const& c);

      /// Return true if there are no contained value_type objects.
      bool empty() const;

      /// Return true if there are any contained value_type objects.
      bool notEmpty() const;

      /// Return the number of contained value_type objects.
      size_type size() const;

      /// Allow iteration through the contents of the registry. Only
      /// const access is provided to the entries of the registry.
      const_iterator begin() const;
      const_iterator end() const;

      /// Print the contents of this registry to the given ostream.
      void print(std::ostream& os) const;

      /// Provide access to the contained collection
      collection_type& data();
      collection_type const& data() const;

      /// Provide access to the appendage "extra". The
      /// ThreadSafeIndexedRegistry doesn't know what this is for, but
      /// instantiations of the template can use it.
      extra_type& extra();
      extra_type const& extra() const;      

    private:
      ThreadSafeIndexedRegistry();
      ~ThreadSafeIndexedRegistry();

      // The following two are not implemented.
      ThreadSafeIndexedRegistry(ThreadSafeIndexedRegistry<T, E> const&); 
    
      ThreadSafeIndexedRegistry<T, E>& 
      operator= (ThreadSafeIndexedRegistry<T, E> const&);

      collection_type data_;
      extra_type      extra_;

    };

    template <typename T, typename E>
    inline
    std::ostream&
    operator<< (std::ostream& os, ThreadSafeIndexedRegistry<T, E> const& reg) {
      reg.print(os);
      return os;
    }

    template <typename T, typename E>
    void 
    ThreadSafeIndexedRegistry<T, E>::insertCollection(collection_type const& c) {
      for (typename collection_type::const_iterator it = c.begin(), itEnd = c.end(); it != itEnd; ++it) {
	insertMapped(*it);
      }
    }

    template <typename T, typename E>
    inline
    bool
    ThreadSafeIndexedRegistry<T, E>::empty() const {
      return data_.empty();
    }
    
    template <typename T, typename E>
    inline
    bool
    ThreadSafeIndexedRegistry<T, E>::notEmpty() const {
      return !empty();
    }

    template <typename T, typename E>
    inline
    typename ThreadSafeIndexedRegistry<T, E>::size_type
    ThreadSafeIndexedRegistry<T, E>::size() const {
      return data_.size();
    }

    template <typename T, typename E>
    inline
    typename ThreadSafeIndexedRegistry<T, E>::const_iterator
    ThreadSafeIndexedRegistry<T, E>::begin() const {
      return data_.begin();
    }

    template <typename T, typename E>
    inline
    typename ThreadSafeIndexedRegistry<T, E>::const_iterator
    ThreadSafeIndexedRegistry<T, E>::end() const {
      return data_.end();
    }
    
    template <typename T, typename E>
    void
    ThreadSafeIndexedRegistry<T, E>::print(std::ostream& os) const {
      os << "Registry with " << size() << " entries\n";
      for (const_iterator i = begin(), e = end(); i != e; ++i) {
	  os << i - begin() << " " << i << '\n';
      }
    }

    template <typename T, typename E>
    inline
    typename ThreadSafeIndexedRegistry<T, E>::collection_type&
    ThreadSafeIndexedRegistry<T, E>::data() {
      return data_;
    }

    template <typename T, typename E>
    inline
    typename ThreadSafeIndexedRegistry<T, E>::extra_type&
    ThreadSafeIndexedRegistry<T, E>::extra() {
      return extra_;
    }

    template <typename T, typename E>
    inline
    typename ThreadSafeIndexedRegistry<T, E>::extra_type const&
    ThreadSafeIndexedRegistry<T, E>::extra() const {
      return extra_;
    }

    template <typename T, typename E>
    inline
    typename ThreadSafeIndexedRegistry<T, E>::collection_type const&
    ThreadSafeIndexedRegistry<T, E>::data() const {
      return data_;
    }

    template <typename T, typename E> 
    ThreadSafeIndexedRegistry<T, E>::ThreadSafeIndexedRegistry() : 
      data_()
    { }


    template <typename T, typename E> 
    ThreadSafeIndexedRegistry<T, E>::~ThreadSafeIndexedRegistry() 
    { }

  } // namespace detail
} // namespace edm
#pragma GCC visibility pop

#endif //  FWCore_Utilities_ThreadSafeIndexedRegistry_h
