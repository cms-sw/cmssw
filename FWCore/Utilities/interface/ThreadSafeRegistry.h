#ifndef FWCore_Utilities_ThreadSafeRegistry_h
#define FWCore_Utilities_ThreadSafeRegistry_h

#include <map>
#include <vector>
#include "boost/thread.hpp"

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

namespace edm {
  namespace detail {
    struct empty { };

    template <typename KEY, typename T, typename E=empty>
    class ThreadSafeRegistry {
    public:
      typedef KEY   key_type;
      typedef T     value_type;
      typedef E     extra_type;
      typedef typename std::map<key_type, value_type> collection_type;
      typedef typename collection_type::size_type      size_type;

      typedef typename collection_type::const_iterator const_iterator;

      typedef typename std::vector<value_type> vector_type;

      static ThreadSafeRegistry* instance();

      /// Retrieve the value_type object with the given key.
      /// If we return 'true', then 'result' carries the
      /// value_type object.
      /// If we return 'false, no matching key was found, and
      /// the value of 'result' is undefined.
      bool getMapped(key_type const& k, value_type& result) const;

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
      /// ThreadSafeRegistry doesn't know what this is for, but
      /// instantiations of the template can use it.
      extra_type& extra();
      extra_type const& extra() const;      

    private:
      ThreadSafeRegistry();
      ~ThreadSafeRegistry();

      // The following two are not implemented.
      ThreadSafeRegistry(ThreadSafeRegistry<KEY,T,E> const&); 
    
      ThreadSafeRegistry<KEY,T,E>& 
      operator= (ThreadSafeRegistry<KEY,T,E> const&);

      collection_type data_;
      extra_type      extra_;

      static ThreadSafeRegistry* instance_;

      static boost::mutex registry_mutex;
    };

    template <typename KEY, typename T, typename E>
    inline
    std::ostream&
    operator<< (std::ostream& os, ThreadSafeRegistry<KEY,T,E> const& reg) {
      reg.print(os);
      return os;
    }

    // ----------------------------------------------------------------------
    // Declarations of static data for classes instantiated from the
    // class template.

    template <typename KEY, typename T, typename E>
    ThreadSafeRegistry<KEY,T,E>* ThreadSafeRegistry<KEY,T,E>::instance_ = 0;

    template <typename KEY, typename T, typename E>
    boost::mutex ThreadSafeRegistry<KEY,T,E>::registry_mutex;

    // ----------------------------------------------------------------------

    template <typename KEY, typename T, typename E>
    ThreadSafeRegistry<KEY,T,E>*
    ThreadSafeRegistry<KEY,T,E>::instance() {
      if (instance_ == 0) {
	boost::mutex::scoped_lock lock(registry_mutex);
	if (instance_ == 0) {
	  static ThreadSafeRegistry<KEY,T,E> me;
	  instance_ = &me;
	}
      }
      return instance_;      
    }

    template <typename KEY, typename T, typename E>
    bool 
    ThreadSafeRegistry<KEY,T,E>::getMapped(key_type const& k, value_type& result) const {
      bool found;
      const_iterator i;
      {
	// This scope limits the lifetime of the lock to the shorted
	// required interval.
	boost::mutex::scoped_lock lock(registry_mutex);
	i = data_.find(k);
	found = (i != data_.end());
      }
      if (found) result = i->second;
      return found;
    }

    template <typename KEY, typename T, typename E>
    bool 
    ThreadSafeRegistry<KEY,T,E>::insertMapped(value_type const& v) {
      bool newly_added = false;
      boost::mutex::scoped_lock lock(registry_mutex);

      key_type id = v.id();
      if (data_.find(id) == data_.end()) {
	  data_[id] = v;
	  newly_added = true;
      }
      return newly_added;
    }

    template <typename KEY, typename T, typename E>
    void 
    ThreadSafeRegistry<KEY,T,E>::insertCollection(collection_type const& c) {
      for (typename collection_type::const_iterator it = c.begin(), itEnd = c.end(); it != itEnd; ++it) {
	insertMapped(it->second);
      }
    }

    template <typename KEY, typename T, typename E>
    void 
    ThreadSafeRegistry<KEY,T,E>::insertCollection(vector_type const& c) {
      for (typename vector_type::const_iterator it = c.begin(), itEnd = c.end(); it != itEnd; ++it) {
	insertMapped(*it);
      }
    }

    template <typename KEY, typename T, typename E>
    inline
    bool
    ThreadSafeRegistry<KEY,T,E>::empty() const {
      return data_.empty();
    }
    
    template <typename KEY, typename T, typename E>
    inline
    bool
    ThreadSafeRegistry<KEY,T,E>::notEmpty() const {
      return !empty();
    }

    template <typename KEY, typename T, typename E>
    inline
    typename ThreadSafeRegistry<KEY,T,E>::size_type
    ThreadSafeRegistry<KEY,T,E>::size() const {
      return data_.size();
    }

    template <typename KEY, typename T, typename E>
    inline
    typename ThreadSafeRegistry<KEY,T,E>::const_iterator
    ThreadSafeRegistry<KEY,T,E>::begin() const {
      return data_.begin();
    }

    template <typename KEY, typename T, typename E>
    inline
    typename ThreadSafeRegistry<KEY,T,E>::const_iterator
    ThreadSafeRegistry<KEY,T,E>::end() const {
      return data_.end();
    }
    
    template <typename KEY, typename T, typename E>
    void
    ThreadSafeRegistry<KEY,T,E>::print(std::ostream& os) const {
      os << "Registry with " << size() << " entries\n";
      for (const_iterator i=begin(), e=end(); i!=e; ++i) {
	  os << i->first << " " << i->second << '\n';
      }
    }

    template <typename KEY, typename T, typename E>
    inline
    typename ThreadSafeRegistry<KEY,T,E>::collection_type&
    ThreadSafeRegistry<KEY,T,E>::data() {
      return data_;
    }

    template <typename KEY, typename T, typename E>
    inline
    typename ThreadSafeRegistry<KEY,T,E>::extra_type&
    ThreadSafeRegistry<KEY,T,E>::extra() {
      return extra_;
    }

    template <typename KEY, typename T, typename E>
    inline
    typename ThreadSafeRegistry<KEY,T,E>::extra_type const&
    ThreadSafeRegistry<KEY,T,E>::extra() const {
      return extra_;
    }

    template <typename KEY, typename T, typename E>
    inline
    typename ThreadSafeRegistry<KEY,T,E>::collection_type const&
    ThreadSafeRegistry<KEY,T,E>::data() const {
      return data_;
    }

    template <typename KEY, typename T, typename E> 
    ThreadSafeRegistry<KEY,T,E>::ThreadSafeRegistry() : 
      data_()
    { }


    template <typename KEY, typename T, typename E> 
    ThreadSafeRegistry<KEY,T,E>::~ThreadSafeRegistry() 
    { }

  } // namespace detail
} // namespace edm

#endif //  FWCore_Utilities_ThreadSafeRegistry_h
