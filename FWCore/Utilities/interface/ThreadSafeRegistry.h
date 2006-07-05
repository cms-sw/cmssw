#ifndef FWCoreUtilities_ThreadSafeRegistry_h
#define FWCoreUtilities_ThreadSafeRegistry_h

#include <map>
#include "boost/thread.hpp"

// ----------------------------------------------------------------------
// $Id: ThreadSafeRegistry.h,v 1.1.2.2 2006/06/29 21:55:17 paterno Exp $

/// A ThreadSafeRegistry is used to keep track of the instances of
/// some type 'mapped_typed'.  These objects are each associated with
/// a given 'key_type' object, which must be retrievable from the
/// mapped_type object, and which must uniquely identify the
/// mapped_type's value.
///
/// This class is sufficiently thread-safe to be usable in a
/// thread-safe manner. Don't let  the name mislead you  into thinking
/// it provides more guarantee than that!
///
/// If 'm' is of type 'mapped_type const&', the expression
///
///    key_type k = m.id();
///
///  must be legal, and must return the unique key associated with
///  the value of 'm'.
// ----------------------------------------------------------------------

namespace edm
{
  namespace detail
  {
    template <class KEY, class T>
    class ThreadSafeRegistry
    {
    public:
      typedef KEY   key_type;
      typedef T     mapped_type;
      typedef typename std::map<key_type, mapped_type> collection_type;
      typedef typename collection_type::size_type      size_type;

      typedef typename collection_type::const_iterator const_iterator;

      static ThreadSafeRegistry* instance();

      /// Retrieve the mapped_type object with the given key.
      /// If we return 'true', then 'result' carries the
      /// mapped_type object.
      /// If we return 'false, no matching key was found, and
      /// the value of 'result' is undefined.
      bool getMapped(key_type const& k, mapped_type& result) const;

      /// Insert the given mapped_type object into the
      /// registry. If there was already a mapped_type object
      /// with the same key, we don't change it. This should be OK,
      /// since it should have the same contents if the key is the
      /// same.  Return 'true' if we really added the new
      /// mapped_type object, and 'false' if the
      /// mapped_type object was already present.

      bool insertMapped(mapped_type const& v);

      /// Return the number of contained mapped_type objects.
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

    private:
      ThreadSafeRegistry();
      ~ThreadSafeRegistry();

      // The following two are not implemented.
      ThreadSafeRegistry(ThreadSafeRegistry<KEY,T> const&); 
    
      ThreadSafeRegistry<KEY,T>& 
      operator= (ThreadSafeRegistry<KEY,T> const&);

      collection_type data_;

      static ThreadSafeRegistry* instance_;

      static boost::mutex registry_mutex;
    };

    template <class KEY, class T>
    inline
    std::ostream&
    operator<< (std::ostream& os, ThreadSafeRegistry<KEY,T> const& reg)
    {
      reg.print(os);
      return os;
    }

    // ----------------------------------------------------------------------
    // Declarations of static data for classes instantiated from the
    // class template.

    template <class KEY, class T>
    ThreadSafeRegistry<KEY,T>* ThreadSafeRegistry<KEY,T>::instance_ = 0;

    template <class KEY, class T>
    boost::mutex ThreadSafeRegistry<KEY,T>::registry_mutex;

    // ----------------------------------------------------------------------

    template <class KEY, class T>
    ThreadSafeRegistry<KEY,T>*
    ThreadSafeRegistry<KEY,T>::instance()
    {
      if (instance_ == 0)
	{
	  boost::mutex::scoped_lock lock(registry_mutex);
	  if (instance_ == 0)
	    {
	      static ThreadSafeRegistry<KEY,T> me;
	      instance_ = &me;
	    }
	}
      return instance_;      
    }

    template <class KEY, class T>
    bool 
    ThreadSafeRegistry<KEY,T>::getMapped(key_type const& k, 
					 mapped_type& result) const
    {
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

    template <class KEY, class T>
    bool 
    ThreadSafeRegistry<KEY,T>::insertMapped(mapped_type const& v)
    {
      bool newly_added = false;
      boost::mutex::scoped_lock lock(registry_mutex);

      key_type id = v.id();
      if (data_.find(id) == data_.end())
	{
	  data_[id] = v;
	  newly_added = true;
	}
      return newly_added;
    }

    template <class KEY, class T>
    typename ThreadSafeRegistry<KEY,T>::size_type
    ThreadSafeRegistry<KEY,T>::size() const
    {
      return data_.size();
    }

    template <class KEY, class T>
    typename ThreadSafeRegistry<KEY,T>::const_iterator
    ThreadSafeRegistry<KEY,T>::begin() const
    {
      return data_.begin();
    }

    template <class KEY, class T>
    typename ThreadSafeRegistry<KEY,T>::const_iterator
    ThreadSafeRegistry<KEY,T>::end() const
    {
      return data_.end();
    }
    
    template <class KEY, class T>
    void
    ThreadSafeRegistry<KEY,T>::print(std::ostream& os) const
    {
      os << "Registry with " << size() << " entries\n";
      for (const_iterator i=begin(), e=end(); i!=e; ++i)
	{
	  os << i->first << " " << i->second << '\n';
	}
    }

    template <class KEY, class T>
    typename ThreadSafeRegistry<KEY,T>::collection_type&
    ThreadSafeRegistry<KEY,T>::data()
    {
      return data_;
    }

    template <class KEY, class T>
    typename ThreadSafeRegistry<KEY,T>::collection_type const&
    ThreadSafeRegistry<KEY,T>::data() const
    {
      return data_;
    }

    template <class KEY, class T> 
    ThreadSafeRegistry<KEY,T>::ThreadSafeRegistry() : 
      data_()
    { }


    template <class KEY, class T> 
    ThreadSafeRegistry<KEY,T>::~ThreadSafeRegistry() 
    { }

  } // namespace detail
} // namespace edm

#endif //  FWCoreUtilities_ThreadSafeRegistry_h
