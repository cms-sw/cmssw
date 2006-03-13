// ----------------------------------------------------------------------
// $Id: Registry.cc,v 1.5 2006/03/08 22:14:52 wmtan Exp $
//
// ----------------------------------------------------------------------

#include "boost/thread.hpp" // maybe need only thread/mutex.hpp?

#include "FWCore/ParameterSet/interface/Registry.h"

namespace
{
  boost::mutex registry_mutex;
}

namespace edm
{
  namespace pset
  {
    // Is this initialization itself thread-safe?
    Registry* Registry::instance_ = 0;


    Registry*
    Registry::instance()
    {
      if (instance_ == 0)
	{
	  boost::mutex::scoped_lock lock(registry_mutex);
	  if (instance_ == 0)
	    {
	      static Registry me;
	      instance_ = &me;
	    }
	}
      return instance_;
    }

    bool
    Registry::getParameterSet(ParameterSetID const& id,
			      ParameterSet & result) const
    {
      bool found;
      const_iterator i;
      {
        // This scope limits the lifetime of the lock to the shortest
        // required interval.
        boost::mutex::scoped_lock lock(registry_mutex);
        i = psets_.find(id);
        found = (i != psets_.end());
      }
      if (found) result = i->second;
      return found;
    }



    bool
    Registry::insertParameterSet(ParameterSet const& p)
    {
      bool newly_added;
      ParameterSet tracked_part(p.trackedPart());
      ParameterSetID id = tracked_part.id();

      boost::mutex::scoped_lock lock(registry_mutex);
      const_iterator i = psets_.find(id);

      if (i != psets_.end()) // we already have it!
	{
	  newly_added = false;
	}
      else
	{
	  psets_[p.id()] = tracked_part;
	  newly_added = true;
	}
      return newly_added;
    }


    Registry::size_type
    Registry::size() const
    {
      return psets_.size();
    }

    Registry::const_iterator
    Registry::begin() const
    {
      return psets_.begin();
    }

    Registry::const_iterator
    Registry::end() const
    {
      return psets_.end();
    }

    void
    Registry::print(std::ostream& os) const
    {
      const_iterator i = begin();
      const_iterator e = end();
      os << "ParameterSet registry with " << size() << " entries\n";
      for ( ; i != e; ++i )
	{
	  os << i->first << "   " << i->second.toString() << '\n';
	}
    }

    // Private member functions
    Registry::Registry() : psets_() { }

    Registry::~Registry() { }

    // Associated functions
    std::ostream&
    operator<< (std::ostream& os, Registry const& reg)
    {
      reg.print(os);
      return os;
    }
    
    void loadAllNestedParameterSets(ParameterSet const& main)
    {
      Registry* reg = Registry::instance();
      std::vector<ParameterSet> all_main_psets;
      explode(main, all_main_psets);
      std::vector<ParameterSet>::const_iterator i = all_main_psets.begin();
      std::vector<ParameterSet>::const_iterator e = all_main_psets.end();
      for (; i != e; ++i) reg->insertParameterSet(*i);
    }

  } // namespace pset
} // namespace edm

