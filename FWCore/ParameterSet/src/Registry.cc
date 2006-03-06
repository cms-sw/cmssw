// ----------------------------------------------------------------------
// $Id: Registry.cc,v 1.2 2006/02/27 15:32:33 paterno Exp $
//
// ----------------------------------------------------------------------

#include "boost/thread.hpp" // maybe need only thread/mutex.hpp?

#include "FWCore/ParameterSet/interface/Registry.h"

namespace
{
  boost::mutex registry_mutex;
}

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
  Registry::getParameterSet(edm::ParameterSetID const& id,
			    edm::ParameterSet & result) const
  {
    bool found;
    const_iterator i;
    {
      // This scope limits the lifetime of the lock to the shortest
      // required interval.
      boost::mutex::scoped_lock lock(registry_mutex);
      i = psets_.find(id);
      found = ( i != psets_.end() );
    }
    if (found) result = i->second;
    return found;
  }

  void
  Registry::insertParameterSet(edm::ParameterSet const& p)
  {
    edm::ParameterSet tracked_part(p.trackedPart());
    {
      // This scope limits the lifetime of the lock to the shortest
      // required interval.
      boost::mutex::scoped_lock lock(registry_mutex);
      psets_[p.id()] = tracked_part;
    }
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

  // Private member functions
  Registry::Registry() : psets_() { }

  Registry::~Registry() { }

  // Associated functions
    void loadAllNestedParameterSets(edm::ParameterSet const& main)
    {
      pset::Registry* reg = pset::Registry::instance();
      std::vector<edm::ParameterSet> all_main_psets;
      pset::explode(main, all_main_psets);
      std::vector<edm::ParameterSet>::const_iterator i = all_main_psets.begin();
      std::vector<edm::ParameterSet>::const_iterator e = all_main_psets.end();
      for ( ; i != e; ++i ) reg->insertParameterSet(*i);
    }


} // namespace pset

