// ----------------------------------------------------------------------
// $Id: ParameterSet.cc,v 1.16 2006/02/14 20:18:09 wmtan Exp $
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
			    edm::ParameterSet & rseult) const
  {
    return false;
  }

  void
  Registry::insertParameterSet(edm::ParameterSet const& p)
  {
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


} // namespace pset

