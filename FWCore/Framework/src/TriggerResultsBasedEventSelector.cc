#include <algorithm>

#include "boost/bind.hpp"

#include "FWCore/Framework/interface/TriggerResultsBasedEventSelector.h"
#include "FWCore/Utilities/interface/Algorithms.h"

static const edm::TypeID s_TrigResultsType(typeid(edm::TriggerResults));

namespace edm 
{
  namespace detail
  {
    void NamedEventSelector::fill(EventPrincipal const& e) {
      edm::BasicHandle h = e.getByLabel(PRODUCT_TYPE,
                                        s_TrigResultsType,
                                        inputTag_,
                                        nullptr);
      convert_handle(h,product_);
    }
    
    typedef detail::NamedEventSelector NES;


    TriggerResultsBasedEventSelector::TriggerResultsBasedEventSelector() :
      fillDone_(false),
      numberFound_(0),
      selectors_()
    { }

    void
    TriggerResultsBasedEventSelector::setupDefault(std::vector<std::string> const& triggernames)
    {

      // Set up one NamedEventSelector, with default configuration
      std::vector<std::string> paths;
      EventSelector es(paths, triggernames);
      selectors_.emplace_back("", es);
      //selectors_.push_back(NES("", EventSelector("",triggernames)));
    }

    void
    TriggerResultsBasedEventSelector::setup(std::vector<parsed_path_spec_t> const& path_specs,
			  std::vector<std::string> const& triggernames,
                          const std::string& process_name)
    {
      // paths_for_process maps each PROCESS names to a sequence of
      // PATH names
      std::map<std::string, std::vector<std::string> > paths_for_process;
      for (std::vector<parsed_path_spec_t>::const_iterator 
	     i = path_specs.begin(), 
	     e = path_specs.end();
 	   i != e;
	   ++i)
	{
          // Default to current process if none specified
          if (i->second == "") {
            paths_for_process[process_name].push_back(i->first);
          }
          else {
            paths_for_process[i->second].push_back(i->first);
          }
	}
      // Now go through all the PROCESS names, and create a
      // NamedEventSelector for each.
      for (std::map<std::string, std::vector<std::string> >::const_iterator
	     i = paths_for_process.begin(),
	     e = paths_for_process.end();
	   i != e;
	   ++i)
	{
          // For the current process we know the trigger names
          // from the configuration file
          if (i->first == process_name) {
            selectors_.emplace_back(i->first, 
				    EventSelector(i->second, 
				    triggernames));
          }
          // For previous processes we do not know the trigger
          // names yet.
          else {
            selectors_.emplace_back(i->first, EventSelector(i->second));
          }
	}
    }

    TriggerResultsBasedEventSelector::handle_t
    TriggerResultsBasedEventSelector::getOneTriggerResults(EventPrincipal const& ev)
    {
      fill(ev);
      return returnOneHandleOrThrow();
    }

    bool
    TriggerResultsBasedEventSelector::wantEvent(EventPrincipal const& ev)
    {
      // We have to get all the TriggerResults objects before we test
      // any for a match, because we have to deal with the possibility
      // of multiple TriggerResults objects --- note that the presence
      // of more than one TriggerResults object in the event is
      // intended to lead to an exception throw *unless* either the
      // configuration has been set to match all events, or the
      // configuration is set to use specific process names.

      fill(ev);

      // Now we go through and see if anyone matches...
      iter i = selectors_.begin();
      iter e = selectors_.end();
      bool match_found = false;
      while (!match_found && (i!=e))
	{
	  match_found = i->match();
	  ++i;
	}
      return match_found;
    }
    
    TriggerResultsBasedEventSelector::handle_t
    TriggerResultsBasedEventSelector::returnOneHandleOrThrow()
    {
      switch (numberFound_)
	{
	case 0:
	  throw edm::Exception(edm::errors::ProductNotFound,
			       "TooFewProducts")
	    << "TriggerResultsBasedEventSelector::returnOneHandleOrThrow: "
	    << " too few products found, "
	    << "exepcted one, got zero\n";
	case 1:

	  break;
	default:
	  throw edm::Exception(edm::errors::ProductNotFound,
			       "TooManyMatches")
	    << "TriggerResultsBasedEventSelector::returnOneHandleOrThrow: "
	    << "too many products found, "
	    << "expected one, got " << numberFound_ << '\n';
	}
      return selectors_[0].product();
    }

    TriggerResultsBasedEventSelector::size_type
    TriggerResultsBasedEventSelector::fill(EventPrincipal const& ev)
    {
      if (!fillDone_)
	{
	  fillDone_ = true;
	  for (iter i = selectors_.begin(), e = selectors_.end(); 
	       i != e; ++i)
	    {
	      i->fill(ev);     // fill might throw...
	      ++numberFound_ ; // so numberFound_ might be less than expected
	    }
	}
      return numberFound_;
    }

    void
    TriggerResultsBasedEventSelector::clear()
    { 
      for_all(selectors_, boost::bind(&NamedEventSelector::clear, _1));
      fillDone_ = false;
      numberFound_ = 0;
    }


  } // namespace detail
} // namespace edm
