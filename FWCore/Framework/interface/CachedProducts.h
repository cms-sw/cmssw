#ifndef Framework_CachedProducts_h
#define Framework_CachedProducts_h

// -------------------------------------------------------------------
//
// CachedProducts: This class is used by OutputModule to interact with
// the TriggerResults objects upon which the decision to write out an
// event is made.
//
// -------------------------------------------------------------------
#include <string>
#include <utility>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSelector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Selector.h"

namespace edm
{
  namespace detail
  {
    typedef edm::Handle<edm::TriggerResults> handle_t;

    class NamedEventSelector
    {
    public:
      NamedEventSelector(std::string const& n, EventSelector const& s) :
	nameSelector_(n), 
	eventSelector_(s), 
	product_() 
      { }

      void fill(Event const& e)
      {
	e.get(nameSelector_, product_);
      }

      bool match()
      {
	return eventSelector_.acceptEvent(*product_);
      }

      handle_t product() const
      {
	return product_;
      }

      void clear()
      {
	product_ = handle_t();
      }
      
    private:
      ProcessNameSelector nameSelector_;
      EventSelector       eventSelector_;
      handle_t            product_;
    };


    class CachedProducts
    {
    public:
      CachedProducts();
      typedef detail::handle_t                    handle_t;
      typedef std::vector<NamedEventSelector>     selectors_t;
      typedef selectors_t::size_type              size_type;
      typedef std::pair<std::string, std::string> parsed_path_spec_t;

      void setupDefault(std::vector<std::string> const& triggernames);
      
      void setup(std::vector<parsed_path_spec_t> const& path_specs,
		 std::vector<std::string> const& triggernames,
                 const std::string& process_name);

      bool wantEvent(Event const& e);

      // Get all TriggerResults objects for the process names we're
      // interested in.
      size_type fill(Event const& ev);
      
      handle_t getOneTriggerResults(Event const& e);

      // Clear the cache
      void clear();

    private:
      typedef selectors_t::iterator iter;

      // Return the number of cached TriggerResult handles
      //size_type size() const { return numberFound_; }

      // If we have only one handle cached, return it; otherwise throw.
      handle_t returnOneHandleOrThrow();


      bool        fillDone_;
      size_type   numberFound_;
      selectors_t selectors_;
    };
  }
}

#endif
