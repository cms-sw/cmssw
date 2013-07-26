#ifndef FWCore_Framework_TriggerResultsBasedEventSelector_h
#define FWCore_Framework_TriggerResultsBasedEventSelector_h

// -------------------------------------------------------------------
//
// TriggerResultsBasedEventSelector: This class is used by OutputModule to interact with
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
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm
{
  namespace detail
  {
    typedef edm::Handle<edm::TriggerResults> handle_t;

    class NamedEventSelector
    {
    public:
      NamedEventSelector(std::string const& n, EventSelector const& s) :
	inputTag_("TriggerResults", "", n), 
	eventSelector_(s), 
	product_() 
      { }

      void fill(Event const& e)
      {
	e.getByLabel(inputTag_, product_);
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
      InputTag            inputTag_;
      EventSelector       eventSelector_;
      handle_t            product_;
    };


    class TriggerResultsBasedEventSelector
    {
    public:
      TriggerResultsBasedEventSelector();
      typedef detail::handle_t                    handle_t;
      typedef std::vector<NamedEventSelector>     selectors_t;
      typedef selectors_t::size_type              size_type;
      typedef std::pair<std::string, std::string> parsed_path_spec_t;

      void setupDefault(std::vector<std::string> const& triggernames);
      
      void setup(std::vector<parsed_path_spec_t> const& path_specs,
		 std::vector<std::string> const& triggernames,
                 const std::string& process_name);

      bool wantEvent(Event const& e);

      handle_t getOneTriggerResults(Event const& e);

      // Clear the cache
      void clear();

    private:
      typedef selectors_t::iterator iter;

      // Get all TriggerResults objects for the process names we're
      // interested in.
      size_type fill(Event const& ev);
      
      // If we have only one handle cached, return it; otherwise throw.
      handle_t returnOneHandleOrThrow();


      bool        fillDone_;
      size_type   numberFound_;
      selectors_t selectors_;
    };
  }
}

#endif
