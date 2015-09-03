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
#include <map>

#include "FWCore/Framework/interface/EventSelector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
  class EventPrincipal;
  class ModuleCallingContext;
  class ParameterSet;
  class RunPrincipal;
  class TriggerResults;

  namespace detail {
    typedef edm::Handle<edm::TriggerResults> handle_t;

    class NamedEventSelector {
    public:
      NamedEventSelector(std::string const& n, EventSelector const& s) :
	inputTag_("TriggerResults", "", n),
	eventSelector_(s)
      { }

      bool match(TriggerResults const& product) const {
	return eventSelector_.acceptEvent(product);
      }

      InputTag const& inputTag() const {
        return inputTag_;
      }

      void beginRun(ProcessHistory const& ph);

    private:
      InputTag            inputTag_;
      EventSelector       eventSelector_;
    };

    class TriggerResultsBasedEventSelector {
    public:
      TriggerResultsBasedEventSelector();
      typedef detail::handle_t                    handle_t;
      typedef std::vector<NamedEventSelector>     selectors_t;
      typedef std::pair<std::string, std::string> parsed_path_spec_t;

      void setupDefault(std::vector<std::string> const& pathNames);

      void setup(std::vector<parsed_path_spec_t> const& pathSpecs,
		 std::vector<std::string> const& pathNames,
                 const std::string& processName);

      bool wantEvent(EventPrincipal const& e, ModuleCallingContext const*) const;
      void beginRun(RunPrincipal const& rp);

    private:
      selectors_t selectors_;
    };

    /** Handles the final initialization of the TriggerResutsBasedEventSelector
     \return true if all events will be selected
     */
    bool configureEventSelector(edm::ParameterSet const& iPSet,
                                std::string const& iProcessName,
                                std::vector<std::string> const& iAllTriggerNames,
                                edm::detail::TriggerResultsBasedEventSelector& oSelector);
    /** Takes the user specified SelectEvents PSet and creates a new one
     which conforms to the canonical format required for provenance
     */
    ParameterSetID registerProperSelectionInfo(edm::ParameterSet const& iInitial,
                                               std::string const& iLabel,
                                               std::map<std::string, std::vector<std::pair<std::string, int> > > const& outputModulePathPositions,
                                               bool anyProductProduced);

  }
}

#endif
