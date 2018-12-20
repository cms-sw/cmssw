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

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventSelector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
  class EventForOutput;
  class TriggerResults;

  namespace detail {
    typedef edm::Handle<edm::TriggerResults> handle_t;

    class NamedEventSelector {
    public:
      NamedEventSelector(std::string const& n, EventSelector const& s, ConsumesCollector&& iC)
          : inputTag_("TriggerResults", "", n), token_(iC.consumes<TriggerResults>(inputTag_)), eventSelector_(s) {}

      bool match(TriggerResults const& product) { return eventSelector_.acceptEvent(product); }

      InputTag const& inputTag() const { return inputTag_; }

      EDGetTokenT<TriggerResults> const& token() const { return token_; }

    private:
      InputTag inputTag_;
      EDGetTokenT<TriggerResults> token_;
      EventSelector eventSelector_;
    };

    class TriggerResultsBasedEventSelector {
    public:
      TriggerResultsBasedEventSelector();
      typedef detail::handle_t handle_t;
      typedef std::vector<NamedEventSelector> selectors_t;
      typedef std::pair<std::string, std::string> parsed_path_spec_t;

      void setupDefault();

      void setup(std::vector<parsed_path_spec_t> const& path_specs,
                 std::vector<std::string> const& triggernames,
                 std::string const& process_name,
                 ConsumesCollector&& iC);

      bool wantEvent(EventForOutput const& e);

      unsigned int numberOfTokens() const { return selectors_.size(); }
      EDGetToken token(unsigned int index) const { return selectors_[index].token(); }

    private:
      selectors_t selectors_;
      bool wantAllEvents_;
    };

    /** Handles the final initialization of the TriggerResutsBasedEventSelector
     \return true if all events will be selected
     */
    bool configureEventSelector(edm::ParameterSet const& iPSet,
                                std::string const& iProcessName,
                                std::vector<std::string> const& iAllTriggerNames,
                                edm::detail::TriggerResultsBasedEventSelector& oSelector,
                                ConsumesCollector&& iC);
    /** Takes the user specified SelectEvents PSet and creates a new one
     which conforms to the canonical format required for provenance
     */
    ParameterSetID registerProperSelectionInfo(
        edm::ParameterSet const& iInitial,
        std::string const& iLabel,
        std::map<std::string, std::vector<std::pair<std::string, int> > > const& outputModulePathPositions,
        bool anyProductProduced);

  }  // namespace detail
}  // namespace edm

#endif
