#ifndef CommonTools_UtilAlgos_EventSelectorAdapter_h
#define CommonTools_UtilAlgos_EventSelectorAdapter_h

/** \class EventSelectorAdapter
 *
 * Provide classes derrived from EventSelectorBase with an EDFilter interface
 *
 * \author Christian Veelken, UC Davis
 *
 * \version $Revision: 1.1 $
 *
 * $Id: EventSelectorAdapter.h,v 1.1 2009/03/03 13:07:26 llista Exp $
 *
 */

#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"

template <typename T>
struct SelectorFillDescriptions {
};

namespace impl {
  template <typename SFD>
  struct FillDescriptions {
    void operator()(edm::ConfigurationDescriptions& descriptions) {
      SFD::fillDescriptions(descriptions);
    }
  };
  struct DoDefault {
    void operator()(edm::ConfigurationDescriptions& descriptions) {
      // Same default as framework itself as I don't know how I could
      // avoid defining fillDescriptions() depending whether a "trait"
      // class template has fillDescriptions() or not.
      edm::fillDetails::DoFillAsUnknown<void> filler; // AFAICT DoFillAsUnknown wouldn't have to be a template, so passing just void here
      filler(descriptions);
    }
  };
}

template <typename T>
void selectorFillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  using Traits = SelectorFillDescriptions<T>;
  std::conditional_t<edm::fillDetails::has_fillDescriptions_function<Traits>::value,
                     impl::FillDescriptions<Traits>,
                     impl::DoDefault> fill_descriptions;
  fill_descriptions(descriptions);
}

template<typename T>
class EventSelectorAdapter : public edm::global::EDFilter<>
{
 public:
  // constructor
  explicit EventSelectorAdapter(const edm::ParameterSet& cfg) :
    eventSelector_( cfg, consumesCollector() ) {
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    selectorFillDescriptions<EventSelectorAdapter<T> >(descriptions);
  }

  // destructor
  ~EventSelectorAdapter() override {}

 private:
  bool filter(edm::StreamID, edm::Event& evt, const edm::EventSetup& es) const override { return eventSelector_(evt, es); }

  T eventSelector_;
};

#endif

