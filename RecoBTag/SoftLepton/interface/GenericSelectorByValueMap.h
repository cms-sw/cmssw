#ifndef GenericSelectorByValueMap_h
#define GenericSelectorByValueMap_h

/** \class GenericSelectorByValueMap
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"

namespace edm {

  namespace details {

    // which type should be used in edm::ParameterSet::getParameter<_> to read a parameter compatible with T ?

    // most types can use themselves
    template <typename C>
    struct CompatibleConfigurationType {
      typedef C type;
    };

    // "float" is not allowed, as it conflicts with "double"
    template <>
    struct CompatibleConfigurationType<float> {
      typedef double type;
    };

  }  // namespace details

  template <typename T, typename C>
  class GenericSelectorByValueMap : public edm::global::EDProducer<> {
  public:
    explicit GenericSelectorByValueMap(edm::ParameterSet const& config);

  private:
    typedef T candidate_type;
    typedef C selection_type;
    typedef typename details::template CompatibleConfigurationType<selection_type>::type cut_type;

    void produce(edm::StreamID, edm::Event& event, edm::EventSetup const& setup) const override;

    edm::EDGetTokenT<edm::View<candidate_type>> token_electrons;
    edm::EDGetTokenT<edm::ValueMap<selection_type>> token_selection;

    cut_type m_cut;
  };

}  // namespace edm

//------------------------------------------------------------------------------

#include <vector>
#include <memory>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"

//------------------------------------------------------------------------------

namespace edm {

  template <typename T, typename C>
  GenericSelectorByValueMap<T, C>::GenericSelectorByValueMap(edm::ParameterSet const& config)
      : token_electrons(consumes<edm::View<candidate_type>>(config.getParameter<edm::InputTag>("input"))),
        token_selection(consumes<edm::ValueMap<selection_type>>(config.getParameter<edm::InputTag>("selection"))),
        m_cut(config.getParameter<cut_type>("cut")) {
    // register the product
    produces<edm::RefToBaseVector<candidate_type>>();
  }

  //------------------------------------------------------------------------------

  template <typename T, typename C>
  void GenericSelectorByValueMap<T, C>::produce(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
    auto candidates = std::make_unique<edm::RefToBaseVector<candidate_type>>();

    // read the collection of GsfElectrons from the Event
    edm::Handle<edm::View<candidate_type>> h_electrons;
    event.getByToken(token_electrons, h_electrons);
    edm::View<candidate_type> const& electrons = *h_electrons;

    // read the selection map from the Event
    edm::Handle<edm::ValueMap<selection_type>> h_selection;
    event.getByToken(token_selection, h_selection);
    edm::ValueMap<selection_type> const& selectionMap = *h_selection;

    for (unsigned int i = 0; i < electrons.size(); ++i) {
      edm::RefToBase<candidate_type> ptr = electrons.refAt(i);
      if (selectionMap[ptr] > m_cut)
        candidates->push_back(ptr);
    }

    // put the product in the event
    event.put(std::move(candidates));
  }

}  // namespace edm

#endif  // GenericSelectorByValueMap_h
