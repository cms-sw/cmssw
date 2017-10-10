#ifndef RecoTauTag_TauTagTools_CopyProducer_h
#define RecoTauTag_TauTagTools_CopyProducer_h

/*
 * Generic template class to take a View of C::value_type and produce an
 * output collection (type C) of clones.
 *
 * Author: Evan K. Friis (UC Davis)
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"

#include <algorithm>

namespace reco { namespace tautools {

template<typename Collection>
class CopyProducer : public edm::EDProducer {
  public:
    /// constructor from parameter set
    explicit CopyProducer( const edm::ParameterSet& pset)
        :src_(pset.template getParameter<edm::InputTag>("src")) {
          produces<Collection>();
        }
    /// destructor
    ~CopyProducer() override{};
    /// process an event
    void produce(edm::Event& evt, const edm::EventSetup& es) override {
      // Output collection
      auto coll = std::make_unique<Collection>();
      typedef edm::View<typename Collection::value_type> CView;
      // Get input
      edm::Handle<CView> input;
      evt.getByLabel(src_, input);
      // Reserve space
      coll->reserve(input->size());
      // Copy the input to the output
      std::copy(input->begin(), input->end(), std::back_inserter(*coll));
      evt.put(std::move(coll));
    }
  private:
    /// labels of the collection to be converted
    edm::InputTag src_;
};

}}

#endif
