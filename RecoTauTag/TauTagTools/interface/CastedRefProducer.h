#ifndef RecoTauTag_TauTagTools_CastedRefProducer_h
#define RecoTauTag_TauTagTools_CastedRefProducer_h

/*
 * Taking a View<BaseType> as input, create a output collection of Refs to the
 * original collection with the desired type.
 *
 * Author: Evan K. Friis (UC Davis)
 *
 * Based on CommonTools/CandAlgos/interface/ShallowClone.h
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"

namespace reco { namespace tautools {

template<typename DerivedCollection, typename BaseType>
class CastedRefProducer : public edm::EDProducer {
  public:
    typedef typename edm::RefToBaseVector<BaseType> base_ref_vector;
    typedef typename base_ref_vector::value_type base_ref;
    typedef typename edm::Ref<DerivedCollection> derived_ref;
    typedef typename edm::RefVector<DerivedCollection> OutputCollection;
    /// constructor from parameter set
    explicit CastedRefProducer( const edm::ParameterSet& pset)
        :src_(pset.template getParameter<edm::InputTag>("src")) {
          produces<OutputCollection>();
        }
    /// destructor
    ~CastedRefProducer(){};
    /// process an event
    virtual void produce(edm::Event& evt, const edm::EventSetup& es) {
      // Output collection
      std::auto_ptr<OutputCollection> coll(new OutputCollection());
      // Get input
      edm::Handle<edm::View<BaseType> > input;
      evt.getByLabel(src_, input);
      // Get references to the base
      const base_ref_vector &baseRefs = input->refVector();
      for(size_t i = 0; i < baseRefs.size(); ++i) {
        // Cast the base class to the derived class
        base_ref base = baseRefs.at(i);
        derived_ref derived = base.template castTo<derived_ref>();
        coll->push_back(derived);
      }
      evt.put( coll );
    }
  private:
    /// labels of the collection to be converted
    edm::InputTag src_;
};

}}

#endif
