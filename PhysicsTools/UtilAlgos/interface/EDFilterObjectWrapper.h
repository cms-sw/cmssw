#ifndef PhysicsTools_UtilAlgos_interface_EDFilterObjectWrapper_h
#define PhysicsTools_UtilAlgos_interface_EDFilterObjectWrapper_h

/**
  \class    EDFilterObjectWrapper EDFilterObjectWrapper.h "PhysicsTools/UtilAlgos/interface/EDFilterObjectWrapper.h"
  \brief    Wrapper class for a class of type BasicFilter to "convert" it into a full EDFilter

   This template class is a wrapper round classes of type Selector<T> and similar signature.
   It operates on container classes of type C which roughly satisfy std::vector template
   parameters.

   From this class the wrapper expects the following member functions:

   + a contructor with a const edm::ParameterSet& as input.
   + a filter function that operates on classes of type C::value_type

   the function is called within the wrapper. The wrapper translates the common class into
   a basic EDFilter as shown below:

   #include "PhysicsTools/UtilAlgos/interface/EDFilterObjectWrapper.h"
   #include "PhysicsTools/SelectorUtils/interface/PFJetIdSelectionFunctor.h"
   typedef edm::FilterWrapper<PFJetIdSelectionFunctor> PFJetIdFilter;

   #include "FWCore/Framework/interface/MakerMacros.h"
   DEFINE_FWK_MODULE(PFJetIdFilter);

   You can find this example in PhysicsTools/UtilAlgos/plugins/JetIDSelectionFunctorFilter.cc.
   In the first place the module will act as an EDProducer: it will put a new collection
   containing the selected objects into the event. Depending on the choice of parameter _filter_
   it will discard events for which the collection of selected events is empty. It will such also
   act as an EDFilter and thus is specified as such. The parameter _filter_ does not need to be
   specified.

   NOTE: in the current implementation this wrapper class does not support use of the EventSetup.
   If you want to make use of this feature we recommend you to start from an EDProducer from the
   very beginning and just to stay within the full framework.
*/

#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Common/interface/EventBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {

  template <class T, class C>
  class FilterObjectWrapper : public edm::stream::EDFilter<> {
  public:
    /// some convenient typedefs. Recall that C is a container class.
    typename C::iterator iterator;
    typename C::const_iterator const_iterator;

    /// default contructor. Declares the output (type "C") and the filter (of type T, operates on C::value_type)
    FilterObjectWrapper(const edm::ParameterSet& cfg) : src_(consumes<C>(cfg.getParameter<edm::InputTag>("src"))) {
      filter_ = std::shared_ptr<T>(new T(cfg.getParameter<edm::ParameterSet>("filterParams")));
      if (cfg.exists("filter")) {
        doFilter_ = cfg.getParameter<bool>("filter");
      } else {
        doFilter_ = false;
      }
      produces<C>();
    }
    /// default destructor
    ~FilterObjectWrapper() override {}
    /// everything which has to be done during the event loop. NOTE: We can't use the eventSetup in FWLite so ignore it
    bool filter(edm::Event& event, const edm::EventSetup& eventSetup) override {
      // create a collection of the objects to put into the event
      auto objsToPut = std::make_unique<C>();
      // get the handle to the objects in the event.
      edm::Handle<C> h_c;
      event.getByToken(src_, h_c);
      // loop through and add passing value_types to the output vector
      for (typename C::const_iterator ibegin = h_c->begin(), iend = h_c->end(), i = ibegin; i != iend; ++i) {
        if ((*filter_)(*i)) {
          objsToPut->push_back(*i);
        }
      }
      // put objs in the event
      bool pass = !objsToPut->empty();
      event.put(std::move(objsToPut));
      if (doFilter_)
        return pass;
      else
        return true;
    }

  protected:
    /// InputTag of the input source
    edm::EDGetTokenT<C> src_;
    /// shared pointer to analysis class of type BasicAnalyzer
    std::shared_ptr<T> filter_;
    /// whether or not to filter based on size
    bool doFilter_;
  };

}  // namespace edm

#endif
