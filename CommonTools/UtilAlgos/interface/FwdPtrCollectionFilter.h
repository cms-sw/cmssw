#ifndef CommonTools_UtilAlgos_FwdPtrCollectionFilter_h
#define CommonTools_UtilAlgos_FwdPtrCollectionFilter_h

/**
   \class    edm::FwdPtrCollectionFilter FwdPtrCollectionFilter.h "CommonTools/UtilAlgos/interface/FwdPtrCollectionFilter.h"
   \brief    Selects a list of FwdPtr's to a product T (templated) that satisfy a method S(T) (templated). Can also handle input as View<T>.
   Can also have a factory class to create new instances of clones if desired.

   \author   Salvatore Rappoccio
*/

#include "CommonTools/UtilAlgos/interface/FwdPtrConversionFactory.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <algorithm>
#include <vector>

namespace edm {

  template <class T, class S, class H = ProductFromFwdPtrFactory<T>>
  class FwdPtrCollectionFilter : public edm::stream::EDFilter<> {
  public :

    explicit FwdPtrCollectionFilter(edm::ParameterSet const& ps) :
      srcToken_{consumes<std::vector<edm::FwdPtr<T>>>(ps.getParameter<edm::InputTag>("src"))},
      srcViewToken_{mayConsume<edm::View<T>>(ps.getParameter<edm::InputTag>("src"))},
      filter_{ps.exists("filter") ? ps.getParameter<bool>("filter") : false},
      makeClones_{ps.exists("makeClones") ? ps.getParameter<bool>("makeClones") : false},
      selector_{ps}
    {
      produces<std::vector<edm::FwdPtr<T>>>();
      if (makeClones_) {
        produces<std::vector<T>>();
      }
    }

    bool filter(edm::Event& iEvent, edm::EventSetup const& iSetup) override
    {
      auto pOutput = std::make_unique<std::vector<edm::FwdPtr<T>>>();

      // First try to access as a vector<FwdPtr<T>>; otherwise try as a View<T>.
      edm::Handle<std::vector<edm::FwdPtr<T>>> hSrcAsFwdPtr;
      if (iEvent.getByToken(srcToken_, hSrcAsFwdPtr)) {
        std::copy_if(std::cbegin(*hSrcAsFwdPtr), std::cend(*hSrcAsFwdPtr), std::back_inserter(*pOutput),
                     [this](auto const ptr) { return selector_(*ptr); });
      }
      else {
        edm::Handle<edm::View<T>> hSrcAsView;
        iEvent.getByToken(srcViewToken_, hSrcAsView);
        for (auto ibegin = std::cbegin(*hSrcAsView), iend = std::cend(*hSrcAsView), i = ibegin; i!= iend; ++i) {
          if (selector_(*i)) {
            auto const p = hSrcAsView->ptrAt(i-ibegin);
            pOutput->emplace_back(p,p);
          }
        }
      }

      // Must form pClones *before* std::move(pOutput) has been called.
      if (makeClones_) {
        H factory;
        auto pClones = std::make_unique<std::vector<T>>();
        std::transform(std::cbegin(*pOutput), std::cend(*pOutput), std::back_inserter(*pClones),
                       [&factory](auto ptr){ return factory(ptr); });
        iEvent.put(std::move(pClones));
      }

      bool const pass {!pOutput->empty()};
      iEvent.put(std::move(pOutput));

      return filter_ ? pass : true;
    }

  protected :
    edm::EDGetTokenT<std::vector<edm::FwdPtr<T>>> const srcToken_;
    edm::EDGetTokenT<edm::View<T>> const srcViewToken_;
    bool const filter_;
    bool const makeClones_;
    S selector_;
  };
}

#endif
