#include "FWCore/Framework/interface/global/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "DataFormats/TestObjects/interface/TrackOfThings.h"
#include "DataFormats/Provenance/interface/ProductID.h"

namespace {
  template <typename F>
  void requireExceptionCategory(edm::errors::ErrorCodes code, F&& function) {
    bool threwException = false;
    try {
      function();
    } catch (edm::Exception& ex) {
      if (ex.categoryCode() != code) {
        throw cms::Exception("TestFailure")
            << "Got edm::Exception with category code " << ex.categoryCode() << " expected " << code << " message:\n"
            << ex.explainSelf();
      }
      threwException = true;
    }
    if (not threwException) {
      throw cms::Exception("TestFailure") << "Expected edm::Exception, but was not thrown";
    }
  }
}  // namespace

namespace edmtest {
  class ThinnedRefFromTestAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit ThinnedRefFromTestAnalyzer(edm::ParameterSet const& pset);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    void analyze(edm::StreamID streamID, edm::Event const& e, edm::EventSetup const& c) const override;

  private:
    const edm::EDGetTokenT<ThingCollection> parentToken_;
    const edm::EDGetTokenT<ThingCollection> thinnedToken_;
    const edm::EDGetTokenT<ThingCollection> unrelatedToken_;
    const edm::EDGetTokenT<TrackOfThingsCollection> trackToken_;
  };

  ThinnedRefFromTestAnalyzer::ThinnedRefFromTestAnalyzer(edm::ParameterSet const& pset)
      : parentToken_{consumes<ThingCollection>(pset.getParameter<edm::InputTag>("parentTag"))},
        thinnedToken_{consumes<ThingCollection>(pset.getParameter<edm::InputTag>("thinnedTag"))},
        unrelatedToken_{consumes<ThingCollection>(pset.getParameter<edm::InputTag>("unrelatedTag"))},
        trackToken_{consumes<TrackOfThingsCollection>(pset.getParameter<edm::InputTag>("trackTag"))} {}

  void ThinnedRefFromTestAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("parentTag");
    desc.add<edm::InputTag>("thinnedTag");
    desc.add<edm::InputTag>("unrelatedTag");
    desc.add<edm::InputTag>("trackTag");
    descriptions.addDefault(desc);
  }

  void ThinnedRefFromTestAnalyzer::analyze(edm::StreamID streamID,
                                           edm::Event const& event,
                                           edm::EventSetup const& c) const {
    auto parentHandle = event.getHandle(parentToken_);
    auto thinnedHandle = event.getHandle(thinnedToken_);
    auto unrelatedHandle = event.getHandle(unrelatedToken_);
    auto const& trackCollection = event.get(trackToken_);

    edm::RefProd parentRefProd{parentHandle};
    edm::RefProd thinnedRefProd{thinnedHandle};
    edm::RefProd unrelatedRefProd{unrelatedHandle};

    requireExceptionCategory(edm::errors::InvalidReference, [&]() {
      auto invalidParentRef = edm::thinnedRefFrom(edm::Ref(unrelatedHandle, 0), thinnedRefProd, event.productGetter());
    });
    if (auto invalidParentRef =
            edm::tryThinnedRefFrom(edm::Ref(unrelatedHandle, 0), thinnedRefProd, event.productGetter());
        invalidParentRef.isNonnull()) {
      throw cms::Exception("TestFailure") << "Expected to get Null Ref when passing in a Ref to unrelated parent "
                                             "collection, got a non-null Ref instead";
    }

    for (auto const& track : trackCollection) {
      auto parentRef1 = edm::thinnedRefFrom(track.ref1, parentRefProd, event.productGetter());
      if (parentRef1.id() != track.ref1.id()) {
        throw cms::Exception("TestFailure")
            << "Ref1-to-parent ProductID " << parentRef1.id() << " expected " << track.ref1.id();
      }
      if (parentRef1.key() != track.ref1.key()) {
        throw cms::Exception("TestFailure")
            << "Ref1-to-parent key " << parentRef1.key() << " expected " << track.ref1.key();
      }

      auto thinnedRef1 = edm::thinnedRefFrom(track.ref1, thinnedRefProd, event.productGetter());
      if (thinnedRef1.id() != thinnedRefProd.id()) {
        throw cms::Exception("TestFailure")
            << "Ref1-to-thinned ProductID " << thinnedRef1.id() << " expected " << thinnedRefProd.id();
      }

      requireExceptionCategory(edm::errors::InvalidReference, [&]() {
        auto invalidThinnedRef = edm::thinnedRefFrom(track.ref1, unrelatedRefProd, event.productGetter());
      });
      if (auto invalidThinnedRef = edm::tryThinnedRefFrom(track.ref1, unrelatedRefProd, event.productGetter());
          invalidThinnedRef.isNonnull()) {
        throw cms::Exception("TestFailure") << "Expected to get Null Ref when passing in a RefProd to unrelated "
                                               "thinned collection, got a non-null Ref instead";
      }
    }
  }
}  // namespace edmtest

using edmtest::ThinnedRefFromTestAnalyzer;
DEFINE_FWK_MODULE(ThinnedRefFromTestAnalyzer);
