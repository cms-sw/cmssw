/** \class edmtest::MissingDictionaryTestProducer
\author W. David Dagenhart, created 26 May 2016
*/

// Without manual intervention this simply tests the case where all
// the test dictionaries are defined, which is not very interesting.
// Its primary purpose is to be run manually where specific dictionaries
// have been removed from classes_def.xml and checking that the proper
// exceptions are thrown without having to generate this code from scratch.

#include "FWCore/Framework/interface/one/EDProducer.h"
#include "DataFormats/TestObjects/interface/MissingDictionaryTestObject.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/View.h"

#include <list>
#include <vector>

namespace edm {
  class EventSetup;
}

namespace edmtest {

  class MissingDictionaryTestProducer : public edm::one::EDProducer<> {
  public:
    explicit MissingDictionaryTestProducer(edm::ParameterSet const&);
    ~MissingDictionaryTestProducer() override;

    void produce(edm::Event&, edm::EventSetup const&) override;

  private:
    edm::EDGetTokenT<MissingDictionaryTestA> inputToken1_;
    edm::EDGetTokenT<std::vector<MissingDictionaryTestA> > inputToken2_;
    edm::EDGetTokenT<std::list<MissingDictionaryTestA> > inputToken3_;
  };

  MissingDictionaryTestProducer::MissingDictionaryTestProducer(edm::ParameterSet const& pset) {
    consumes<edm::View<MissingDictionaryTestA> >(pset.getParameter<edm::InputTag>("inputTag"));
    inputToken1_ = consumes<MissingDictionaryTestA>(pset.getParameter<edm::InputTag>("inputTag"));
    inputToken2_ = consumes<std::vector<MissingDictionaryTestA> >(pset.getParameter<edm::InputTag>("inputTag"));
    inputToken3_ = consumes<std::list<MissingDictionaryTestA> >(pset.getParameter<edm::InputTag>("inputTag"));

    produces<MissingDictionaryTestA>();
    produces<MissingDictionaryTestA>("anInstance");
    produces<std::vector<MissingDictionaryTestA> >();
    produces<std::vector<MissingDictionaryTestA> >("anInstance");
    produces<std::list<MissingDictionaryTestA> >();
  }

  MissingDictionaryTestProducer::~MissingDictionaryTestProducer() {}

  void MissingDictionaryTestProducer::produce(edm::Event& event, edm::EventSetup const&) {
    edm::Handle<MissingDictionaryTestA> h1;
    //event.getByToken(inputToken1_, h1);

    edm::Handle<std::vector<MissingDictionaryTestA> > h2;
    //event.getByToken(inputToken2_, h2);

    auto result1 = std::make_unique<MissingDictionaryTestA>();
    event.put(std::move(result1));

    auto result2 = std::make_unique<MissingDictionaryTestA>();
    event.put(std::move(result2), "anInstance");

    auto result3 = std::make_unique<std::vector<MissingDictionaryTestA> >();
    event.put(std::move(result3));

    auto result4 = std::make_unique<std::vector<MissingDictionaryTestA> >();
    event.put(std::move(result4), "anInstance");
  }
}  // namespace edmtest
using edmtest::MissingDictionaryTestProducer;
DEFINE_FWK_MODULE(MissingDictionaryTestProducer);
