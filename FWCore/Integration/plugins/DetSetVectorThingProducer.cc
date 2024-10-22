#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "ThingAlgorithm.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

namespace edmtest {
  class DetSetVectorThingProducer : public edm::global::EDProducer<> {
  public:
    explicit DetSetVectorThingProducer(edm::ParameterSet const& iConfig)
        : detSets_(iConfig.getParameter<std::vector<int>>("detSets")),
          nPerDetSet_(iConfig.getParameter<int>("nThings")),
          alg_(iConfig.getParameter<int>("offsetDelta"),
               nPerDetSet_ * detSets_.size(),
               iConfig.getParameter<bool>("grow")),
          evToken_(produces<edmNew::DetSetVector<Thing>>()) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<int>("offsetDelta", 0)
          ->setComment(
              "How much extra to increment the value used when creating Things for a new container. E.g. the last "
              "value "
              "used to create Thing from the previous event is incremented by 'offsetDelta' to compute the value to "
              "use "
              "of the first Thing created in the next Event.");
      desc.add<std::vector<int>>("detSets", std::vector<int>{1, 2, 3})->setComment("Vector of DetSet ids");
      desc.add<int>("nThings", 20)->setComment("How many Things to put in each DetSet.");
      desc.add<bool>("grow", false)
          ->setComment("If true, multiply 'nThings' by the value of offset for each run of the algorithm.");
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override {
      edmNew::DetSetVector<Thing> ret(2);

      ThingCollection tmp;
      alg_.run(tmp);
      assert(tmp.end() == tmp.begin() + nPerDetSet_ * detSets_.size());
      auto begin = tmp.begin();
      auto end = begin + nPerDetSet_;

      for (int detSetID : detSets_) {
        edmNew::DetSetVector<Thing>::FastFiller filler(ret, detSetID);
        std::copy(begin, end, std::back_inserter(filler));
        begin = end;
        std::advance(end, nPerDetSet_);
      }

      e.emplace(evToken_, std::move(ret));
    }

  private:
    std::vector<int> detSets_;
    unsigned int nPerDetSet_;
    ThingAlgorithm alg_;
    edm::EDPutTokenT<edmNew::DetSetVector<Thing>> evToken_;
  };

}  // namespace edmtest

using edmtest::DetSetVectorThingProducer;
DEFINE_FWK_MODULE(DetSetVectorThingProducer);
