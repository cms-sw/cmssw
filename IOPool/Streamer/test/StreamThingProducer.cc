
/** \class ThingProducer
 *
 * \version   1st Version Apr. 6, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#if 1
#include "DataFormats/TestObjects/interface/StreamTestThing.h"
#include "DataFormats/TestObjects/interface/StreamTestTmpl.h"
typedef edmtestprod::StreamTestThing WriteThis;
#else
#include "FWCore/Integration/interface/IntArray.h"
typedef edmtestprod::IntArray WriteThis;
#endif

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <sstream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include <string>
#include <vector>

namespace edmtest_thing {
  class StreamThingProducer : public edm::global::EDProducer<> {
  public:
    explicit StreamThingProducer(edm::ParameterSet const& ps);

    ~StreamThingProducer() override;

    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    int size_;
    int inst_count_;
    std::vector<std::string> names_;
    int start_count_;

    bool apply_bit_mask_;
    unsigned int bit_mask_;
  };
}  // namespace edmtest_thing
using namespace edmtestprod;

namespace edmtest_thing {
  typedef StreamTestTmpl<OSimple> TestDbl;

  StreamThingProducer::StreamThingProducer(edm::ParameterSet const& ps)
      : size_(ps.getParameter<int>("array_size")),
        inst_count_(ps.getParameter<int>("instance_count")),
        start_count_(ps.getUntrackedParameter<int>("start_count", 0)),
        apply_bit_mask_(ps.getUntrackedParameter<bool>("apply_bit_mask", false)),
        bit_mask_(ps.getUntrackedParameter<uint32_t>("bit_mask", 0)) {
    for (int i = 0; i < inst_count_; ++i) {
      std::ostringstream ost;
      ost << (i + start_count_);
      names_.push_back(ost.str());
      produces<WriteThis>(ost.str());
    }

    // produces<TestDbl>();
    //produces<StreamTestSimple>();
    // produces<Pig>();
  }

  StreamThingProducer::~StreamThingProducer() {}

  // Functions that gets called by framework every event
  void StreamThingProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    for (int i = 0; i < inst_count_; ++i) {
      auto result = std::make_unique<WriteThis>(size_);

      // The purpose of this masking is to allow
      // some limited control of how much smaller these
      // vectors will get when compressed.  The more bits
      // are set to zero the more effect compression will have.
      if (apply_bit_mask_) {
        for (int j = 0; j < size_; ++j) {
          result->data_.at(j) &= bit_mask_;
        }
      }

      e.put(std::move(result), names_[i]);
    }

    //e.put(std::make_unique<TestDbl>());
    //e.put(std::make_unique<StreamTestSimple>());
    //e.put(std::make_unique<Pig>());
  }

  void StreamThingProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<int>("array_size");
    desc.add<int>("instance_count");
    desc.addUntracked<int>("start_count", 0);
    desc.addUntracked<bool>("apply_bit_mask", false);
    desc.addUntracked<uint32_t>("bit_mask", 0);
    descriptions.addDefault(desc);
  }
}  // namespace edmtest_thing

using edmtest_thing::StreamThingProducer;
DEFINE_FWK_MODULE(StreamThingProducer);
