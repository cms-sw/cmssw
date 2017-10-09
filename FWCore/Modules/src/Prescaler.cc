
#include <atomic>
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  class Prescaler : public global::EDFilter<> {
  public:
    explicit Prescaler(ParameterSet const&);
    virtual ~Prescaler();

    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    virtual bool filter(StreamID, Event& e, EventSetup const& c) const override final;

  private:
    mutable std::atomic<int> count_;
    int n_; // accept one in n
    int offset_; // with offset, ie. sequence of events does not have to start at first event
  };

  Prescaler::Prescaler(ParameterSet const& ps) :
    count_(),
    n_(ps.getParameter<int>("prescaleFactor")),
    offset_(ps.getParameter<int>("prescaleOffset")) {
  }

  Prescaler::~Prescaler() {
  }

  bool Prescaler::filter(StreamID, Event&, EventSetup const&) const {
    //have to capture the value here since it could change by the time we do the comparision
    int count = ++count_;
    return count % n_ == offset_ ? true : false;
  }

  void
  Prescaler::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.add<int>("prescaleFactor")->setComment("Accept one event every N events");
    desc.add<int>("prescaleOffset")->setComment("The first event to accept should be the Mth one. Choose 'prescaleFactor'=1 to accept the first event from the source.");
    descriptions.add("preScaler", desc);
  }
}

using edm::Prescaler;
DEFINE_FWK_MODULE(Prescaler);
