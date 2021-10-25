#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class SimpleEventFilter : public edm::stream::EDFilter<> {
public:
  SimpleEventFilter(const edm::ParameterSet &);
  ~SimpleEventFilter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  bool filter(edm::Event &, edm::EventSetup const &) override;
  int nEvent_;
  int nInterval_;
  bool verbose_;
};

//
// -- Constructor
//
SimpleEventFilter::SimpleEventFilter(const edm::ParameterSet &pset) {
  nInterval_ = pset.getUntrackedParameter<int>("EventsToSkip", 10);
  verbose_ = pset.getUntrackedParameter<bool>("DebugOn", false);
  nEvent_ = 0;
}

//
// -- Destructor
//
SimpleEventFilter::~SimpleEventFilter() = default;

bool SimpleEventFilter::filter(edm::Event &, edm::EventSetup const &) {
  nEvent_++;
  bool ret = true;
  if (nEvent_ % nInterval_ != 0)
    ret = false;
  //if (verbose_ && !ret)
  if (!ret)
    edm::LogInfo("SimpleEventFilter") << ">>> filtering event" << nEvent_ << std::endl;
  return ret;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SimpleEventFilter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("filters one event every N");
  desc.addUntracked<bool>("DebugOn", false)->setComment("activates debugging");
  desc.addUntracked<int>("EventsToSkip", 10)->setComment("events to skip");
  descriptions.add("_simpleEventFilter", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimpleEventFilter);
