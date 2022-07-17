#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class StopAfterNEvents : public edm::stream::EDFilter<> {
public:
  StopAfterNEvents(const edm::ParameterSet&);
  ~StopAfterNEvents() override = default;

private:
  bool filter(edm::Event&, edm::EventSetup const&) override;
  const int nMax_;
  int n_;
  const bool verbose_;
};

#include <iostream>

using namespace std;
using namespace edm;

StopAfterNEvents::StopAfterNEvents(const ParameterSet& pset)
    : nMax_(pset.getParameter<int>("maxEvents")), n_(0), verbose_(pset.getUntrackedParameter<bool>("verbose", false)) {}

bool StopAfterNEvents::filter(Event&, EventSetup const&) {
  if (n_ < 0)
    return true;
  n_++;
  bool ret = n_ <= nMax_;
  if (verbose_)
    edm::LogInfo("StopAfterNEvents") << ">>> filtering event" << n_ << "/" << nMax_ << "(" << (ret ? "true" : "false")
                                     << ")" << endl;
  return ret;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(StopAfterNEvents);
