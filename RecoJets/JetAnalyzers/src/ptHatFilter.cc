// Name: ptHatFilter.cc
// Description:  Filter events in a range of Monte Carlo ptHat.
// Author: R. Harris
// Date:  28 - October - 2008
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <atomic>

class ptHatFilter : public edm::global::EDFilter<> {
public:
  ptHatFilter(const edm::ParameterSet&);
  void beginJob() override;
  bool filter(edm::StreamID, edm::Event& e, edm::EventSetup const& iSetup) const override;
  void endJob() override;

private:
  const edm::EDGetTokenT<double> token_;
  double ptHatLowerCut;
  double ptHatUpperCut;
  mutable std::atomic<int> totalEvents;
  mutable std::atomic<int> acceptedEvents;
};

using namespace edm;
using namespace reco;
using namespace std;
////////////////////////////////////////////////////////////////////////////////////////
ptHatFilter::ptHatFilter(edm::ParameterSet const& cfg) : token_{consumes(edm::InputTag("genEventScale"))} {
  ptHatLowerCut = cfg.getParameter<double>("ptHatLowerCut");
  ptHatUpperCut = cfg.getParameter<double>("ptHatUpperCut");
}
////////////////////////////////////////////////////////////////////////////////////////
void ptHatFilter::beginJob() {
  totalEvents = 0;
  acceptedEvents = 0;
}

////////////////////////////////////////////////////////////////////////////////////////
bool ptHatFilter::filter(edm::StreamID, edm::Event& evt, edm::EventSetup const& iSetup) const {
  bool result = false;
  totalEvents++;
  double pt_hat = evt.get(token_);
  if (pt_hat > ptHatLowerCut && pt_hat < ptHatUpperCut) {
    acceptedEvents++;
    result = true;
  }
  return result;
}
////////////////////////////////////////////////////////////////////////////////////////
void ptHatFilter::endJob() {
  std::cout << "Total Events = " << totalEvents << std::endl;
  std::cout << "Accepted Events = " << acceptedEvents << std::endl;
}
/////////// Register Modules ////////
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ptHatFilter);
