// C++ headers
#include <memory>

// CMSSW headers
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

class FastTimerServiceClient : public edm::EDAnalyzer {
public:
  explicit FastTimerServiceClient(edm::ParameterSet const &);
  ~FastTimerServiceClient();

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  // this must be implemented
  void analyze(edm::Event const & event, edm::EventSetup const & setup);

  // a DQM harversting client should only implement the endJob method
  void endJob();
};

FastTimerServiceClient::FastTimerServiceClient(edm::ParameterSet const & config)
{
}


FastTimerServiceClient::~FastTimerServiceClient()
{
}

void
FastTimerServiceClient::analyze(edm::Event const & event, edm::EventSetup const & setup) 
{
}

void 
FastTimerServiceClient::endJob() 
{
}

void
FastTimerServiceClient::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FastTimerServiceClient);
