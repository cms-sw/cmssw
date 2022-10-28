// system includes
#include <iostream>
#include <memory>
#include <string>

// user includes
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"

using namespace edm;

class CPEAccessTester : public edm::one::EDAnalyzer<> {
public:
  CPEAccessTester(const edm::ParameterSet& pset)
      : cpeName_(pset.getParameter<std::string>("PixelCPE")), cpeToken_(esConsumes(edm::ESInputTag("", cpeName_))) {
    edm::LogPrint("CPEAccessTester") << "Asking for the CPE with name " << cpeName_;
  }

  ~CPEAccessTester() = default;
  void analyze(const edm::Event& event, const edm::EventSetup& setup) override {
    //
    // access the CPE
    //
    edm::ESHandle<PixelClusterParameterEstimator> theEstimator = setup.getHandle(cpeToken_);
    auto& estimator = *theEstimator;
    edm::LogPrint("CPEAccessTester") << "Got a " << typeid(estimator).name();
  }

private:
  const std::string cpeName_;
  const edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> cpeToken_;
};

// define this as a plug-in
DEFINE_FWK_MODULE(CPEAccessTester);
