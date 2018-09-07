#include <iostream>
#include <memory>
#include <string>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"

using namespace edm;

class CPEAccessTester : public edm::EDAnalyzer {
 public:
  CPEAccessTester(const edm::ParameterSet& pset) :
    conf_(pset)
  { }

  ~CPEAccessTester(){}

  void analyze(const edm::Event& event, const edm::EventSetup& setup) override {
    //
    // access the CPE
    //
    std::string cpeName = conf_.getParameter<std::string>("PixelCPE");   
    std::cout << "Asking for the CPE with name " << cpeName << std::endl;

    edm::ESHandle<PixelClusterParameterEstimator> theEstimator;
    setup.get<TkPixelCPERecord>().get(cpeName, theEstimator);
   
    auto & estimator = *theEstimator; 
    std::cout << "Got a " << typeid(estimator).name() << std::endl;
  }

private:
  edm::ParameterSet conf_;

};

// define this as a plug-in
DEFINE_FWK_MODULE(CPEAccessTester);
