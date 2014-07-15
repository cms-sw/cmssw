#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"


#include "DataFormats/Common/interface/Handle.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"

#include <iostream>
#include <string>

using namespace edm;

class CPEAccessTester : public edm::EDAnalyzer {
 public:
  CPEAccessTester(const edm::ParameterSet& pset) {conf_ = pset;}

  ~CPEAccessTester(){}

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){
    //
    // access the CPE
    //
    using namespace std;
    std::string cpeName = conf_.getParameter<std::string>("PixelCPE");   
    cout <<" Asking for the CPE with name "<<cpeName<<endl;

    edm::ESHandle<PixelClusterParameterEstimator> theEstimator;
    setup.get<TkPixelCPERecord>().get(cpeName,theEstimator);
    
    cout <<" Got a "<<typeid(*theEstimator).name()<<endl;
    
  }
private:
  edm::ParameterSet conf_;
};
//define this as a plug-in
DEFINE_FWK_MODULE(CPEAccessTester);
