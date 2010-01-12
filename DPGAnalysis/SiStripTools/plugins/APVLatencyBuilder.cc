// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/DBCommon/interface/Time.h"

#include <vector>
#include "DPGAnalysis/SiStripTools/interface/APVLatency.h"

class APVLatencyBuilder : public edm::EDAnalyzer {

 public:

  explicit APVLatencyBuilder( const edm::ParameterSet& iConfig);

  ~APVLatencyBuilder(){};

private:
  virtual void beginJob();
  virtual void endJob();
  virtual void analyze(const edm::Event& , const edm::EventSetup& ){};
  virtual void endRun(const edm::Run& iRun, const edm::EventSetup&);

  std::vector<edm::ParameterSet> _latIOVs;
};

APVLatencyBuilder::APVLatencyBuilder( const edm::ParameterSet& iConfig ):
  _latIOVs(iConfig.getParameter<std::vector<edm::ParameterSet> >("latencyIOVs"))
{}

void APVLatencyBuilder::beginJob() {

  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  // fill the IOV gap at the beginning of the history

  edm::LogInfo("FirstIOV") << "First dummy IOV is needed";
  if( mydbservice.isAvailable() ){
    if(mydbservice->isNewTagRequest("APVLatencyRcd")) {
      APVLatency* dummyobj = new APVLatency();
      mydbservice->createNewIOV<APVLatency>(dummyobj,mydbservice->beginOfTime(),mydbservice->endOfTime(),"APVLatencyRcd");
    }
    else {
      edm::LogInfo("FirstIOV") << "...NO! First dummy IOV is NOT needed: db file is not empty";
    }

    // Loop on the PSet and add new IOVs

    for(std::vector<edm::ParameterSet>::const_iterator ps=_latIOVs.begin();ps!=_latIOVs.end();ps++) {

      int firstrun = ps->getParameter<int>("runNumber");
      int latency =  ps->getParameter<int>("latency");
      APVLatency *apvlat = new APVLatency();
      apvlat->put(latency);

      edm::LogInfo("IOVAdded") << " adding new IOV since run " << firstrun 
					    << " with APV latency " << apvlat->get();

      mydbservice->appendSinceTime<APVLatency>(apvlat,firstrun,"APVLatencyRcd");
      
    }

  }
  else{
    edm::LogError("NoService")<<"Service is unavailable";
  }
}

void APVLatencyBuilder::endRun(const edm::Run& iRun, const edm::EventSetup&) {}

void APVLatencyBuilder::endJob() {}
//define this as a plug-in
DEFINE_FWK_MODULE(APVLatencyBuilder);
