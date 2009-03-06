#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
//#include "CondFormats/PhysicsPerformance/interface/PhysicsPerformancePayload.h"
#include "CondFormats/DataRecord/interface/BTagPerformanceWPRecord.h"
#include "CondFormats/BTauObjects/interface/BtagWorkingPoint.h"

class PhysicsPerformanceDBWriter_WP : public edm::EDAnalyzer
{
public:
  PhysicsPerformanceDBWriter_WP(const edm::ParameterSet&);
  virtual void beginJob(const edm::EventSetup&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&) {}
  virtual void endJob() {}
  ~PhysicsPerformanceDBWriter_WP() {}

private:
  std::string inputTxtFile;
};

PhysicsPerformanceDBWriter_WP::PhysicsPerformanceDBWriter_WP
  (const edm::ParameterSet& p)
{
}

void PhysicsPerformanceDBWriter_WP::beginJob(const edm::EventSetup&)
{
  //
  // create a fake Object, of type (for the moment) native
  //
  
  //PhysicsPerformancePayload*  p = new PhysicsPerformancePayload(stride, col, pl);
  BtagWorkingPoint*  p = new BtagWorkingPoint(9.4,"test_test");

  edm::Service<cond::service::PoolDBOutputService> s;
  if (s.isAvailable())
  {
    if (s->isNewTagRequest("BTagPerformanceWPRecord"))
    {
      s->createNewIOV<BtagWorkingPoint>(p,
                                          s->beginOfTime(),
                                          s->endOfTime(),
                                          "BTagPerformanceWPRecord");
    }
    else
    {

      s->appendSinceTime<BtagWorkingPoint>(p,
					   111,
                                             "BTagPerformanceWPRecord");
    }
  }
}

DEFINE_FWK_MODULE(PhysicsPerformanceDBWriter_WP);
