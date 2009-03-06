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
#include "CondFormats/DataRecord/interface/BTagPerformancePayloadRecord.h"
#include "CondFormats/BTauObjects/interface/BtagPerformancePayloadFromTableEtaJetEt.h"

class PhysicsPerformanceDBWriter : public edm::EDAnalyzer
{
public:
  PhysicsPerformanceDBWriter(const edm::ParameterSet&);
  virtual void beginJob(const edm::EventSetup&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&) {}
  virtual void endJob() {}
  ~PhysicsPerformanceDBWriter() {}

private:
  std::string inputTxtFile;
};

PhysicsPerformanceDBWriter::PhysicsPerformanceDBWriter
  (const edm::ParameterSet& p)
{
}

void PhysicsPerformanceDBWriter::beginJob(const edm::EventSetup&)
{
  //
  // create a fake Object, of type (for the moment) native
  //

  std::string col= "etamin etamax etmin etmax beff berr ceff cerr leff err";
  std::vector<float> pl;
  int stride = 10;
  // first point
  pl.push_back(0); pl.push_back(1.7);
  pl.push_back(30); pl.push_back(50);
  pl.push_back(.7); pl.push_back(0.07);
  pl.push_back(.1); pl.push_back(0.01);
  pl.push_back(.05); pl.push_back(0.01);
  // second point
  pl.push_back(1.7); pl.push_back(2.4);
  pl.push_back(30); pl.push_back(50);
  pl.push_back(.6); pl.push_back(0.07);
  pl.push_back(.05); pl.push_back(0.01);
  pl.push_back(.02); pl.push_back(0.01);
  // third point
  pl.push_back(0); pl.push_back(2.4);
  pl.push_back(50); pl.push_back(100);
  pl.push_back(.5); pl.push_back(0.07);
  pl.push_back(.1); pl.push_back(0.01);
  pl.push_back(.01); pl.push_back(0.01);
  
  //PhysicsPerformancePayload*  p = new PhysicsPerformancePayload(stride, col, pl);
BtagPerformancePayload*  p = new BtagPerformancePayloadFromTableEtaJetEt(stride, col, pl);

  edm::Service<cond::service::PoolDBOutputService> s;
  if (s.isAvailable())
  {
    if (s->isNewTagRequest("BTagPerformancePayloadRecord"))
    {
      s->createNewIOV<BtagPerformancePayload>(p,
                                          s->beginOfTime(),
                                          s->endOfTime(),
                                          "BTagPerformancePayloadRecord");
    }
    else
    {

      s->appendSinceTime<BtagPerformancePayload>(p,
                                             s->currentTime(),
                                             "BTagPerformancePayloadRecord");
    }
  }
}

DEFINE_FWK_MODULE(PhysicsPerformanceDBWriter);
