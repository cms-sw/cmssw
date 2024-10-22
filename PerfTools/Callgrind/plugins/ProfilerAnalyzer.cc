// -*- C++ -*-
//
// Package:    PerfTools/Callgrind
// Class:      ProfilerAnalyzer
//
/**\class ProfilerAnalyzer ProfilerAnalyzer.cc PerfTools/Callgrind/plugins/ProfilerAnalyzer.cc

 Description: an Module that either start or stop profiling


 Implementation:
    ask the ProfileService to either start or stop profiling depeding on a Parameter Set
    

  \author Vincenzo Innocente

*/
//
// Original Author:  Andrea Rizzi
//         Created:  Thu Jan 18 10:34:18 CET 2007

// system include files

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PerfTools/Callgrind/interface/ProfilerService.h"

//
// class declaration
//
class ProfilerAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit ProfilerAnalyzer(const edm::ParameterSet&);
  ~ProfilerAnalyzer() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override = 0;
  void endJob() override;
};

class StartProfilerAnalyzer : public ProfilerAnalyzer {
public:
  explicit StartProfilerAnalyzer(const edm::ParameterSet& pset) : ProfilerAnalyzer(pset) {}
  ~StartProfilerAnalyzer() override {}

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
};

class StopProfilerAnalyzer : public ProfilerAnalyzer {
public:
  explicit StopProfilerAnalyzer(const edm::ParameterSet& pset) : ProfilerAnalyzer(pset) {}
  ~StopProfilerAnalyzer() override {}

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ProfilerAnalyzer::ProfilerAnalyzer(const edm::ParameterSet&) {}

ProfilerAnalyzer::~ProfilerAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void StartProfilerAnalyzer::analyze(const edm::Event&, const edm::EventSetup&) {
  edm::Service<ProfilerService>()->startInstrumentation();
}
void StopProfilerAnalyzer::analyze(const edm::Event&, const edm::EventSetup&) {
  edm::Service<ProfilerService>()->stopInstrumentation();
}

// ------------ method called once each job just before starting event loop  ------------
void ProfilerAnalyzer::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void ProfilerAnalyzer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(StartProfilerAnalyzer);
DEFINE_FWK_MODULE(StopProfilerAnalyzer);
