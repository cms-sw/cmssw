// -*- C++ -*-
//
// Package:    Profiler
// Class:      Profiler
//
/**\class Profiler Profiler.cc PerfTools/Callgrind/plugins/Profiler.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Rizzi
//         Created:  Thu Jan 18 10:34:18 CET 2007
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "valgrind/callgrind.h"
//
// class declaration
//
#include <iostream>
using namespace std;
namespace callgrind {
  class Profiler : public edm::one::EDAnalyzer<> {
  public:
    explicit Profiler(const edm::ParameterSet&);
    ~Profiler() override;

  private:
    void beginJob() override;
    void analyze(const edm::Event&, const edm::EventSetup&) override;
    void endJob() override;

    // ----------member data ---------------------------
    int m_firstEvent;
    int m_lastEvent;
    int m_action;
    int m_evtCount;
  };
}  // namespace callgrind
using namespace callgrind;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
Profiler::Profiler(const edm::ParameterSet& parameters) {
  //now do what ever initialization is needed
  m_firstEvent = parameters.getParameter<int>("firstEvent");
  m_lastEvent = parameters.getParameter<int>("lastEvent");
  m_action = parameters.getParameter<int>("action");
  m_evtCount = 0;
}

Profiler::~Profiler() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
void Profiler::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  m_evtCount++;
  if (m_evtCount >= m_firstEvent && (m_evtCount <= m_lastEvent || m_lastEvent == -1)) {
    switch (m_action) {
      case 0:
        CALLGRIND_STOP_INSTRUMENTATION;
        cout << "Stop Instr" << endl;
        break;
      case 1:
        CALLGRIND_START_INSTRUMENTATION;
        CALLGRIND_DUMP_STATS;
        cout << "Start Instr" << endl;
        break;
      case 2:
        CALLGRIND_DUMP_STATS;
        cout << "Dump stat" << endl;
        break;
    }
  }
}
#pragma GCC diagnostic pop

// ------------ method called once each job just before starting event loop  ------------
void Profiler::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void Profiler::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(Profiler);
