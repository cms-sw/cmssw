
/*----------------------------------------------------------------------

 EDAnalyzer for testing History class and history tracking mechanism.

----------------------------------------------------------------------*/

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "DataFormats/Provenance/interface/History.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace edm;

namespace edmtest {

  class HistoryAnalyzer : public EDAnalyzer {
  public:

    explicit HistoryAnalyzer(const ParameterSet& params);
    void analyze(const Event& event, EventSetup const&);
    void endJob();

  private:
    int pass_;
    int eventCount_; 
    int expectedCount_;
    ParameterSetID emptyID_;
    ParameterSetID outputConfigID_;
  };

  HistoryAnalyzer::HistoryAnalyzer(const ParameterSet& params) :
    pass_(params.getParameter<int>("historySize")),
    eventCount_(0),
    expectedCount_(params.getParameter<int>("expectedCount")),
    emptyID_(),
    outputConfigID_()
  {
    ParameterSet emptyPset;
    emptyPset.registerIt();
    emptyID_ = emptyPset.trackedID();
    ParameterSet temp;
    typedef std::vector<std::string> vstring;
    vstring wanted_paths(1, "f55");
    temp.addParameter<std::vector<std::string> >("SelectEvents", wanted_paths);
    temp.registerIt();
    outputConfigID_ = temp.trackedID();
  }

  void
  HistoryAnalyzer::analyze(const Event& event, EventSetup const&) 
  {
    History const& h = event.history();
    assert(h.size() == static_cast<size_t>(pass_ - 1));

    assert(h.getEventSelectionID(0) == emptyID_);
    assert(h.getEventSelectionID(1) == outputConfigID_);
    assert(h.getEventSelectionID(2) == emptyID_);
    ++eventCount_;    
  }

  void
  HistoryAnalyzer::endJob()
  {
    std::cout << "Expected count is: " << expectedCount_ << std::endl;
    std::cout << "Event count is:    " << eventCount_ << std::endl;
    assert(eventCount_ == expectedCount_);
  }
}

using edmtest::HistoryAnalyzer;
DEFINE_FWK_MODULE(HistoryAnalyzer);
