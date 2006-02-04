
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Log.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/TriggerResults.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <string>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iterator>

using namespace std;
using namespace edm;

namespace edmtest
{

  class TestResultAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit TestResultAnalyzer(edm::ParameterSet const&);
    virtual ~TestResultAnalyzer();

    virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
    void endJob();

  private:
    int passed_;
    int failed_;
    bool dump_;
    string name_;
  };

  // -------

  class TestFilterModule : public edm::EDFilter
  {
  public:
    explicit TestFilterModule(edm::ParameterSet const&);
    virtual ~TestFilterModule();

    virtual bool filter(edm::Event const& e, edm::EventSetup const& c);
    void endJob();

  private:
    int count_;
    int accept_rate_; // how many out of 100 will be accepted?
    bool onlyOne_;
  };

  // -------

  class SewerModule : public edm::OutputModule
  {
  public:
    explicit SewerModule(edm::ParameterSet const&);
    virtual ~SewerModule();

    virtual void write(edm::EventPrincipal const& e);
    void endJob();

  private:
    string name_;
    int num_pass_;
    int total_;
  };

  // -----------------------------------------------------------------

  TestResultAnalyzer::TestResultAnalyzer(edm::ParameterSet const& ps):
    passed_(),
    failed_(),
    dump_(ps.getUntrackedParameter<bool>("dump",false)),
    name_(ps.getUntrackedParameter<string>("name","DEFAULT"))
  {
  }
    
  TestResultAnalyzer::~TestResultAnalyzer()
  {
  }

  void TestResultAnalyzer::analyze(edm::Event const& e,edm::EventSetup const&)
  {
    typedef std::vector<edm::Handle<edm::TriggerResults> > Trig;
    Trig prod;
    e.getManyByType(prod);

    if(prod.size() == 0) return;
    if(prod.size() > 1)
      {
	cerr << "More than one trigger result in the event, using first one"
	     << endl;
      }

    if(prod[0]->pass()) ++passed_;
    else if(prod[0]->fail()) ++failed_;
  }

  void TestResultAnalyzer::endJob()
  {
    cerr << "TESTRESULTANALYZER " << name_ << ": "
	 << "passed=" << passed_ << " failed=" << failed_ << "\n";
  }

  // ---------

  TestFilterModule::TestFilterModule(edm::ParameterSet const& ps):
    count_(),
    accept_rate_(ps.getUntrackedParameter<int>("acceptValue",1)),
    onlyOne_(ps.getUntrackedParameter<bool>("onlyOne",false))
  {
  }
    
  TestFilterModule::~TestFilterModule()
  {
  }

  bool TestFilterModule::filter(edm::Event const& e,edm::EventSetup const&)
  {
    ++count_;
    if(onlyOne_)
      return count_%accept_rate_ ==0 ? true : false;
    else
      return count_%100 <= accept_rate_ ? true : false;
  }

  void TestFilterModule::endJob()
  {
  }

  // ---------

  SewerModule::SewerModule(edm::ParameterSet const& ps):
    edm::OutputModule(ps),
    name_(ps.getParameter<string>("name")),
    num_pass_(ps.getParameter<int>("shouldPass")),
    total_()
  {
  }
    
  SewerModule::~SewerModule()
  {
  }

  void SewerModule::write(edm::EventPrincipal const& e)
  {
    ++total_;
  }

  void SewerModule::endJob()
  {
    cerr << "SEWERMODULE " << name_ << ": should pass " << num_pass_
	 << ", did pass " << total_ << "\n";

    if(total_!=num_pass_) abort();
  }
}

using edmtest::TestFilterModule;
using edmtest::TestResultAnalyzer;
using edmtest::SewerModule;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TestFilterModule)
DEFINE_ANOTHER_FWK_MODULE(TestResultAnalyzer)
DEFINE_ANOTHER_FWK_MODULE(SewerModule)
