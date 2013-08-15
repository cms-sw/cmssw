
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"

#include <string>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iterator>

using namespace edm;

namespace edm {
  class ModuleCallingContext;
}

namespace edmtest
{

  class TestResultAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit TestResultAnalyzer(edm::ParameterSet const&);
    virtual ~TestResultAnalyzer();

    virtual void analyze(edm::Event const& e, edm::EventSetup const& c) override;
    void endJob() override;

  private:
    int    passed_;
    int    failed_;
    bool   dump_;
    std::string name_;
    int    numbits_;
    std::string expected_pathname_; // if empty, we don't know
    std::string expected_modulelabel_; // if empty, we don't know
  };

  // -------

  class TestFilterModule : public edm::EDFilter
  {
  public:
    explicit TestFilterModule(edm::ParameterSet const&);
    virtual ~TestFilterModule();

    virtual bool filter(edm::Event& e, edm::EventSetup const& c) override;
    void endJob() override;

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

  private:
    virtual void write(edm::EventPrincipal const& e, ModuleCallingContext const*) override;
    virtual void writeLuminosityBlock(edm::LuminosityBlockPrincipal const&, ModuleCallingContext const*) override {}
    virtual void writeRun(edm::RunPrincipal const&, ModuleCallingContext const*) override {}
    virtual void endJob() override;

    std::string name_;
    int num_pass_;
    int total_;
  };

  // -----------------------------------------------------------------

  TestResultAnalyzer::TestResultAnalyzer(edm::ParameterSet const& ps):
    passed_(),
    failed_(),
    dump_(ps.getUntrackedParameter<bool>("dump",false)),
    name_(ps.getUntrackedParameter<std::string>("name","DEFAULT")),
    numbits_(ps.getUntrackedParameter<int>("numbits",-1)),
    expected_pathname_(ps.getUntrackedParameter<std::string>("pathname", "")),
    expected_modulelabel_(ps.getUntrackedParameter<std::string>("modlabel", ""))
  {
    consumesMany<edm::TriggerResults>();
  }
    
  TestResultAnalyzer::~TestResultAnalyzer()
  {
  }

  void TestResultAnalyzer::analyze(edm::Event const& e,edm::EventSetup const&)
  {
    typedef std::vector<edm::Handle<edm::TriggerResults> > Trig;
    Trig prod;
    e.getManyByType(prod);
    
    edm::CurrentProcessingContext const* cpc = currentContext();
    assert( cpc != 0 );
    assert( cpc->moduleDescription() != 0 );

    if ( !expected_pathname_.empty() )
      assert( expected_pathname_ == *(cpc->pathName()) );

    if ( !expected_modulelabel_.empty() )
      {
	assert(expected_modulelabel_ == *(cpc->moduleLabel()) );
      }

    if(prod.size() == 0) return;
    if(prod.size() > 1) {
      std::cerr << "More than one trigger result in the event, using first one"
	   << std::endl;
    }

    if (prod[0]->accept()) ++passed_; else ++failed_;

    if(numbits_ < 0) return;

    unsigned int numbits = numbits_;
    if(numbits != prod[0]->size()) {
      std::cerr << "TestResultAnalyzer named: " << name_
	   << " should have " << numbits
	   << ", got " << prod[0]->size() << " in TriggerResults\n";
      abort();
    }
  }

  void TestResultAnalyzer::endJob()
  {
    std::cerr << "TESTRESULTANALYZER " << name_ << ": "
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

  bool TestFilterModule::filter(edm::Event&, edm::EventSetup const&)
  {
    ++count_;
    assert( currentContext() != 0 );
    if(onlyOne_)
      return count_ % accept_rate_ ==0;
    else
      return count_ % 100 <= accept_rate_;
  }

  void TestFilterModule::endJob()
  {
    assert(currentContext() == 0);
  }

  // ---------

  SewerModule::SewerModule(edm::ParameterSet const& ps):
    edm::OutputModule(ps),
    name_(ps.getParameter<std::string>("name")),
    num_pass_(ps.getParameter<int>("shouldPass")),
    total_()
  {
  }
    
  SewerModule::~SewerModule()
  {
  }

  void SewerModule::write(edm::EventPrincipal const&, ModuleCallingContext const*)
  {
    ++total_;
    assert(currentContext() != 0);
  }

  void SewerModule::endJob()
  {
    assert( currentContext() == 0 );
    std::cerr << "SEWERMODULE " << name_ << ": should pass " << num_pass_
	 << ", did pass " << total_ << "\n";

    if(total_!=num_pass_)
      {
	std::cerr << "number passed should be " << num_pass_
	     << ", but got " << total_ << "\n";
	abort();
      }
  }
}

using edmtest::TestFilterModule;
using edmtest::TestResultAnalyzer;
using edmtest::SewerModule;


DEFINE_FWK_MODULE(TestFilterModule);
DEFINE_FWK_MODULE(TestResultAnalyzer);
DEFINE_FWK_MODULE(SewerModule);
