#include <algorithm>
#include <iterator>
#include <sstream>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"

namespace edmtest 
{
  class PathAnalyzer : public edm::EDAnalyzer
  {
  public:

    explicit PathAnalyzer(edm::ParameterSet const&);
    virtual ~PathAnalyzer();
    
    virtual void analyze(edm::Event const&, edm::EventSetup const&);
    virtual void beginJob();
    virtual void endJob();

  private:
    void dumpTriggerNamesServiceInfo(char const* where);
  }; // class PathAnalyzer

  //--------------------------------------------------------------------
  //
  // Implementation details

  PathAnalyzer::PathAnalyzer(edm::ParameterSet const&) { }

  PathAnalyzer::~PathAnalyzer() {}

  void
  PathAnalyzer::analyze(edm::Event const&, edm::EventSetup const&)
  {
    dumpTriggerNamesServiceInfo("analyze");
  }

  void
  PathAnalyzer::beginJob()
  {
    dumpTriggerNamesServiceInfo("beginJob");
  }

  void
  PathAnalyzer::endJob()
  {
    dumpTriggerNamesServiceInfo("endJob");
  }

  void
  PathAnalyzer::dumpTriggerNamesServiceInfo(char const* where)
  {
    typedef edm::Service<edm::service::TriggerNamesService>  TNS;
    typedef std::vector<std::string> stringvec;

    TNS tns;
    std::ostringstream message;

    stringvec const& trigpaths = tns->getTrigPaths();
    message << "dumpTriggernamesServiceInfo called from PathAnalyzer::"
	    << where << '\n';
    message << "trigger paths are: ";

    edm::copy_all(trigpaths, std::ostream_iterator<std::string>(message, " "));
    message << '\n';

    for (stringvec::const_iterator i = trigpaths.begin(), e = trigpaths.end();
	 i != e;
	 ++i)
      {
	message << "path name: " << *i << " contains: ";
	edm::copy_all(tns->getTrigPathModules(*i), std::ostream_iterator<std::string>(message, " "));
	message << '\n';
      }

    message << "trigger ParameterSet:\n"
	    << tns->getTriggerPSet()
	    << '\n';

    edm::LogInfo("PathAnalyzer") << "TNS size: " << tns->size() 
				 << "\n"
				 << message.str()
				 << std::endl;
  }

} // namespace edmtest

using edmtest::PathAnalyzer;
DEFINE_FWK_MODULE(PathAnalyzer);
