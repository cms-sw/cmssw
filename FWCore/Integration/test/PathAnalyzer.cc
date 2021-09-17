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

namespace edmtest {
  class PathAnalyzer : public edm::EDAnalyzer {
  public:
    explicit PathAnalyzer(edm::ParameterSet const&);
    virtual ~PathAnalyzer();

    virtual void analyze(edm::Event const&, edm::EventSetup const&);
    virtual void beginJob();
    virtual void endJob();

  private:
    void dumpTriggerNamesServiceInfo(char const* where);
  };  // class PathAnalyzer

  //--------------------------------------------------------------------
  //
  // Implementation details

  PathAnalyzer::PathAnalyzer(edm::ParameterSet const&) {}

  PathAnalyzer::~PathAnalyzer() {}

  void PathAnalyzer::analyze(edm::Event const&, edm::EventSetup const&) { dumpTriggerNamesServiceInfo("analyze"); }

  void PathAnalyzer::beginJob() { dumpTriggerNamesServiceInfo("beginJob"); }

  void PathAnalyzer::endJob() { dumpTriggerNamesServiceInfo("endJob"); }

  void PathAnalyzer::dumpTriggerNamesServiceInfo(char const* where) {
    edm::LogInfo("PathAnalyzer").log([&](auto& message) {
      edm::Service<edm::service::TriggerNamesService> tns;
      message << "TNS size: " << tns->size() << "\n";

      auto const& trigpaths = tns->getTrigPaths();
      message << "dumpTriggernamesServiceInfo called from PathAnalyzer::" << where << '\n';
      message << "trigger paths are:";
      for (auto const& p : trigpaths) {
        message << " " << p;
      }
      message << '\n';

      for (auto const& p : trigpaths) {
        message << "path name: " << p << " contains:";
        for (auto const& m : tns->getTrigPathModules(p)) {
          message << " " << m;
        }
        message << '\n';
      }

      message << "trigger ParameterSet:\n" << tns->getTriggerPSet() << '\n';
    });
  }

}  // namespace edmtest

using edmtest::PathAnalyzer;
DEFINE_FWK_MODULE(PathAnalyzer);
