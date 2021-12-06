// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCChamber.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

//STL headers
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>

class RPCGeometryDump : public edm::one::EDAnalyzer<> {
public:
  explicit RPCGeometryDump(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  const bool verbose_;
  const edm::ESGetToken<RPCGeometry, MuonGeometryRecord> tokRPC_;
  const RPCGeometry* rpcGeometry_;
};

RPCGeometryDump::RPCGeometryDump(const edm::ParameterSet& iC)
    : verbose_(iC.getParameter<bool>("verbose")),
      tokRPC_{esConsumes<RPCGeometry, MuonGeometryRecord>(edm::ESInputTag{})} {}

void RPCGeometryDump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("verbose", false);
  descriptions.add("rpcGeometryDump", desc);
}

void RPCGeometryDump::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  rpcGeometry_ = &eventSetup.getData(tokRPC_);

  auto const& chambers = rpcGeometry_->chambers();
  edm::LogVerbatim("RPCGeometry") << "RPCGeometry found with " << chambers.size() << " chambers\n";

  for (unsigned int k1 = 0; k1 < chambers.size(); ++k1) {
    edm::LogVerbatim("RPCGeometry") << "\nChamber " << k1 << ":" << chambers[k1]->id() << " with "
                                    << chambers[k1]->nrolls() << " rolls";

    auto const& rolls = chambers[k1]->rolls();
    for (unsigned int k2 = 0; k2 < rolls.size(); ++k2) {
      edm::LogVerbatim("RPCGeometry") << "\nRoll " << k2 << ":" << rolls[k2]->id() << " Barrel|Endcap "
                                      << rolls[k2]->isBarrel() << ":" << rolls[k2]->isForward() << ":"
                                      << rolls[k2]->isIRPC() << " with " << rolls[k2]->nstrips() << " of pitch "
                                      << rolls[k2]->pitch();
      if (verbose_) {
        for (int k = 0; k < rolls[k2]->nstrips(); ++k)
          edm::LogVerbatim("RPCGeometry") << "Strip[" << k << "] " << rolls[k2]->centreOfStrip(k + 1);
      }
    }
  }
}

DEFINE_FWK_MODULE(RPCGeometryDump);
