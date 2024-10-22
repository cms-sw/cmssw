//#include <memory>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"

using namespace std;
using namespace edm;
using namespace sipixelobjects;

// class declaration
class SiPixelFedCablingMapAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelFedCablingMapAnalyzer(const edm::ParameterSet&) : fedCablingToken_(esConsumes()) {}
  ~SiPixelFedCablingMapAnalyzer();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> fedCablingToken_;
};

SiPixelFedCablingMapAnalyzer::~SiPixelFedCablingMapAnalyzer() = default;

void SiPixelFedCablingMapAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  LogPrint("SiPixelFedCablingMapAnalyzer") << "====== SiPixelFedCablingMapAnalyzer" << std::endl;

  const SiPixelFedCablingMap* map = &iSetup.getData(fedCablingToken_);

  LogInfo(" got map, version: ") << map->version();
  auto tree = map->cablingTree();
  LogInfo("SiPixelFedCablingMapAnalyzer") << "PRINT MAP:" << tree->print(100);
  LogInfo("SiPixelFedCablingMapAnalyzer") << "PRINT MAP, end:";
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelFedCablingMapAnalyzer);
