//#include <memory>

#include "FWCore/Framework/interface/EDAnalyzer.h"

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
class SiPixelFedCablingMapAnalyzer : public edm::EDAnalyzer {
   public:
      explicit SiPixelFedCablingMapAnalyzer( const edm::ParameterSet& ) {}
      ~SiPixelFedCablingMapAnalyzer();
      virtual void analyze( const edm::Event&, const edm::EventSetup& );
   private:
};


SiPixelFedCablingMapAnalyzer::~SiPixelFedCablingMapAnalyzer(){}

void SiPixelFedCablingMapAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

   std::cout << "====== SiPixelFedCablingMapAnalyzer" << std::endl;

   edm::ESHandle<SiPixelFedCablingMap> map;
   iSetup.get<SiPixelFedCablingMapRcd>().get(map);

   LogInfo(" got map, version: ") << map->version();
   auto tree = map->cablingTree();
   LogInfo("PRINT MAP:")<<tree->print(100);
   LogInfo("PRINT MAP, end:");

}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelFedCablingMapAnalyzer);
