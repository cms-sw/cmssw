//#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"


using namespace std;
using namespace edm;
using namespace sipixelobjects;

// class declaration
class SiPixelFedCablingMapAnalyzer : public edm::EDAnalyzer {
   public:
      explicit SiPixelFedCablingMapAnalyzer( const edm::ParameterSet& );
      ~SiPixelFedCablingMapAnalyzer();
      virtual void analyze( const edm::Event&, const edm::EventSetup& );
   private:
};


SiPixelFedCablingMapAnalyzer::SiPixelFedCablingMapAnalyzer( const edm::ParameterSet& iConfig )
{
  ::putenv("CORAL_AUTH_USER konec");
  ::putenv("CORAL_AUTH_PASSWORD konecPass");
}


SiPixelFedCablingMapAnalyzer::~SiPixelFedCablingMapAnalyzer(){}

void SiPixelFedCablingMapAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

   std::cout << "====== SiPixelFedCablingMapAnalyzer" << std::endl;

   edm::ESHandle<SiPixelFedCablingMap> map;
   iSetup.get<SiPixelFedCablingMapRcd>().get(map);

   LogInfo(" got map, version: ") << map->version();
   LogInfo("PRINT MAP:")<<map->print(10);
   LogInfo("PRINT MAP, end:");
/*
   PixelFEDLink link = map->link();
   const PixelROC * roc = 0;
   roc = link.roc(0);
   cout << "Link ID: "<< link.id()<<" ROC xx: "<<roc->idInLink()<<endl;
   roc = link.roc(1);
   cout << "Link ID: "<< link.id()<<" ROC xx: "<<roc->idInLink()<<endl;
*/

}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelFedCablingMapAnalyzer)
