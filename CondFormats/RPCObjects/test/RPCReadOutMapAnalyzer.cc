//#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/DataRecord/interface/RPCReadOutMappingRcd.h"

using namespace std;
using namespace edm;

// class declaration
class RPCReadOutMapAnalyzer : public edm::EDAnalyzer {
   public:
      explicit RPCReadOutMapAnalyzer( const edm::ParameterSet& );
      ~RPCReadOutMapAnalyzer();
      virtual void analyze( const edm::Event&, const edm::EventSetup& );
   private:
};


RPCReadOutMapAnalyzer::RPCReadOutMapAnalyzer( const edm::ParameterSet& iConfig )
{
  ::putenv("CORAL_AUTH_USER konec");
  ::putenv("CORAL_AUTH_PASSWORD konecPass");
}


RPCReadOutMapAnalyzer::~RPCReadOutMapAnalyzer(){}

void RPCReadOutMapAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

   std::cout << "RPCReadOutMapAnalyzer - KUKU" << std::endl;

   edm::ESHandle<RPCReadOutMapping> map;
   iSetup.get<RPCReadOutMappingRcd>().get(map);
   cout << "version: " << map->version() << endl;

   pair<int,int> dccRange = map->dccNumberRange();
   cout <<" dcc range: " << dccRange.first <<" to "<<dccRange.second<<endl; 
   vector<const DccSpec *> dccs = map->dccList();
   typedef vector<const DccSpec *>::const_iterator IDCC;
   for (IDCC idcc = dccs.begin(); idcc != dccs.end(); idcc++) (**idcc).print(9);

    cout <<"--------"<<endl; 
   ChamberRawDataSpec chamber;
   chamber.dccId = 790;
   chamber.dccInputChannelNum = 6;
   chamber.tbLinkInputNum = 8;
   chamber.lbNumInLink = 2;
   const ChamberLocationSpec *location = map->location(chamber);
   if (location) location->print();

   
}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCReadOutMapAnalyzer)
