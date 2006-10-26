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

#include "DataFormats/MuonDetId/interface/RPCDetId.h"

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

   std::cout << "====== RPCReadOutMapAnalyzer" << std::endl;

   edm::ESHandle<RPCReadOutMapping> map;
   iSetup.get<RPCReadOutMappingRcd>().get(map);
   cout << "version: " << map->version() << endl;

   pair<int,int> dccRange = map->dccNumberRange();
   cout <<" dcc range: " << dccRange.first <<" to "<<dccRange.second<<endl; 
   vector<const DccSpec *> dccs = map->dccList();
   typedef vector<const DccSpec *>::const_iterator IDCC;
   for (IDCC idcc = dccs.begin(); idcc != dccs.end(); idcc++) (**idcc).print(2);

   cout <<"--- --- --- --- --- --- --- --- ---"<<endl; 
   cout <<"--- --- --- --- --- --- --- --- ---"<<endl; 
   ChamberRawDataSpec linkboard;
   linkboard.dccId = 790;
   linkboard.dccInputChannelNum = 1;
   linkboard.tbLinkInputNum = 1;
   linkboard.lbNumInLink = 2;

   int febInputNum=3;
   int stripPinNum=5;

   const LinkBoardSpec *location = map->location(linkboard);
   if (location) {
     location->print();
     const FebConnectorSpec * feb = location->feb( febInputNum);
     const ChamberStripSpec * strip = feb->strip(stripPinNum);
     feb->print();
     strip->print();
     cout <<" DETID: " << endl;
     uint32_t id = feb->rawId();
     cout << "uint32_t: " << id << endl;
     RPCDetId rpcDetId(id);
     cout << rpcDetId << endl;
  }
     

   
}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCReadOutMapAnalyzer);
