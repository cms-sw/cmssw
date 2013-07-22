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
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>

#include <vector>
#include <bitset>
#include <fstream>
#include <ctime>

using namespace std;
using namespace edm;


typedef bitset<96> LBtestWord;

// class declaration
class TestPulseAnalyzer : public edm::EDAnalyzer {
public:
  explicit TestPulseAnalyzer( const edm::ParameterSet& );
  ~TestPulseAnalyzer();

  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:

  vector<LBtestWord> readTestVectors(string fileName);
  void readRPCDAQStrips(const edm::Event& iEvent);
  void readDTDAQStrips(const edm::Event& iEvent);

};


//////////////////////////////////

TestPulseAnalyzer::TestPulseAnalyzer( const edm::ParameterSet& iConfig )
{
  ::putenv("CORAL_AUTH_USER konec");
  ::putenv("CORAL_AUTH_PASSWORD konecPass");
}


TestPulseAnalyzer::~TestPulseAnalyzer(){}


vector<LBtestWord> TestPulseAnalyzer::readTestVectors(string fileName){

  ifstream in(fileName.c_str());
  string tmp, chName;
  int clockTick;
  bitset<24> bit;
  vector<LBtestWord> vBits;

  in>>tmp>>chName;
  in>>tmp>>tmp>>tmp;

  while(!in.eof()){
    in>>clockTick>>bit;
    //cout<<chName<<" "<<clockTick<<" "<<bit<<endl;
    bitset<96>bit1(0);
    for(int i=0;i<24;i++) if(bit[i]){
      int k = 4*i;
      bit1.set(k+3);
      bit1.set(k+2);
      bit1.set(k+1);
      bit1.set(k);
    }
    //cout<<"Long bit: "<<bit1<<endl;
    vBits.push_back(bit1);
  }

  return vBits;
}


void TestPulseAnalyzer::readRPCDAQStrips(const edm::Event& iEvent){

  edm::Handle<RPCDigiCollection> rpcDigis;
  iEvent.getByType(rpcDigis);
  RPCDigiCollection::DigiRangeIterator rpcDigiCI;
  for(rpcDigiCI = rpcDigis->begin();rpcDigiCI!=rpcDigis->end();rpcDigiCI++){
    //cout<<(*rpcDigiCI).first<<endl;;
    const RPCDigiCollection::Range& range = (*rpcDigiCI).second;    
    for (RPCDigiCollection::const_iterator digiIt = range.first;
         digiIt!=range.second;++digiIt){
      cout<<*digiIt<<endl;
      //      cout<<"Strip number: "<<digiIt->strip()<<endl;
    }
  }
}

void TestPulseAnalyzer::readDTDAQStrips(const edm::Event& iEvent){

  edm::Handle<DTDigiCollection> dtDigis;
  iEvent.getByType(dtDigis);
  DTDigiCollection::DigiRangeIterator dtDigiCI;
  for(dtDigiCI = dtDigis->begin();dtDigiCI!=dtDigis->end();dtDigiCI++){
    //cout<<(*dtDigiCI).first<<endl;;
    const DTDigiCollection::Range& range = (*dtDigiCI).second;    
    for (DTDigiCollection::const_iterator digiIt = range.first;
         digiIt!=range.second;++digiIt){
      digiIt->print();
      //cout<<*digiIt<<endl;
      //      cout<<"Strip number: "<<digiIt->strip()<<endl;
    }
  }
}

void TestPulseAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

   std::cout << "====== TestPulseAnalyzer" << std::endl;
   cout << "--- Run: " << iEvent.id().run()
	<< " Event: " << iEvent.id().event() 
	<< " time: "<<iEvent.time().value();
   time_t aTime = iEvent.time().value();
   cout<<" "<<ctime(&aTime)<<endl;

   
   cout<<"The following strips are read form DAQ: "<<endl;
   cout<<"RPC digis: "<<endl;
   readRPCDAQStrips(iEvent);
   cout<<"DT digis: "<<endl;
   //readDTDAQStrips(iEvent);
   return;

   edm::ESHandle<RPCReadOutMapping> map;
   iSetup.get<RPCReadOutMappingRcd>().get(map);
   cout << "version: " << map->version() << endl;

   cout <<"--- --- --- --- --- --- --- --- ---"<<endl; 
   vector<LBtestWord> testSet = readTestVectors("0_pulse.dat");
   const string chName="RE+1/2/27";
   uint32_t detId = 637570205;
   //std::vector<const LinkBoardSpec*> a = map->getLBforChamber(chName);//
   bitset<96> testVector = testSet[0];
   cout<<"testVector: "<<testVector<<endl;
   cout<<"The following strips should be fired: "<<endl;
   for(int i=0;i<96;i++){
     if(testVector[i]){
       //int strip = a[0]->strip(i).second;
       std::pair<ChamberRawDataSpec, int> 
	 linkboard = map->getRAWSpecForCMSChamberSrip(detId,i,12);
       int strip = map->strip(linkboard.first,i).second;
       cout<<"Chamber: "<<chName
	   <<" strips: "<<strip<<endl;
	   
     }
   }
   cout<<"RAW for strip in chamber: "<<endl;
   for(int i=16;i<100;i++){
   int stripNumber = i;
   std::pair<ChamberRawDataSpec, int>  
     linkboard = map->getRAWSpecForCMSChamberSrip(detId,stripNumber,12);
   cout<<"Link: "<<linkboard.first.tbLinkInputNum<<endl;
   cout<<"LinkChannel: "<<linkboard.first.lbNumInLink<<endl;
   cout<<"LinkLeftBit: "<<linkboard.second<<endl;   
   cout<<"----------------"<<endl;
   }

   //cout<<"The following strips are read form DAQ: "<<endl;
   //readDAQStrips(iEvent);

   return;  
}




//define this as a plug-in
DEFINE_FWK_MODULE(TestPulseAnalyzer);
