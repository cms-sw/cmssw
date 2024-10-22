#include <iostream>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/DataRecord/interface/RPCReadOutMappingRcd.h"
#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"

using namespace std;
using namespace edm;

// class declaration
class RPCReadOutMapAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit RPCReadOutMapAnalyzer(const edm::ParameterSet&);
  ~RPCReadOutMapAnalyzer() override = default;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  bool m_flag;
  edm::ESGetToken<RPCEMap, RPCEMapRcd> readoutMappingToken_;
  edm::ESGetToken<RPCReadOutMapping, RPCReadOutMappingRcd> mapZeroToken_;
};

RPCReadOutMapAnalyzer::RPCReadOutMapAnalyzer(const edm::ParameterSet& iConfig)
    : m_flag(iConfig.getUntrackedParameter<bool>("useNewEMap", false)),
      readoutMappingToken_(esConsumes()),
      mapZeroToken_(esConsumes()) {
  ::putenv(const_cast<char*>(std::string("CORAL_AUTH_USER konec").c_str()));
  ::putenv(const_cast<char*>(std::string("CORAL_AUTH_PASSWORD konecPass").c_str()));
}

void RPCReadOutMapAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::cout << "====== RPCReadOutMapAnalyzer" << std::endl;

  const RPCReadOutMapping* map;
  if (m_flag) {
    map = iSetup.getData(readoutMappingToken_).convert();
  } else {
    map = &iSetup.getData(mapZeroToken_);
  }
  cout << "version: " << map->version() << endl;

  pair<int, int> dccRange = map->dccNumberRange();
  cout << " dcc range: " << dccRange.first << " to " << dccRange.second << endl;
  vector<const DccSpec*> dccs = map->dccList();
  typedef vector<const DccSpec*>::const_iterator IDCC;
  for (IDCC idcc = dccs.begin(); idcc != dccs.end(); idcc++)
    cout << (**idcc).print(4);

  cout << "--- --- --- --- --- --- --- --- ---" << endl;
  cout << "--- --- --- --- --- --- --- --- ---" << endl;
  /*
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
     
*/
}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCReadOutMapAnalyzer);
