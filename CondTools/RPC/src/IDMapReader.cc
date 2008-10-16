#include <string>
#include <map>
#include <vector>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondTools/RPC/interface/RPCDBSimSetUp.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "CondFormats/RPCObjects/interface/RPCObPVSSmap.h"
#include "CondFormats/DataRecord/interface/RPCObPVSSmapRcd.h"

class  IDMapReader : public edm::EDAnalyzer {
public:
  IDMapReader(const edm::ParameterSet& iConfig );
  ~IDMapReader();
  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup);
};


IDMapReader::IDMapReader(const edm::ParameterSet& iConfig ){}

IDMapReader::~IDMapReader(){}

void IDMapReader::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup)
{
  edm::ESHandle<RPCObPVSSmap> IDMapRcd;
  evtSetup.get<RPCObPVSSmapRcd>().get(IDMapRcd);
  edm::LogInfo("IDMapReader") << "[IDMapReader::analyze] End Reading IDMap" << std::endl;


  const RPCObPVSSmap* IDMap = IDMapRcd.product();
  std::vector<RPCObPVSSmap::Item> myidmap = IDMap->ObIDMap_rpc;
  std::vector<RPCObPVSSmap::Item>::iterator iidmap;

  std::cout << "************************************" << std::endl;
  for(iidmap = myidmap.begin(); iidmap < myidmap.end(); ++iidmap){
    std::cout <<"dpid = " << iidmap->dpid << " - since: " << iidmap->since <<  " - ring: " << iidmap->ring 
              <<  " - sect: " << iidmap->sector <<  " - station = " << iidmap->station 
              <<  " - lay: " << iidmap->since <<  " - sublay: " << iidmap->since 
              <<  " - type: " << iidmap->suptype << std::endl;
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(IDMapReader);

