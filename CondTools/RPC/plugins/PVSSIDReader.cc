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

class  PVSSIDReader : public edm::EDAnalyzer {
public:
  PVSSIDReader(const edm::ParameterSet& iConfig );
  ~PVSSIDReader() override;
  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) override;
};


PVSSIDReader::PVSSIDReader(const edm::ParameterSet& iConfig ){}
  
PVSSIDReader::~PVSSIDReader(){}

void PVSSIDReader::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup)
{
  edm::ESHandle<RPCObPVSSmap> pvssmapRcd;
  evtSetup.get<RPCObPVSSmapRcd>().get(pvssmapRcd);
  edm::LogInfo("PVSSIDReader") << "[PVSSIDReader::analyze] End Reading Pvssmap" << std::endl;
    

  const RPCObPVSSmap* pvssmap = pvssmapRcd.product();
  std::vector<RPCObPVSSmap::Item> mypvssmap = pvssmap->ObIDMap_rpc; 
  std::vector<RPCObPVSSmap::Item>::iterator ipvssmap;

  std::cout << "************************************" << std::endl;
  for(ipvssmap = mypvssmap.begin(); ipvssmap < mypvssmap.end(); ++ipvssmap){
    std::cout<<"dpid = " << ipvssmap->dpid << " region = " << ipvssmap->region 
	     << " ring = " << ipvssmap->ring << " sector = " << ipvssmap->sector 
	     << " station = " << ipvssmap->station << " layer = " << ipvssmap->layer 
	     << " subsector = " << ipvssmap->subsector << " suptype = " << ipvssmap->suptype << std::endl;
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(PVSSIDReader);

