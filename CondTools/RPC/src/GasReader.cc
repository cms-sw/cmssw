#include <string>
#include <map>
#include <vector>
#include <sstream>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondTools/RPC/interface/RPCDBSimSetUp.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "CondFormats/RPCObjects/interface/RPCObGas.h"
#include "CondFormats/DataRecord/interface/RPCObGasRcd.h"

class  GasReader : public edm::EDAnalyzer {
public:
  GasReader(const edm::ParameterSet& iConfig );
  ~GasReader();
  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup);
};


GasReader::GasReader(const edm::ParameterSet& iConfig ){}
  
GasReader::~GasReader(){}

void GasReader::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup)
{
  edm::ESHandle<RPCObGas> gasRcd;
  evtSetup.get<RPCObGasRcd>().get(gasRcd);
  edm::LogInfo("RPCReader") << "[RPCpayloadReader::analyze] End Reading RPCpayload" << std::endl;
    

  const RPCObGas* gas = gasRcd.product();
  std::vector<RPCObGas::Item> mygas = gas->ObGas_rpc; 
  std::vector<RPCObGas::Item>::iterator igas;

  std::cout << "************************************" << std::endl;
  for(igas = mygas.begin(); igas < mygas.end(); ++igas){
    std::cout<<"dpid = " << igas->dpid << " - flow IN = " << igas->flowin << " - flow OUT = " << igas->flowin << " - date = " << igas->day << 
    " - time = " << igas->time << std::endl;
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(GasReader);
