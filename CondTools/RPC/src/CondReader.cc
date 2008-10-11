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
#include "CondFormats/RPCObjects/interface/RPCObCond.h"
#include "CondFormats/DataRecord/interface/RPCObCondRcd.h"

class  CondReader : public edm::EDAnalyzer {
public:
  CondReader(const edm::ParameterSet& iConfig );
  ~CondReader();
  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup);
};


CondReader::CondReader(const edm::ParameterSet& iConfig ){}
  
CondReader::~CondReader(){}

void CondReader::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup)
{
  edm::ESHandle<RPCObCond> condRcd;
  evtSetup.get<RPCObCondRcd>().get(condRcd);
  edm::LogInfo("CondReader") << "[CondReader::analyze] End Reading Cond" << std::endl;
    

  const RPCObCond* cond = condRcd.product();
  std::vector<RPCObCond::Item> mycond = cond->ObImon_rpc; 
  std::vector<RPCObCond::Item>::iterator icond;

  std::cout << "************************************" << std::endl;
  for(icond = mycond.begin(); icond < mycond.end(); ++icond){
    std::cout<<"dpid = " << icond->dpid << " - value = " << icond->value << " - day = " << icond->day << " - time = " << icond->time << std::endl;
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(CondReader);
