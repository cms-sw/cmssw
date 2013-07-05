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
#include "CondTools/RPC/interface/RPCFw.h"
#include "CoralBase/TimeStamp.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class  CondReader : public edm::EDAnalyzer {
public:
  CondReader(const edm::ParameterSet& iConfig);
    
  ~CondReader();
  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup);

private:
  unsigned long long since;
  unsigned long long till;
};


CondReader::CondReader(const edm::ParameterSet& iConfig) : 
  since(iConfig.getUntrackedParameter<unsigned long long>("since",0)),
  till(iConfig.getUntrackedParameter<unsigned long long>("till",0))
{}

CondReader::~CondReader(){}

void CondReader::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup)
{
  
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  edm::ESHandle<RPCObImon> condRcd;
  evtSetup.get<RPCObImonRcd>().get(condRcd);
  edm::LogInfo("CondReader") << "[CondReader::analyze] End Reading Cond" << std::endl;
  
  std::cout << "Run start: " << since << " - Run stop: " << till << std::endl;
  
  RPCFw time ("", "", "");
  coral::TimeStamp sTime = time.UTtoT(since);
  coral::TimeStamp tTime = time.UTtoT(till);
  int ndateS = (sTime.day() * 10000) + (sTime.month() * 100) + (sTime.year()-2000);
  int ntimeS = (sTime.hour() * 10000) + (sTime.minute() * 100) + sTime.second();
  int ndateT = (tTime.day() * 10000) + (tTime.month() * 100) + (tTime.year()-2000);
  int ntimeT = (tTime.hour() * 10000) + (tTime.minute() * 100) + tTime.second();
  std::cout << "Run start: " << ndateS << " " << ntimeS << " - Run stop: " << ndateT << " " << ntimeT << std::endl;

  const RPCObImon* cond = condRcd.product();
  std::vector<RPCObImon::I_Item> mycond = cond->ObImon_rpc; 
  std::vector<RPCObImon::I_Item>::iterator icond;

  std::cout << "************************************" << std::endl;
  for(icond = mycond.begin(); icond < mycond.end(); ++icond){

    if (icond->day >= ndateS && icond->time >= ntimeS && icond->day <= ndateT && icond->time <= ntimeT)
      std::cout<<"dpid = " << icond->dpid << " - value = " << icond->value << " - day = " << icond->day << " - time = " << icond->time << std::endl;
  }
  
}

//define this as a plug-in
DEFINE_FWK_MODULE(CondReader);

