//
// Original Author:  Davide Pagano
//         Created:  Wed May 20 12:47:20 CEST 2009
// $Id: RiovTest.cc,v 1.3 2009/05/22 18:18:11 dpagano Exp $
//
//


#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/RPCObjects/interface/RPCObCond.h"
#include "CondFormats/DataRecord/interface/RPCObCondRcd.h"
#include "CoralBase/TimeStamp.h"
#include "CondTools/RPC/interface/RPCRunIOV.h"



class RiovTest : public edm::EDAnalyzer {
public:
  explicit RiovTest(const edm::ParameterSet&);
  ~RiovTest();
  
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  std::vector<RPCObImon::I_Item> filterData(unsigned long long since, unsigned long long till);
  
  RPCRunIOV* connection;
  std::vector<RPCObImon::I_Item> imon;
  std::vector<RPCObImon::I_Item> imon_;
  std::vector<RPCObImon::I_Item> filtImon;
  std::vector<unsigned long long> listIOV;
  unsigned long long min;
  unsigned long long max;
};

RiovTest::RiovTest(const edm::ParameterSet& iConfig)

{
}


RiovTest::~RiovTest()
{
}



void
RiovTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // TIME OF EVENT-------------------------
   std::cout << "=== Event " << iEvent.id().event() << " ===" << std::endl;
   TimeValue_t timeD = iEvent.time().value();
   std::cout << "DAQ  = " << timeD << std::endl;
   timeval tmval=(timeval&)timeD;
   unsigned long long int timeU;
   timeU = (tmval.tv_usec*1000000LL)+tmval.tv_sec;
   timeU = (unsigned long long int)trunc(timeU/1000000);
   std::cout << "UNIX = " << timeU << std::endl;

   // CONNECTION TO DATABASE----------------
   RPCRunIOV* list = new RPCRunIOV(iSetup);
   if (imon.size() == 0) {
     imon = list->getImon();
     min = list->min;
     max = list->max;
   } else if (timeU < min || timeU > max) {
     imon_ = list->getImon();
     std::cout << "Before [size] " << imon.size();  
     for(std::vector<RPCObImon::I_Item>::iterator it = imon_.begin(); it < imon_.end(); ++it)
       imon.push_back(*it);
     std::cout << " <--> After [size] " << imon.size() << std::endl;  
     if (min > list->min) min = list->min;
     if (max < list->max) max = list->max;
   }
   //----------------------------------------
   std::cout << "MIN = " << min << std::endl;
   std::cout << "MAX = " << max << std::endl;
}


// this methos filters data
std::vector<RPCObImon::I_Item> 
RiovTest::filterData(unsigned long long since, unsigned long long till)
{

  std::cout << std::endl << "=============================================" << std::endl;
  std::cout << std::endl << "============    FILTERING DATA    ===========" << std::endl;
  std::cout << std::endl << "=============================================" << std::endl << std::endl;

  std::vector<RPCObImon::I_Item>::iterator it;

  RPCFw conv ("","","");
  int n = 0;
  for ( it=imon.begin(); it < imon.end(); it++ ) {
    n++;
    int day = (int)it->day/10000;
    int mon = (int)(it->day - day*10000)/100;
    int yea = (int)(it->day - day*10000 - mon*100)+2000;
    int hou = (int)it->time/10000;
    int min = (int)(it->time - hou*10000)/100;
    int sec = (int)(it->time - hou*10000 - min*100);
    int nan = 0;
    coral::TimeStamp timeD = coral::TimeStamp(yea, mon, day, hou, min, sec, nan);
    unsigned long long timeU = conv.TtoUT(timeD);
    
    std::cout << n << " dpid = " << it->dpid << " - value = " << it->value << " - day = " << it->day << " (" << day << "/" << mon << "/" << yea << ") - time = " << it->time << " (" << hou << ":" << min << "." << sec << ") - UT = " << timeU << std::endl;

    if (timeU < till && timeU > since) filtImon.push_back(*it);
    
  }

  return filtImon;
}




void 
RiovTest::beginJob()
{
}


void 
RiovTest::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(RiovTest);
