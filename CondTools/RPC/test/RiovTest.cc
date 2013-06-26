//
// Original Author:  Davide Pagano
//         Created:  Wed May 20 12:47:20 CEST 2009
// $Id: RiovTest.cc,v 1.11 2010/02/19 10:44:00 michals Exp $
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
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  std::vector<RPCObImon::I_Item> imon;
  std::vector<RPCObImon::I_Item> imon_;
  std::vector<RPCObVmon::V_Item> vmon;
  std::vector<RPCObVmon::V_Item> vmon_;
  std::vector<RPCObTemp::T_Item> temp;
  std::vector<RPCObTemp::T_Item> temp_;
  std::vector<RPCObImon::I_Item> filtImon;
  std::vector<unsigned long long> listIOV;
  std::map<int, RPCObPVSSmap::Item> pvssMap;
  unsigned long long min_I;
  unsigned long long max_I;
  unsigned long long min_V;
  unsigned long long max_V;
  unsigned long long min_T;
  unsigned long long max_T;
  unsigned long long RunStart;
  unsigned long long RunStop;
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
   //timeval tmval=(timeval&)timeD;
   timeval tmval;
   unsigned int timelow=static_cast<unsigned int>(0xFFFFFFFF & timeD);
   unsigned int timehigh=static_cast<unsigned int>(timeD >> 32);
   tmval.tv_sec=timehigh;
   tmval.tv_usec=timelow;
   unsigned long long int timeU;
   timeU = (tmval.tv_usec*1000000LL)+tmval.tv_sec;
   timeU = (unsigned long long int)trunc(timeU/1000000);
   std::cout << "UNIX = " << timeU << std::endl;

   // CONNECTION TO DATABASE----------------
   RPCRunIOV* list = new RPCRunIOV(iSetup);
   // get PVSS map
   if (pvssMap.size() == 0) {
     pvssMap = list->getPVSSMap();
   }
   // get current
   if (imon.size() == 0) {
     imon = list->getImon();
     min_I = list->min_I;
     max_I = list->max_I;
   } else if (timeU < min_I || timeU > max_I) {
     imon_ = list->getImon();
     for(std::vector<RPCObImon::I_Item>::iterator it = imon_.begin(); it <
imon_.end(); ++it)
       imon.push_back(*it);
     if (min_I > list->min_I) min_I = list->min_I;
     if (max_I < list->max_I) max_I = list->max_I;
   }
   // get high voltage
   if (vmon.size() == 0) {
     vmon = list->getVmon();
     min_V = list->min_V;
     max_V = list->max_V;
   } else if (timeU < min_V || timeU > max_V) {
     vmon_ = list->getVmon();
     for(std::vector<RPCObVmon::V_Item>::iterator itV = vmon_.begin(); itV <
vmon_.end(); ++itV)
       vmon.push_back(*itV);
     if (min_V > list->min_V) min_V = list->min_I;
     if (max_V < list->max_V) max_V = list->max_I;
   }
   // get temperature
   if (temp.size() == 0) {
     temp = list->getTemp();
     min_T = list->min_T;
     max_T = list->max_T;
   } else if (timeU < min_T || timeU > max_T) {
     temp_ = list->getTemp();
     for(std::vector<RPCObTemp::T_Item>::iterator itT = temp_.begin(); itT <
temp_.end(); ++itT)
       temp.push_back(*itT);
     if (min_T > list->min_T) min_T = list->min_T;
     if (max_T < list->max_T) max_T = list->max_T;
   }



}



void
RiovTest::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  std::cout << ">>> Run " << iRun.run() << std::endl;
  std::cout << "    start " << iRun.beginTime().value() << std::endl;
  std::cout << "     stop " << iRun.endTime().value() << std::endl;
}



void 
RiovTest::beginJob()
{}



void 
RiovTest::endJob() {
  
  if (imon.size() > 0) {
    std::cout << ">>> Object IMON" << std::endl;
    std::cout << "    size " << imon.size() << std::endl;
    std::cout << "    from " << min_I << " to " << max_I << std::endl;
    
    // filtering has to be here
    RPCRunIOV* filter = new RPCRunIOV();
    filtImon = filter->filterIMON(imon, min_I+1000, max_I-1000);
    std::cout << ">>> Filtered IMON" << std::endl;
    std::cout << "    size " << filtImon.size() << std::endl;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(RiovTest);
