//
// Original Author:  Davide Pagano
//         Created:  Wed May 20 12:47:20 CEST 2009
// $Id: RiovTest.cc,v 1.6 2009/05/24 14:44:13 dpagano Exp $
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
  std::vector<RPCObImon::I_Item> filtImon;
  std::vector<unsigned long long> listIOV;
  unsigned long long min;
  unsigned long long max;
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
     //     std::cout << "Before [size] " << imon.size();  
     for(std::vector<RPCObImon::I_Item>::iterator it = imon_.begin(); it < imon_.end(); ++it)
       imon.push_back(*it);
     //     std::cout << " <--> After [size] " << imon.size() << std::endl;  
     if (min > list->min) min = list->min;
     if (max < list->max) max = list->max;
   }
   //   std::cout << "MIN = " << min << std::endl;
   //   std::cout << "MAX = " << max << std::endl;
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
    std::cout << "    from " << min << " to " << max << std::endl;
    
    // filtering has to be here
    RPCRunIOV* filter = new RPCRunIOV();
    filtImon = filter->filterIMON(imon, min+1000, max-1000);
    std::cout << ">>> Filtered IMON" << std::endl;
    std::cout << "    size " << filtImon.size() << std::endl;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(RiovTest);
