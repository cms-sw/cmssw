// -*- C++ -*-
//
// Package:    RiovTest
// Class:      RiovTest
// 
/**\class RiovTest RiovTest.cc CondTools/RiovTest/src/RiovTest.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Davide Pagano
//         Created:  Wed May 20 12:47:20 CEST 2009
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/RPCObjects/interface/RPCObCond.h"
#include "CondFormats/DataRecord/interface/RPCObCondRcd.h"
#include "CondTools/RPC/interface/RPCIOVReader.h"
#include "CoralBase/TimeStamp.h"
#include "CondTools/RPC/interface/RPCRunIOV.h"


//
// class decleration
//

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
  std::vector<RPCObImon::I_Item> filtImon;
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
   
   unsigned long long since = 1242731700;
   unsigned long long till  = 1242733500;
   
   connection = new RPCRunIOV(since, till);
   // get data
   imon = connection->getData();

   // filter data
   this->filterData(since, till);

   // print data
   for (std::vector<RPCObImon::I_Item>::iterator it=filtImon.begin(); it < filtImon.end(); it++ ) {
     //std::cout<<"dpid = " << it->dpid << " - value = " << it->value << " - day = " << it->day << " - time = " << it->time << std::endl;
   }
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
    
    //std::cout << n << " dpid = " << it->dpid << " - value = " << it->value << " - day = " << it->day << " (" << day << "/" << mon << "/" << yea << ") - time = " << it->time << " (" << hou << ":" << min << "." << sec << ") - UT = " << timeU << std::endl;

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
