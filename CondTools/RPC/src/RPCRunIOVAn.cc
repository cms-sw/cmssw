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
#include "CondTools/RPC/interface/RPCIOVReader.h"
#include "CoralBase/TimeStamp.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class  RPCRunIOVAn : public edm::EDAnalyzer {
public:
  RPCRunIOVAn(const edm::ParameterSet& iConfig);
  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup);
  ~RPCRunIOVAn();

private:
  unsigned long long since;
  unsigned long long till;
};


RPCRunIOVAn::RPCRunIOVAn(const edm::ParameterSet& iConfig) : 
  since(iConfig.getUntrackedParameter<unsigned long long>("since",0)),
  till(iConfig.getUntrackedParameter<unsigned long long>("till",0))
{}

RPCRunIOVAn::~RPCRunIOVAn(){}

void 
RPCRunIOVAn::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  
  std::cout << std::endl << "=============================================" << std::endl;
  std::cout << std::endl << "============  RUN IOV ASSOCIATOR  ===========" << std::endl;
  std::cout << std::endl << "=============================================" << std::endl << std::endl;
  
  
  std::cout << ">> RUN start: " << since << std::endl;
  std::cout << ">> RUN  stop: " << till << std::endl;
  std::cout << std::endl << "=============================================" << std::endl << std::endl;

  std::vector<unsigned long long> iov_vect;
  RPCIOVReader iov_list ("sqlite_file:dati.db", "CMS_COND_GENERAL_R", "rd0548in");
  iov_vect = iov_list.listIOV();
  
  

   unsigned long long iov;
   std::vector<unsigned long long> final_vect;
   std::vector<unsigned long long>::iterator it, it_fin;

   if (iov_vect.front() < since) {
     for (it = iov_vect.begin(); it != iov_vect.end(); it++) {
       iov = *(it);
       //std::cout << iov << std::endl;
       if (since < iov && iov < till) {
	 if (final_vect.size() == 0) {
	   *it--;
	   final_vect.push_back(*it);
	   *it++;
	 } 
	 final_vect.push_back(iov); 
       } 
     }
     std::cout << std::endl << "=============================================" << std::endl;
     std::cout <<              "        Accessing the following IOVs\n        "<< std::endl; 
     for (it_fin = final_vect.begin(); it_fin != final_vect.end(); it_fin++) {
       iov = *(it_fin);
       std::cout << iov << "\n";
     }
   } else {
     std::cout << "   WARNING: run not included in data range\n";
   }

   std::vector<RPCObImon::I_Item> IMON;
   IMON = iov_list.getIMON(final_vect.front(), final_vect.back());
   std::cout << "\n>> Imon vector created --> size: " << IMON.size() << std::endl;
   
   // PRINT
   RPCObImon::I_Item temp;
   std::string day,time;
   for (std::vector<RPCObImon::I_Item>::iterator ii = IMON.begin(); ii != IMON.end(); ii++) {
     temp = *(ii);
     day  = iov_list.toDay(temp.day);
     time = iov_list.toTime(temp.time);
     std::cout << "ID: " << temp.dpid << " - Val: " << temp.value << " - Day: " << day << " - Time: " << time << std::endl;
   }



}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCRunIOVAn);
