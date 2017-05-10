// system include files
#include <memory>
#include <stdio.h>
#include <string>

// Framework

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include "DQM/RPCMonitorClient/interface/RPCClient.h"


//
// class decleration
//

class RPCQualityTests : public  DQMEDHarvester{
   public:
      explicit RPCQualityTests(const edm::ParameterSet&);
      ~RPCQualityTests();


 protected:
  void beginJob();
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&); //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob

 
   private:
      
      std::map<std::string , RPCClient*> makeQTestMap();


      int nevents_;

       std::string hostName_;
      int hostPort_;
      std::string clientName_;
      

      bool  getQualityTestsFromFile_;


      edm::ParameterSet parameters_;
      bool enableMonitorDaemon_;
      bool enableQTests_;
      bool enableInputFile_;

      std::vector<RPCClient*> qtests_; 
      std::vector<std::string>  qtestList_; 
      std::string inputFile_;
      std::string eventInfoPath_;
      int eSummary_;
      int prescaleFactor_;

      //   std::vector<std::string> enabledQTests_;
      //          std::map<std::string , RPCClient*> qTest_;

      // ----------member data ---------------------------
};
