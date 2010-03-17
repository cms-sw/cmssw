// system include files
#include <memory>
#include <stdio.h>
#include <string>

// Framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/RPCMonitorClient/interface/RPCClient.h"
#include "DQMServices/Core/interface/DQMStore.h"


//
// class decleration
//

class RPCQualityTests : public edm::EDAnalyzer {
   public:
      explicit RPCQualityTests(const edm::ParameterSet&);
      ~RPCQualityTests();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob();
      virtual void beginRun(const edm::Run& r, const edm::EventSetup& c);
      virtual void endRun(const edm::Run& r, const edm::EventSetup& c);  

      void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;
      void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);
     
      std::map<std::string , RPCClient*> makeQTestMap();


      int nevents_;

      DQMStore * dbe_ ;

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
