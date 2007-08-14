#ifndef L1TBASECLIENT_H
#define L1TBASECLIENT_H
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include <DQMServices/UI/interface/MonitorUIRoot.h>
#include <TH1F.h>
#include <TH2F.h>

using namespace edm;
using namespace std;

//class SubscriptionHandle;
//class QTestHandle;//
   
class L1TBaseClient 

{
   public:
      explicit L1TBaseClient(std::string name = "DQMBaseClient", 
                 std::string server = "localhost", 
                 int port = 9090, 
                 int reconnect_delay_secs = 5,
                 bool actAsServer = false);

      ~L1TBaseClient();
      void getReport(string meName, DaqMonitorBEInterface * dbi ,  string criterion);
      vector<dqm::me_util::Channel> getBadChannels(string meName, DaqMonitorBEInterface * dbi ,  string criterion);
      TH1F * get1DHisto(string meName, DaqMonitorBEInterface * dbi);
      TH2F * get2DHisto(string meName, DaqMonitorBEInterface * dbi);

   private:

      void bookHisto(string sourceName, string meName);

      void createQtMEResult(string sourceName, MonitorElement * qtResultName);
      
      string getME(string sourceName, string meName);
      
      void saveResults();

      
      DaqMonitorBEInterface * dbe;
      MonitorElement *histo;
      MonitorElement *qtResultName;
      string histoName;

   protected:
 
   MonitorUserInterface *mui_; 
};

#endif
