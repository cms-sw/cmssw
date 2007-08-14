#include "DQM/L1TMonitorClient/interface/L1TBaseClient.h"
#include "DQMServices/Core/interface/MonitorElementBaseT.h"
#include "DQMServices/ClientConfig/interface/SubscriptionHandle.h"
#include "DQMServices/ClientConfig/interface/QTestHandle.h"


using namespace edm;
using namespace std;


L1TBaseClient::L1TBaseClient( std::string name, 
			      std::string server, 
			      int port,
			      int reconnect_delay_secs,
			      bool actAsServer)
{

  mui_ = new MonitorUIRoot(server, port, name, reconnect_delay_secs,actAsServer);
  

}

//////////////////////////////////////

L1TBaseClient::~L1TBaseClient()
{

}

//////////////////////////////////////

void L1TBaseClient::bookHisto(string sourceName, string meName)
{

  dbe->setCurrentFolder("Collector/GlobalDQM/L1TMonitor/QualityResults/" + sourceName);
  
  histo = dbe->book1D("test","test",25,-250.,250.);

}

//////////////////////////////////////

void L1TBaseClient::createQtMEResult(string sourceName, MonitorElement * qtResultName)
{

  dbe->setCurrentFolder("Collector/GlobalDQM/L1TMonitor/QualityResults/" + sourceName);
  
  qtResultName = dbe->book1D("test","test",25,-250.,250.);

}

//////////////////////////////////////

string L1TBaseClient::getME(string sourceName, string meName)
{
  histoName = "Collector/GlobalDQM/L1TMonitor/" + sourceName +"/" + meName;
  
  return histoName;
}

//////////////////////////////////////

void L1TBaseClient::saveResults()
{
  dbe->save("pippo.root");
}


void L1TBaseClient::getReport(string meName, DaqMonitorBEInterface * dbi, string criterion)
{

//  string mePath = "Collector/GlobalDQM/L1TMonitor/" + meName;

//  MonitorElement * me_ = dbi->get(mePath);

  MonitorElement * me_ = dbi->get(meName);
  
//  LogInfo("TriggerDQM") << "me_" << me_;

  if(me_) {
  LogInfo("TriggerDQM") << "[" << meName << "] has " << me_->hasError() << " errors, " 
                        << me_->hasWarning() << " warnings, " 
			<< me_->hasOtherReport() << " other reports.";

  const QReport * testReport = me_->getQReport(criterion);

    
  if(testReport) {

//  LogInfo("TriggerDQM") << "testReport " << testReport;

  LogInfo("TriggerDQM") << "[" << meName << "] Quality Report Message = " << testReport->getMessage();

   vector<dqm::me_util::Channel> badChannels = testReport->getBadChannels();
    for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin();
      channel != badChannels.end(); channel++) {
      LogInfo("tTrigCalibration") << " Bad channels: " << (*channel).getBin() << "  Contents : "<< (*channel).getContents();
      LogInfo("tTrigCalibration") << testReport->getMessage() << " ------- " << testReport->getStatus();
 
   }

  }
  
 } else { LogInfo("TriggerDQM") << "Element " << meName << " not available.";}


}




vector<dqm::me_util::Channel> L1TBaseClient::getBadChannels(string meName, DaqMonitorBEInterface * dbi, string criterion)
{

  string mePath = "Collector/GlobalDQM/L1TMonitor/" + meName;

  MonitorElement * me_ = dbi->get(mePath);

  const QReport * testReport = me_->getQReport(criterion);
  
  vector<dqm::me_util::Channel> badChannels;
  
  badChannels.clear();
   
  if(testReport) {

     badChannels = testReport->getBadChannels();

  } else { LogInfo("tTrigCalibration") << " No bad channels found!";} 

  return badChannels;
}





TH1F * L1TBaseClient::get1DHisto(string meName, DaqMonitorBEInterface * dbi)
{

//  string mePath = "Collector/GlobalDQM/L1TMonitor/" + meName;

//  MonitorElement * me_ = dbi->get(mePath);
  MonitorElement * me_ = dbi->get(meName);

  MonitorElementT<TNamed>* me_temp = dynamic_cast<MonitorElementT<TNamed>*>(me_);
  
  TH1F * meHisto = NULL;
  
  if (me_temp) {
	      
	 meHisto = dynamic_cast<TH1F*> (me_temp->operator->());
  }
  else { 
   
         LogInfo("TriggerDQM") << "ME " << meName << " NOT FOUND!";
     
       }
  
  return meHisto;
}


TH2F * L1TBaseClient::get2DHisto(string meName, DaqMonitorBEInterface * dbi)
{

  string mePath = "Collector/GlobalDQM/L1TMonitor/" + meName;

  MonitorElement * me_ = dbi->get(mePath);

  MonitorElementT<TNamed>* me_temp = dynamic_cast<MonitorElementT<TNamed>*>(me_);
  
  TH2F * meHisto = NULL;
  
  if (me_temp) {
	      
	 meHisto = dynamic_cast<TH2F*> (me_temp->operator->());
  }
  else { 
   
         LogInfo("TriggerDQM") << "ME NOT FOUND.";
     
       }
  
  return meHisto;
}





