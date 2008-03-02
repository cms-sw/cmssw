#ifndef SiStripHistoricInfoClient_SiStripHistoricInfoClient_h
#define SiStripHistoricInfoClient_SiStripHistoricInfoClient_h
// -*- C++ -*-
//
// Package:     SiStripHistoricInfoClient
// Class  :     SiStripHistoricInfoClient
// 
/**\class SiStripHistoricInfoClient SiStripHistoricInfoClient.h DQM/SiStripHistoricInfoClient/interface/SiStripHistoricInfoClient.h

 Description: <Non interactive DQM client of the SiStripTk. Writes out in a DB the information needed for the DQM historic plots: https://uimon.cern.ch/twiki/bin/view/CMS/DQMHistoricInfoPlots>

 Usage:
    <usage>

*/
//
// Original Author:  dkcira
//         Created:  Thu Jun 15 09:32:34 CEST 2006
// $Id: SiStripHistoricInfoClient.h,v 1.10 2007/12/11 20:13:53 dutta Exp $
//

#include "DQMServices/XdaqCollector/interface/DQMBaseClient.h"
#include "DQMServices/XdaqCollector/interface/Updater.h"
#include "DQMServices/XdaqCollector/interface/UpdateObserver.h"
#include "DQM/SiStripHistoricInfoClient/interface/SiStripHistoricInfoWebInterface.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondFormats/SiStripObjects/interface/SiStripPerformanceSummary.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
  //GB: 25/11/07 commented out this apparently useless Online DB dependency
//#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"

#include "xoap/SOAPBody.h"
#include "xoap/SOAPEnvelope.h"
#include "xdata/Table.h"
#include "xdata/TimeVal.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class SiStripHistoricInfoClient : public DQMBaseClient, public dqm::UpdateObserver
{
public:
  // You always need to have this line! Do not remove:
  XDAQ_INSTANTIATOR();
  // The class constructor:  
  SiStripHistoricInfoClient(xdaq::ApplicationStub *s);
  // implement the method that outputs the page with the widgets (declared in DQMBaseClient):
  void general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception);
  // the method which answers all HTTP requests of the form ".../Request?RequestID=..."
  void handleWebRequest(xgi::Input * in, xgi::Output * out);
  // this obligatory method is called whenever the client enters the "Configured" state:
  void configure();
  // this obligatory method is called whenever the client enters the "Enabled" state:
  void newRun();
  // this obligatory method is called whenever the client enters the "Halted" state:
  void endRun();
  // this obligatory method is called by the Updater component, whenever there is an update 
  void onUpdate() const;
  // write to cond db
  void writeToDB() const;
  // print out list of mean and rms values for MEs, moved in a separate method so that can call from different points of state machine
  void printMEs() const;
public:
  // this client has a web interface:  
  SiStripHistoricInfoWebInterface * webInterface_p;
private:
  //
  void retrievePointersToModuleMEs() const;
  void fillSummaryObjects() const;
private:
  mutable bool firstTime;
  mutable int  firstUpdate;
  mutable std::map<uint32_t, std::vector<MonitorElement *> > ClientPointersToModuleMEs;
  SiStripPerformanceSummary* pSummary_;
  // Access to the configuration DB interface class
  //GB: 25/11/07 commented out this apparently useless Online DB dependency
  //  SiStripConfigDb* db_;
};
// You always need to have this line! Do not remove:
XDAQ_INSTANTIATOR_IMPL(SiStripHistoricInfoClient)
#endif
