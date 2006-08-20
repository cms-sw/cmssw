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
// $Id$
//

#include "DQMServices/Components/interface/DQMBaseClient.h"
#include "DQMServices/Components/interface/Updater.h"
#include "DQMServices/Components/interface/UpdateObserver.h"

#include "DQMServices/Core/interface/MonitorUserInterface.h"

#include "DQM/SiStripHistoricInfoClient/interface/SiStripHistoricInfoWebInterface.h"


class SiStripHistoricInfoClient : public DQMBaseClient,
                              public dqm::UpdateObserver
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

  // test TStore 
  void tstore_connect();

private:
  //
  void retrievePointersToModuleMEs() const;
  void fillSummaryObjects();


public:
  // this client has a web interface:  
  SiStripHistoricInfoWebInterface * webInterface_p;

private:
  //
  mutable bool firstTime;
  mutable int  firstUpdate;
  //
  mutable std::map<uint32_t, std::vector<MonitorElement *> > ClientPointersToModuleMEs;
  mutable std::map<uint32_t, std::pair<double, double> > ClusterChargeMeanRMS;
  mutable std::map<uint32_t, std::pair<double, double> > OccupancyMeanRMS;

private:

};

// You always need to have this line! Do not remove:
XDAQ_INSTANTIATOR_IMPL(SiStripHistoricInfoClient)

#endif
