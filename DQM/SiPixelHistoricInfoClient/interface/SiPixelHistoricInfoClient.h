#ifndef SiPixelHistoricInfoClient_SiPixelHistoricInfoClient_h
#define SiPixelHistoricInfoClient_SiPixelHistoricInfoClient_h

/* Description: non-interactive DQM client of the SiPixelTk(?) 
                writes out in a DB the information needed for 
                DQM historic plots in 
 https://uimon.cern.ch/twiki/bin/view/CMS/DQMHistoricInfoPlots */

#include "DQMServices/Components/interface/DQMBaseClient.h"
#include "DQMServices/Components/interface/Updater.h"
#include "DQMServices/Components/interface/UpdateObserver.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelHistoricInfoWebInterface.h"


class SiPixelHistoricInfoClient : public DQMBaseClient, public dqm::UpdateObserver {
public: 
  // You always need to have this line! Do not remove:
  XDAQ_INSTANTIATOR();

  // The class constructor:  
  SiPixelHistoricInfoClient(xdaq::ApplicationStub* stub);

  // implement the method that outputs the page with the widgets (declared in DQMBaseClient):
  void general(xgi::Input* in, xgi::Output* out ) throw (xgi::exception::Exception);

  // the method which answers all HTTP requests of the form ".../Request?RequestID=..."
  void handleWebRequest(xgi::Input* in, xgi::Output* out);

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

  // this client has a web interface:  
  SiPixelHistoricInfoWebInterface* webInterface_p;

private:
  void retrievePointersToModuleMEs() const;
  void fillSummaryObjects();

  mutable bool firstTime;
  mutable int  firstUpdate;

  mutable std::map< uint32_t, std::pair<double, double> > ClusterChargeMeanRMS;
  mutable std::map< uint32_t, std::pair<double, double> > OccupancyMeanRMS; 
  mutable std::map< uint32_t, std::vector<MonitorElement*> > ClientPointersToModuleMEs;
};


// You always need to have this line! Do not remove:
XDAQ_INSTANTIATOR_IMPL(SiPixelHistoricInfoClient)

#endif
