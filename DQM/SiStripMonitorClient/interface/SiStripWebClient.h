#ifndef _SiStripWebClient_h_
#define _SiStripWebClient_h_

#include "DQMServices/WebComponents/interface/DQMWebClient.h"
#include "DQMServices/WebComponents/interface/WebElement.h"
#include "DQMServices/WebComponents/interface/WebPage.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiStripMonitorClient/interface/SiStripQualityTester.h"

class TrackerMap;

class SiStripWebClient : public DQMWebClient
{
 private:

  WebPage * page;
  // Summary Histogram
  MonitorElement * h1;
  // the SiStripQualityTester
  SiStripQualityTester* theQualityTester;

  TrackerMap*   theTrackerMap;
 public:
  typedef std::map< int, vector<string> > DetMapType;
 
  XDAQ_INSTANTIATOR();
  
  SiStripWebClient(xdaq::ApplicationStub * s);
 ~SiStripWebClient();

  void Default(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception);
  void Request(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception);

  void subscribeAll(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception);
  void setupQTest(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception);
  void createTkMap(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception);
  void createSummary(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception);
  void saveToFile(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception);

  int getUpdates();
  void getValuesForTkMap(DetMapType& values);
  void fillSummary(string name, string type);
  MonitorElement* getSummaryME(string name, string tag);

 protected:


};

#endif
