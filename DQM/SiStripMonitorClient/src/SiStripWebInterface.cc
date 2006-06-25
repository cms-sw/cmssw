#include "DQM/SiStripMonitorClient/interface/SiStripWebInterface.h"
#include "DQMServices/WebComponents/interface/Button.h"
#include "DQMServices/WebComponents/interface/CgiWriter.h"
#include "DQMServices/WebComponents/interface/CgiReader.h"
#include "DQMServices/WebComponents/interface/ConfigBox.h"
#include "DQMServices/WebComponents/interface/Navigator.h"
#include "DQMServices/WebComponents/interface/ContentViewer.h"
#include "DQMServices/WebComponents/interface/GifDisplay.h"
#include "DQMServices/CoreROOT/interface/DaqMonitorROOTBackEnd.h"
#include <map>
#include <iostream>

//
// -- Constructor
// 
SiStripWebInterface::SiStripWebInterface(std::string theContextURL, std::string theApplicationURL, MonitorUserInterface ** _mui_p) 
  : WebInterface(theContextURL, theApplicationURL, _mui_p) {
  
  theActionFlag = NoAction;

  Navigator * nav = new Navigator(getApplicationURL(), "50px", "50px");
  ContentViewer * cont = new ContentViewer(getApplicationURL(), "180px", "50px");
  GifDisplay * dis = new GifDisplay(getApplicationURL(), "25px","300px", "400px", "550px", "MyGifDisplay"); 

  Button * subcrBut = new Button(getApplicationURL(), "320px", "50px", "SubscribeAll", "Subscribe All");
  Button * compBut = new Button(getApplicationURL(), "360px", "50px", "CheckQTResults", "Check QTest Results");
  Button * sumBut = new Button(getApplicationURL(), "400px", "50px", "CreateSummary", "Create Summary");
  Button * collBut = new Button(getApplicationURL(), "440px", "50px", "CollateME", "Collate ME");
  Button * tkMapBut1 = new Button(getApplicationURL(), "480px", "50px", "CreateTrackerMap1", "Create Persistant TrackerMap");
  Button * tkMapBut2 = new Button(getApplicationURL(), "480px", "300px", "CreateTrackerMap2", "Create TempTrackerMap");
  Button * saveBut = new Button(getApplicationURL(), "480px", "550px", "SaveToFile", "Save To File");


  page_p = new WebPage(getApplicationURL());
  page_p->add("navigator", nav);
  page_p->add("contentViewer", cont);
  page_p->add("gifDisplay", dis);
  page_p->add("Sbbutton", subcrBut);
  page_p->add("Cbutton", compBut);
  page_p->add("Smbutton", sumBut);
  page_p->add("SvButton", saveBut);
  page_p->add("ClButton", collBut);
  page_p->add("Tbutton1", tkMapBut1);
  page_p->add("Tbutton2", tkMapBut2);


}
//
// --  Destructor
// 
SiStripWebInterface::~SiStripWebInterface() {
}
// 
// -- Handles requests from WebElements submitting non-default requests 
//
void SiStripWebInterface::handleCustomRequest(xgi::Input* in,xgi::Output* out)
  throw (xgi::exception::Exception)
{
  // put the request information in a multimap...
  std::multimap<std::string, std::string> request_multimap;
  CgiReader reader(in);
  reader.read_form(request_multimap);

  // get the string that identifies the request:
  std::string requestID = get_from_multimap(request_multimap, "RequestID");

  if (requestID == "SubscribeAll") subscribeAll(in, out);
  if (requestID == "CheckQTResults") theActionFlag = QTestResult;
  if (requestID == "CreateSummary") theActionFlag = Summary;
  if (requestID == "SaveToFile")    theActionFlag = SaveData;
  if (requestID == "CollateME")     theActionFlag = Collate;
  if (requestID == "CreateTrackerMap1") theActionFlag = PersistantTkMap;
  if (requestID == "CreateTrackerMap2") theActionFlag = TemporaryTkMap;
}
//
// -- Subscribe All
//
void SiStripWebInterface::subscribeAll(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception)
{
  cout << " SiStripWebInterface::subscribeAll " << endl;
  (*mui_p)->subscribe("Collector/*");
  return;
}
