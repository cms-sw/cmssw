#include "DQM/SiStripMonitorClient/interface/SiStripWebInterface.h"
#include "DQMServices/WebComponents/interface/Button.h"
#include "DQMServices/WebComponents/interface/CgiWriter.h"
#include "DQMServices/WebComponents/interface/CgiReader.h"
#include "DQMServices/WebComponents/interface/ConfigBox.h"
#include "DQMServices/WebComponents/interface/Navigator.h"
#include "DQMServices/WebComponents/interface/ContentViewer.h"
#include "DQMServices/WebComponents/interface/GifDisplay.h"
#include "DQMServices/CoreROOT/interface/DaqMonitorROOTBackEnd.h"
#include "DQM/SiStripMonitorClient/interface/SiStripQualityTester.h"
#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include <map>
#include <iostream>

//
// -- Constructor
// 
SiStripWebInterface::SiStripWebInterface(std::string theContextURL, std::string theApplicationURL, MonitorUserInterface ** _mui_p) 
  : WebInterface(theContextURL, theApplicationURL, _mui_p) {
  
  theQualityTester = 0;
  theActionExecutor = 0;

  Navigator * nav = new Navigator(getApplicationURL(), "50px", "50px");
  ContentViewer * cont = new ContentViewer(getApplicationURL(), "180px", "50px");
  GifDisplay * dis = new GifDisplay(getApplicationURL(), "50px","370px", "270px", "570px", "MyGifDisplay"); 

  Button * subcrBut = new Button(getApplicationURL(), "320px", "50px", "SubscribeAll", "Subscribe All");
  Button * compBut = new Button(getApplicationURL(), "360px", "50px", "CompareWithRef", "Compare with Reference");
  Button * tkMapBut = new Button(getApplicationURL(), "400px", "50px", "CreateTrackerMap", "Create TrackerMap");
  Button * sumBut = new Button(getApplicationURL(), "440px", "50px", "CreateSummary", "Create Summary");
  Button * saveBut = new Button(getApplicationURL(), "480px", "50px", "SaveToFile", "Save To File");


  page_p = new WebPage(getApplicationURL());
  page_p->add("navigator", nav);
  page_p->add("contentViewer", cont);
  page_p->add("gifDisplay", dis);
  page_p->add("Sbbutton", subcrBut);
  page_p->add("Cbutton", compBut);
  page_p->add("Tbutton", tkMapBut);
  page_p->add("Smbutton", sumBut);
  page_p->add("SvButton", saveBut);

  if (theActionExecutor == 0) theActionExecutor = new SiStripActionExecutor();
}
//
// --  Destructor
// 
SiStripWebInterface::~SiStripWebInterface() {
  if (theQualityTester) delete theQualityTester;
  theQualityTester = 0;
  if (theActionExecutor) delete theActionExecutor;
  theActionExecutor = 0;
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
  if (requestID == "CompareWithRef") setupQTest(in, out);
  if (requestID == "CreateTrackerMap") createTkMap(in, out);
  if (requestID == "CreateSummary") createSummary(in, out);
  if (requestID == "SaveToFile") saveToFile(in, out);
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
//
// -- Set Up Quality Test
//
void SiStripWebInterface::setupQTest(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception)
{
  std::cout << "A setupQTest request was received" << endl;
  if (theQualityTester == 0) {
    theQualityTester = new SiStripQualityTester();
    theQualityTester->setupQTests((*mui_p));
  } else {
    if (getUpdates() > 30) {
      cout << " Checking Comparison results "<< endl;
      theActionExecutor->checkTestResults((*mui_p));
    }
  }
  return;
}
//
// -- Create Tracker Map
//
void SiStripWebInterface::createTkMap(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception)
{
  std::cout << "A createTkMap request was received" << endl;
  int updates = getUpdates();
  if (updates == 0) {
    subscribeAll(in, out);
    return;
  } else if (updates < 10) {
    cout << " Not enough updates received !!" << endl;
    return;
  }
  // Create the Quality Test
  //  if (theQualityTester == 0) setupQTest(in, out);
  std::vector<std::string> me_names;
  int nval = theQualityTester->getMEsUnderTest(me_names);
  if (nval != 0) theActionExecutor->createTkMap((*mui_p), me_names);
  else {
    me_names.clear();
    me_names.push_back("ClustersPerDetector"); 
    me_names.push_back("DigisPerDetector"); 
    theActionExecutor->createTkMap((*mui_p), me_names);
  }
  return;
}
//
// -- Create Summary
//
void SiStripWebInterface::createSummary(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception) {
  std::cout << "A createSummary request was received" << endl;
  if (getUpdates() > 10) theActionExecutor->fillSummary((*mui_p), "string_", "DigisPerDetector");
  return;
}
//
// -- Save to File
//
void SiStripWebInterface::saveToFile(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception) {
  cout << " Saving Monitoring Elements " << endl;
  (*mui_p)->save("SiStripWebInterface.root", "Collector",90);
//  (*mui_p)->save("SiStripWebInterface.root");
  return;
}
//
// -- get # of updates 
//
int SiStripWebInterface::getUpdates() {
  if (!(*mui_p)) return -1;
  int updates = (*mui_p)->getNumUpdates();
  (*mui_p)->subscribeNew("Collector/*");
  return updates;
}
  
