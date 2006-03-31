#include "DQM/SiStripMonitorClient/interface/SiStripWebClient.h"
#include "DQM/SiStripMonitorClient/interface/TrackerMap.h"
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
SiStripWebClient::SiStripWebClient(xdaq::ApplicationStub * s) : DQMWebClient(s)
{
  theQualityTester = 0;
  theTrackerMap = 0;

  ConfigBox * box = new ConfigBox(getApplicationURL(), "50px", "50px");
  Navigator * nav = new Navigator(getApplicationURL(), "210px", "50px");
  ContentViewer * cont = new ContentViewer(getApplicationURL(), "340px", "50px");
  GifDisplay * dis = new GifDisplay(getApplicationURL(), "50px","370px", "270px", "550px", "MyGifDisplay"); 
GifDisplay * dis2 = new GifDisplay(getApplicationURL(), "370px", "370px", "270px", "550px", "MyOtherGifDisplay");

  Button * subcrBut = new Button(getApplicationURL(), "470px", "50px", "SubscribeAll", "Subscribe All");
  Button * compBut = new Button(getApplicationURL(), "520px", "50px", "CompareWithRef", "Compare with Reference");
  Button * tkMapBut = new Button(getApplicationURL(), "570px", "50px", "CreateTrackerMap", "Create TrackerMap");
  Button * sumBut = new Button(getApplicationURL(), "620px", "50px", "CreateSummary", "Create Summary");
  Button * saveBut = new Button(getApplicationURL(), "670px", "50px", "SaveToFile", "Save To File");


  page = new WebPage(getApplicationURL());
  page->add("configBox", box);
  page->add("navigator", nav);
  page->add("contentViewer", cont);
  page->add("gifDisplay", dis);
  page->add("otherGifDisplay", dis2);
  page->add("Sbbutton", subcrBut);
  page->add("Cbutton", compBut);
  page->add("Tbutton", tkMapBut);
  page->add("Smbutton", sumBut);
  page->add("SvButton", saveBut);


}
//
// --  Destructor
// 
SiStripWebClient::~SiStripWebClient() {
 if (theTrackerMap) delete theTrackerMap;
  theTrackerMap = 0;
  if (theQualityTester) delete theQualityTester;
  theQualityTester = 0;
}

//
// -- The method that prints out the webpage
//
void SiStripWebClient::Default(xgi::Input * in, xgi::Output * out ) 
  throw (xgi::exception::Exception)
{
  std::string pagetitle = "SiStripWebClient";
  CgiWriter writer(out, getContextURL());
  writer.output_preamble();
  writer.output_head();
  page->printHTML(out);
  writer.output_finish();
}

// 
// -- A method that responds to WebElements submitting non-default requests (like Buttons)
//
void SiStripWebClient::Request(xgi::Input * in, xgi::Output * out ) 
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
void SiStripWebClient::subscribeAll(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception)
{
  mui->subscribe("Collector/FU0/SiStrip/*");
  return;
}
//
// -- Set Up Quality Test
//
void SiStripWebClient::setupQTest(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception)
{
  std::cout << "A setupQTest request was received" << endl;
  if (theQualityTester == 0) {
    theQualityTester = new SiStripQualityTester();
    theQualityTester->readFromFile("test.txt");
    theQualityTester->setupQTests(mui);
    theQualityTester->attachTests(mui,"*/SiStrip/MechanicalView/*/*/*/*/DigisPerDetector*");
  } else {
    if (getUpdates() > 30) {
      cout << " Checking Comparison results "<< endl;
      theQualityTester->checkTestResults(mui);
    }
  }
  return;
}
//
// -- Create Tracker Map
//
void SiStripWebClient::createTkMap(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception)
{
  std::cout << "A createTkMap request was received" << endl;
  if (theTrackerMap == 0) theTrackerMap = new TrackerMap("Number of Digis per module");
  int updates = getUpdates();
  if (updates == 0) {
    subscribeAll(in, out);
    return;
  } else if (updates < 10) {
    cout << " Not enough updates received !!" << endl;
    return;
  }
  // Create the Quality Test
  if (theQualityTester == 0) setupQTest(in, out);
  // Get the values for the Tracker Map
  mui->cd();
  SiStripWebClient::DetMapType valueMap;
  getValuesForTkMap(valueMap);
  
  std::string comment;
  int rval = 0;
  int gval = 0;
  int bval = 0;
  for (SiStripWebClient::DetMapType::const_iterator it = valueMap.begin();
       it != valueMap.end(); it++) {
    if (it->second.size() < 2) continue;
    cout << " Detector ID : " << it->first 
	 << " Mean Value : " << it->second[1] 
         << " Status : " << it->second[0]  << endl;
    // Fill Tracker Map with color from the status
    if (it->second[0] == "Ok") { 
      rval = 0;   gval = 255;   bval = 0; 
    } else if (it->second[0] == "Warning") { 
      rval = 255; gval = 255; bval = 0;
    } else if (it->second[0] == "Error") { 
      rval = 255; gval = 0;  bval = 0;
    }
    theTrackerMap->fillc(it->first, rval, gval, bval);
    // Fill Tracker Map with Mean Value
    theTrackerMap->fill_current_val(it->first, atof(it->second[1].c_str()));
    // Fill Tracker Map with Mean Value as Comment
    comment =  "Mean value of Digi " + it->second[1];
    theTrackerMap->setText(it->first, comment);
  }
  theTrackerMap->print(true);
  return;
}
//
// -- Create Summary
//
void SiStripWebClient::createSummary(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception) {
  std::cout << "A createSummary request was received" << endl;
  if (getUpdates() > 10) fillSummary("module", "Digi");
  return;
}
//
//
//
void SiStripWebClient::saveToFile(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception) {
  cout << " Saving Monitoring Elements " << endl;
  //  mui->save("SiStripWebClient.root", "Collector/FU0/SiStrip/MechanicalView/",90);
  mui->save("SiStripWebClient.root");
  return;
  
}
//
// -- get # of updates 
//
int SiStripWebClient::getUpdates() {
  if (!mui) return -1;
  int updates = mui->getNumUpdates();
  mui->subscribeNew("*/SiStrip/MechanicalView/*");
  return updates;
}
//
// -- Browse through monitorable and get values need for TrackerMap
//
void SiStripWebClient::getValuesForTkMap(SiStripWebClient::DetMapType & values) {
  std::string currDir = mui->pwd();
  //  cout << " current Dir " << currDir << endl;
  // browse through monitorable; check if MEs exist
  if (currDir.find("detector") != std::string::npos)  {
    TCanvas canvas("display");
    std::string status;
    std::vector<std::string> contents = mui->getMEs();    
    for (std::vector<std::string>::const_iterator it = contents.begin();
	 it != contents.end(); it++) {
      if ((*it).find("DigisPerDetector") == 0) {
	std::string fullpathname = currDir + "/" + (*it); 
        MonitorElement * me = mui->get(fullpathname);
        if (me) {
          // Geometric ID
	  int id = atoi((currDir.substr(currDir.find("detector_")+9)).c_str());
          // Mean Value
	  ostringstream mean_str;
	  mean_str << me->getMean();
          // Status after comparison to Referece 
	  if (me->hasError()) status = "Error";
	  else if (me->hasWarning()) status = "Warning";
	  else  status = "Ok";
          // creation of jpg file
	  canvas.Clear();
	  // Access the Root object
	  MonitorElementT<TNamed>* ob = 
	    dynamic_cast<MonitorElementT<TNamed>*>(me);
	  if (ob) {
	    ob->operator->()->Draw();
	    ostringstream name_str;
	    name_str << id << ".jpg";
	    canvas.SaveAs(name_str.str().c_str());
	  }
          vector<std::string> vtemp;
          vtemp.push_back(status);
          vtemp.push_back(mean_str.str());  
	  values.insert(pair<int,std::vector <std::string> >(id, vtemp));
        }
      }
    }
  } else {
    std::vector<std::string> subdirs = mui->getSubdirs();
    for (std::vector<std::string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      mui->cd(*it);
      getValuesForTkMap(values);
      mui->goUp();
    }
  } 
}
//
// -- Browse through the Folder Structure
//
void SiStripWebClient::fillSummary(std::string name, std::string type) {
  std::string currDir = mui->pwd();
  if (currDir.find(name) != std::string::npos)  {
    std::string tag = "Summary" + type;
    MonitorElement* sum_me = getSummaryME(name, tag);
    if (!sum_me) return;
    std::vector<std::string> subdirs = mui->getSubdirs();
    int ndet = 0;
    for (std::vector<std::string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      if ( (*it).find("detector") == std::string::npos) continue;
      mui->cd(*it);
      ndet++;
      std::vector<std::string> contents = mui->getMEs();    
      for (std::vector<std::string>::const_iterator im = contents.begin();
	 im != contents.end(); im++) {
	if ((*im).find("DigisPerDetector") == 0) {
	  std::string fullpathname = mui->pwd() + "/" + (*im); 
	  MonitorElement *  me = mui->get(fullpathname);
	  if (me) sum_me->Fill(ndet*1.0, me->getMean());
	}
      }
      mui->goUp();
    }    
  } else {  
    std::vector<std::string> subdirs = mui->getSubdirs();
    for (std::vector<std::string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      mui->cd(*it);
      fillSummary(name, type);
      mui->goUp();
    }
  }
}
//
// -- Get Summary ME
//
MonitorElement* SiStripWebClient::getSummaryME(std::string name, std::string tag) {
  MonitorElement* me = 0;
  std::string currDir = mui->pwd();
  if (currDir.find(name) != std::string::npos)  {
    // If already booked
    std::vector<std::string> contents = mui->getMEs();    
    for (std::vector<std::string>::const_iterator it = contents.begin();
	 it != contents.end(); it++) {
      if ((*it).find(tag) == 0) {
	std::string fullpathname = currDir + "/" + (*it); 
        me = mui->get(fullpathname);
        if (me) {
	  MonitorElementT<TNamed>* obh1 = 
                  dynamic_cast<MonitorElementT<TNamed>*> (me);
	  if (obh1) {
	    TH1F * root_obh1 = dynamic_cast<TH1F *> (obh1->operator->());
	    if (root_obh1) root_obh1->Reset();        
	  }
          return me;
        }
      }
    }
    DaqMonitorBEInterface * bei = mui->getBEInterface();
    ostringstream hname;
    hname << tag 
	  << (currDir.substr(currDir.find(name)+name.size())).c_str();
    me = bei->book1D(hname.str().c_str(), hname.str().c_str(),20,0.5,20.5);
    bei->setCurrentFolder(mui->pwd().c_str());
    return me;
  }
  return 0;
}
//
// -- provides factory method for instantion of SimpleWeb application
//
XDAQ_INSTANTIATOR_IMPL(SiStripWebClient)
  
