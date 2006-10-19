#include "DQM/SiPixelMonitorClient/interface/SiPixelWebInterface.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelActionExecutor.h"

#include "DQMServices/WebComponents/interface/Button.h"
#include "DQMServices/WebComponents/interface/CgiWriter.h"
#include "DQMServices/WebComponents/interface/CgiReader.h"
#include "DQMServices/WebComponents/interface/ConfigBox.h"
#include "DQMServices/WebComponents/interface/Navigator.h"
#include "DQMServices/WebComponents/interface/ContentViewer.h"
#include "DQMServices/WebComponents/interface/GifDisplay.h"
#include "DQMServices/WebComponents/interface/Select.h"

#include <SealBase/Callback.h>
#include <map>
#include <iostream>


/*
  Create your widgets in the constructor of your web interface
*/
SiPixelWebInterface::SiPixelWebInterface(std::string theContextURL, std::string theApplicationURL, MonitorUserInterface ** _mui_p)
  : WebInterface(theContextURL, theApplicationURL, _mui_p)
{
  theActionFlag = NoAction;
  actionExecutor_ = 0;
  tkMapOptions_.push_back("Persistant");
  tkMapOptions_.push_back("Temporary");
  
  createAll();

  if (actionExecutor_ == 0) actionExecutor_ = new SiPixelActionExecutor();
}


//
// -- Create default and customised Widgets
// 
void SiPixelWebInterface::createAll() { 
  Navigator * nav = new Navigator(getApplicationURL(), "50px", "50px");
  ContentViewer * cont = new ContentViewer(getApplicationURL(), "180px", "50px");
  GifDisplay * dis = new GifDisplay(getApplicationURL(), "25px","300px", "400px", "550px", "MyGifDisplay"); 
  
  Button * subcrBut = new Button(getApplicationURL(), "320px", "50px", "SubscribeAll", "Subscribe All");
  Button * compBut = new Button(getApplicationURL(), "360px", "50px", "CheckQTResults", "Check QTest Results");
  Button * sumBut = new Button(getApplicationURL(), "400px", "50px", "CreateSummary", "Create Summary");
  Button * collBut = new Button(getApplicationURL(), "440px", "50px", "CollateME", "Collate ME");
  /*  Button * tkMapBut1 = new Button(getApplicationURL(), "480px", "50px", "CreateTrackerMap1", "Create Persistant TrackerMap");
  Button * tkMapBut2 = new Button(getApplicationURL(), "480px", "300px", "CreateTrackerMap2", "Create TempTrackerMap");*/
  Button * saveBut = new Button(getApplicationURL(), "480px", "50px", "SaveToFile", "Save To File");
  
  Select *selTkMap = new Select(getApplicationURL(), "520px", "50px", "SelectTkMap", "Select Tk Map");

  selTkMap->setOptionsVector(tkMapOptions_);

  page_p = new WebPage(getApplicationURL());
  page_p->add("navigator", nav);
  page_p->add("contentViewer", cont);
  page_p->add("gifDisplay", dis);
  page_p->add("Sbbutton", subcrBut);
  page_p->add("Cbutton", compBut);
  page_p->add("Smbutton", sumBut);
  page_p->add("SvButton", saveBut);
  page_p->add("ClButton", collBut);
  /*  page_p->add("Tbutton1", tkMapBut1);
      page_p->add("Tbutton2", tkMapBut2);*/
  page_p->add("Tselect", selTkMap);

}
//
// --  Destructor
// 
SiPixelWebInterface::~SiPixelWebInterface() {
  if (actionExecutor_) delete actionExecutor_;
  actionExecutor_ = 0;
}
// 
// -- Handles requests from WebElements submitting non-default requests 
//
void SiPixelWebInterface::handleCustomRequest(xgi::Input* in,xgi::Output* out)
  throw (xgi::exception::Exception)
{
  // put the request information in a multimap...
  std::multimap<std::string, std::string> request_multimap;
  CgiReader reader(in);
  reader.read_form(request_multimap);

  // get the string that identifies the request:
  std::string requestID = get_from_multimap(request_multimap, "RequestID");
  cout << " requestID " << requestID << endl;
  if (requestID == "SubscribeAll") {
    theActionFlag = SubscribeAll;
  } else if (requestID == "CheckQTResults") {
    theActionFlag = QTestResult;
  } else if (requestID == "CreateSummary") {
     theActionFlag = Summary;
  } else if (requestID == "SaveToFile") {
     theActionFlag = SaveData;
  } else if (requestID == "CollateME") {
     theActionFlag = Collate;
     /*  } else if (requestID == "CreateTrackerMap1") {
     theActionFlag = PersistantTkMap;
  } else if (requestID == "CreateTrackerMap2") {
  theActionFlag = TemporaryTkMap;*/
  } else if (requestID == "SelectTkMap") {
    std::multimap<std::string, std::string> selection_multimap;
    std::string choice;
    readSelectedRequest(in, out, choice);
    if (choice == tkMapOptions_[0]) theActionFlag = PersistantTkMap;
    else if (choice == tkMapOptions_[1]) theActionFlag = TemporaryTkMap;
  }
  configureCustomRequest(in, out);
}
//
// -- Read the option specified in the Select widget
//
void SiPixelWebInterface::readSelectedRequest(xgi::Input * in, xgi::Output * out, std::string& choice) throw (xgi::exception::Exception){
  std::multimap<std::string, std::string> selection_multimap;
  CgiReader reader(in);
  reader.read_form(selection_multimap);
  choice = get_from_multimap(selection_multimap, "Argument");

}
//
// -- Schedule Custom Action
//
void SiPixelWebInterface::configureCustomRequest(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception){
  seal::Callback action(seal::CreateCallback(this, 
                      &SiPixelWebInterface::performAction));
  (*mui_p)->addCallback(action);
}
//
// -- Setup Quality Tests
// 
void SiPixelWebInterface::setupQTests() {
//  actionExecutor_->setupQTests((*mui_p));
}
//
// -- Read Configurations 
//
void SiPixelWebInterface::readConfiguration(int& tkmap_freq, int& sum_freq){
  if (actionExecutor_)  {
    if (actionExecutor_->readConfiguration(tkmap_freq,sum_freq));
  } else {
    tkmap_freq = -1;
    sum_freq   = -1;
  }
}
//
// -- Perform action
//
void SiPixelWebInterface::performAction() {
  switch (theActionFlag) {
  case SiPixelWebInterface::SubscribeAll :
    {
      cout << " SiPixelWebInterface::subscribeAll " << endl;
      (*mui_p)->subscribe("Collector/*");
       setActionFlag(SiPixelWebInterface::NoAction);
      break;
    } 
  case SiPixelWebInterface::Collate :
    {
      actionExecutor_->createCollation((*mui_p));
      setActionFlag(SiPixelWebInterface::NoAction);
      break;
    }
  case SiPixelWebInterface::PersistantTkMap :
    {
      system("rm -rf tkmap_files_old/*.jpg; rm -rf  tkmap_files_old/*.svg");
      system("rm -rf tkmap_files_old");
      system("mv tkmap_files tkmap_files_old");
      system("mkdir -p tkmap_files");
      actionExecutor_->createTkMap((*mui_p));
      system(" mv *.jpg tkmap_files/. ; mv *.svg tkmap_files/.");
      setActionFlag(SiPixelWebInterface::NoAction);
      break;
    }
  case SiPixelWebInterface::TemporaryTkMap :
    {
      system("mkdir -p tkmap_files");
      system("rm -rf tkmap_files/*.jpg; rm -rf tkmap_files/*.svg");
      actionExecutor_->createTkMap((*mui_p));
      system(" mv *.jpg tkmap_files/. ; mv *.svg tkmap_files/.");
      setActionFlag(SiPixelWebInterface::NoAction);
      break;
    }
  case SiPixelWebInterface::Summary :
    {
      actionExecutor_->createSummary((*mui_p));
      setActionFlag(SiPixelWebInterface::NoAction);
      break;
    }
  case SiPixelWebInterface::QTestResult :
    {
      //actionExecutor_->checkQTestResults((*mui_p));
      setActionFlag(SiPixelWebInterface::NoAction);
      break;
    }
  case SiPixelWebInterface::SaveData :
    {
      cout << " Saving Monitoring Elements " << endl;
      //  (*mui_p)->save("SiPixelWebInterface.root", "Collector/Collated",90);
      (*mui_p)->save("SiPixelWebInterface.root");
      setActionFlag(SiPixelWebInterface::NoAction);
      break;
    }
  case SiPixelWebInterface::NoAction :
    {
      break;
    }
  }
}
