#include "DQM/SiStripMonitorClient/interface/SiStripWebInterface.h"
#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "DQM/SiStripMonitorClient/interface/SiStripInformationExtractor.h"
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
//
// -- Constructor
// 
SiStripWebInterface::SiStripWebInterface(std::string theContextURL, std::string theApplicationURL, MonitorUserInterface ** _mui_p) 
  : WebInterface(theContextURL, theApplicationURL, _mui_p) {
  
  theActionFlag = NoAction;
  actionExecutor_ = 0;
  infoExtractor_  = 0;
  tkMapOptions_.push_back("Persistant");
  tkMapOptions_.push_back("Temporary");
  tkMapCreated = false;
  createAll();

  if (actionExecutor_ == 0) actionExecutor_ = new SiStripActionExecutor();
  if (infoExtractor_ == 0) infoExtractor_ = new SiStripInformationExtractor();
}
//
// -- Create default and customised Widgets
// 
void SiStripWebInterface::createAll() { 
  Navigator * nav = new Navigator(getApplicationURL(), "50px", "50px");
  ContentViewer * cont = new ContentViewer(getApplicationURL(), "180px", "50px");
  GifDisplay * dis = new GifDisplay(getApplicationURL(), "25px","300px", "500px", "600px", "MyGifDisplay"); 
  

  /*  Button * subcrBut = new Button(getApplicationURL(), "340px", "50px", "SubscribeAll", "Subscribe All");
  Button * compBut = new Button(getApplicationURL(), "380px", "50px", "CheckQTResults", "Check QTest Results");
  Button * sumBut = new Button(getApplicationURL(), "420px", "50px", "UpdateSummary", "Create Summary");
  Button * collBut = new Button(getApplicationURL(), "460px", "50px", "CollateME", "Collate ME");
  Button * saveBut = new Button(getApplicationURL(), "500px", "50px", "SaveToFile", "Save To File");
  
  Select *selTkMap = new Select(getApplicationURL(), "540px", "50px", "SelectTkMap", "Select Tk Map");

  selTkMap->setOptionsVector(tkMapOptions_); */

  page_p = new WebPage(getApplicationURL());
  page_p->add("navigator", nav);
  page_p->add("contentViewer", cont);
  page_p->add("gifDisplay", dis);
  /*  page_p->add("Sbbutton", subcrBut);
  page_p->add("Cbutton", compBut);
  page_p->add("Smbutton", sumBut);
  page_p->add("SvButton", saveBut);
  page_p->add("ClButton", collBut);
  page_p->add("Tselect", selTkMap); */

}
//
// --  Destructor
// 
SiStripWebInterface::~SiStripWebInterface() {
  if (actionExecutor_) delete actionExecutor_;
  actionExecutor_ = 0;
  if (infoExtractor_) delete infoExtractor_;
  infoExtractor_ = 0; 
}
// 
// -- Handles requests from WebElements submitting non-default requests 
//
void SiStripWebInterface::handleCustomRequest(xgi::Input* in,xgi::Output* out)
  throw (xgi::exception::Exception)
{
  // put the request information in a multimap...
  //  std::multimap<std::string, std::string> requestMap_;
   CgiReader reader(in);
  reader.read_form(requestMap_);
  std::string requestID = get_from_multimap(requestMap_, "RequestID");
  // get the string that identifies the request:
  cout << " requestID " << requestID << endl;
  if (requestID == "SubscribeAll") {
    theActionFlag = SubscribeAll;
  } 
  else if (requestID == "CheckQTResults") {
    theActionFlag = QTestResult;
  } 
  else if (requestID == "CreateSummary") {
     theActionFlag = Summary;
  } 
  else if (requestID == "SaveToFile") {
     theActionFlag = SaveData;
  } 
  else if (requestID == "CollateME") {
     theActionFlag = Collate;
  } 
  else if (requestID == "SelectTkMap") {
    std::string choice = get_from_multimap(requestMap_, "Argument");
    if (choice == tkMapOptions_[0]) theActionFlag = PersistantTkMap;
    else if (choice == tkMapOptions_[1]) theActionFlag = TemporaryTkMap;
  } 
  else if (requestID == "OpenTkMap") {
    std::string name = "TkMap";
    std::string comment;
    if (tkMapCreated) comment = "Successful";
    else  comment = "Failed";
    returnReplyXml(out, name, comment);
    theActionFlag = NoAction;    
  } 
  else if (requestID == "SingleModuleHistoList") {
    theActionFlag = NoAction;
    
    infoExtractor_->readModuleAndHistoList((*mui_p), out,
                          actionExecutor_->getCollationFlag() );    
  } 
  else if (requestID == "SummaryHistoList") {
    theActionFlag = NoAction;
    string sname = get_from_multimap(requestMap_, "StructureName");
   infoExtractor_->readSummaryHistoList((*mui_p), sname, out,
                           actionExecutor_->getCollationFlag());    
  } 
  else if (requestID == "PlotAsModule") {
    theActionFlag = PlotSingleModuleHistos;    
  }
  else if (requestID == "PlotSummaryHistos") {
    theActionFlag = PlotSummaryHistos;    
  }
  else if (requestID == "UpdatePlot") {
   out->getHTTPResponseHeader().addHeader("Content-Type", "image/png");
   out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
   out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
   out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
   *out << infoExtractor_->getImage().str();
    theActionFlag = NoAction;    
  }
    
  configureCustomRequest(in, out);
}
//
// -- Scedule Custom Action
//
void SiStripWebInterface::configureCustomRequest(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception){
  seal::Callback action(seal::CreateCallback(this, 
                      &SiStripWebInterface::performAction));
  (*mui_p)->addCallback(action);
}
//
// -- Setup Quality Tests
// 
void SiStripWebInterface::setupQTests() {
  actionExecutor_->setupQTests((*mui_p));
}
//
// -- Read Configurations 
//
void SiStripWebInterface::readConfiguration(int& tkmap_freq, int& sum_freq){
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
void SiStripWebInterface::performAction() {
  switch (theActionFlag) {
  case SiStripWebInterface::SubscribeAll :
    {
      cout << " SiStripWebInterface::subscribeAll " << endl;
      (*mui_p)->subscribe("Collector/*");
      break;
    } 
  case SiStripWebInterface::Collate :
    {
      actionExecutor_->createCollation((*mui_p));
      break;
    }
  case SiStripWebInterface::PersistantTkMap :
    {
     if (createTkMap()) {
       tkMapCreated = true;
     }
      break;
    }
  case SiStripWebInterface::TemporaryTkMap :
    {
     if (createTkMap()) {
       tkMapCreated = true;
      }
      break;
    }
  case SiStripWebInterface::Summary :
    {
      actionExecutor_->createSummary((*mui_p));
      break;
    }
  case SiStripWebInterface::QTestResult :
    {
      actionExecutor_->checkQTestResults((*mui_p));
      break;
    }
  case SiStripWebInterface::SaveData :
    {
      cout << " Saving Monitoring Elements " << endl;
      //  (*mui_p)->save("SiStripWebInterface.root", "Collector/Collated",90);
      (*mui_p)->save("SiStripWebInterface.root","Collector/Collated");
      //      actionExecutor_->saveMEs((*mui_p), "SiStripWebInterface.root");
      break;
    }
  case SiStripWebInterface::PlotSingleModuleHistos :
    {
      infoExtractor_->plotSingleModuleHistos((*mui_p), requestMap_);
      break;
    }
  case SiStripWebInterface::PlotSummaryHistos :
    {
      infoExtractor_->plotSummaryHistos((*mui_p), requestMap_);
      break;
    }
  case SiStripWebInterface::NoAction :
    {
      break;
    }
  }
  setActionFlag(SiStripWebInterface::NoAction);
}
void SiStripWebInterface::createButton(xgi::Output * out, string& href) {

  std::string position = "position:absolute; left:50px; top: 350px;";
  *out << cgicc::div().set("style", position.c_str()) << std::endl;

  std::string js_command = "var w = window.open('" + href +"');w.focus()";
  *out << cgicc::input().set("type", "button")
    .set("value", "Open SiStrip Interface")
    .set("onclick", js_command.c_str()) << std::endl;

  *out << cgicc::div()   << std::endl;
}
void SiStripWebInterface::returnReplyXml(xgi::Output * out, const std::string& name, const std::string& comment){
   out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  *out << "<?xml version=\"1.0\" ?>" << std::endl;
  *out << "<TkMap>" << endl;
  *out << " <Response>" << comment << "</Response>" << endl;
  *out << "</TkMap>" << endl;
  cout <<  "<?xml version=\"1.0\" ?>" << std::endl;
  cout << "<TkMap>" << endl;
  cout << " <Response>" << comment << "</Response>" << endl;
  cout << "</TkMap>" << endl;

}
bool SiStripWebInterface::createTkMap() {
  if (theActionFlag == SiStripWebInterface::TemporaryTkMap) {
    system("mkdir -p tkmap_files");
    system("rm -rf tkmap_files/*.jpg; rm -rf tkmap_files/*.svg");
    actionExecutor_->createTkMap((*mui_p));
    system(" mv *.jpg tkmap_files/. ; mv *.svg tkmap_files/.");
    return true;
  } else if (theActionFlag == SiStripWebInterface::PersistantTkMap) {
    system("rm -rf tkmap_files_old/*.jpg; rm -rf  tkmap_files_old/*.svg");
    system("rm -rf tkmap_files_old");
    system("mv tkmap_files tkmap_files_old");
    system("mkdir -p tkmap_files");
    actionExecutor_->createTkMap((*mui_p));
    system(" mv *.jpg tkmap_files/. ; mv *.svg tkmap_files/.");
    return true;
  } else {
    return false;
  }

}
