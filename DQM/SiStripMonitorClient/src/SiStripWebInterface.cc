#include "DQM/SiStripMonitorClient/interface/SiStripWebInterface.h"
#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutorQTest.h"
#include "DQM/SiStripMonitorClient/interface/SiStripInformationExtractor.h"
#include "DQMServices/WebComponents/interface/Button.h"
#include "DQMServices/WebComponents/interface/CgiWriter.h"
#include "DQMServices/WebComponents/interface/CgiReader.h"
#include "DQMServices/WebComponents/interface/ConfigBox.h"
#include "DQMServices/WebComponents/interface/Navigator.h"
#include "DQMServices/WebComponents/interface/ContentViewer.h"
#include "DQMServices/WebComponents/interface/GifDisplay.h"
#include "DQMServices/WebComponents/interface/Select.h"
#include "DQMServices/WebComponents/interface/HTMLLink.h"

#include <SealBase/Callback.h>
#include <map>
#include <iostream>
using namespace std;
//
// -- Constructor
// 
SiStripWebInterface::SiStripWebInterface(std::string theContextURL, std::string theApplicationURL, MonitorUserInterface ** _mui_p) 
  : WebInterface(theContextURL, theApplicationURL, _mui_p) {
  
  theActionFlag = NoAction;
  actionExecutor_ = 0;
  infoExtractor_  = 0;
  tkMapCreated = false;
  fileName_ = "SiStripWebInterface.root";

  if (actionExecutor_ == 0) actionExecutor_ = new SiStripActionExecutorQTest();
  if (infoExtractor_ == 0) infoExtractor_ = new SiStripInformationExtractor();
}
//
// -- Create default and customised Widgets
// 
void SiStripWebInterface::createAll() { 
  Navigator * nav = new Navigator(getApplicationURL(), "50px", "50px");
  ContentViewer * cont = new ContentViewer(getApplicationURL(), "180px", "50px");
  GifDisplay * dis = new GifDisplay(getApplicationURL(), "25px","300px", "500px", "600px", "MyGifDisplay"); 
  // an html link
  HTMLLink *link = new HTMLLink(getApplicationURL(), "380px", "50px", 
				"<i>SiStripWebInterface</i>", 
				"/temporary/Online.html");
  
  page_p = new WebPage(getApplicationURL());
  page_p->add("navigator", nav);
  page_p->add("contentViewer", cont);
  page_p->add("gifDisplay", dis);
  page_p->add("htmlLink", link);
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
  DaqMonitorBEInterface* bei = (*mui_p)->getBEInterface();
  // put the request information in a multimap...
  //  std::multimap<std::string, std::string> requestMap_;
   CgiReader reader(in);
  reader.read_form(requestMap_);
  std::string requestID = get_from_multimap(requestMap_, "RequestID");
  // get the string that identifies the request:
  cout << " requestID " << requestID << endl;
  if (requestID == "IsReady") {
    theActionFlag = NoAction;    
    if ((*mui_p)->getNumUpdates() > 2) {
      infoExtractor_->readLayoutNames(out);
    } else {
      returnReplyXml(out, "ReadyState", "wait");
    }
  }    
    else if (requestID == "SubscribeAll") {
    theActionFlag = SubscribeAll;
  } 
  else if (requestID == "CheckQTResults") {
    out->getHTTPResponseHeader().addHeader("Content-Type", "text/plain");
    std::string infoType = get_from_multimap(requestMap_, "InfoType");
    if (infoType == "Lite") *out <<  actionExecutor_->getQTestSummaryLite(bei) << endl;
    else *out <<  actionExecutor_->getQTestSummary(bei) << endl;
    theActionFlag = NoAction;
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
  //  else if (requestID == "CreateTkMap") {
  //     theActionFlag = CreateTkMap;
  //  } 
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
    
    infoExtractor_->readModuleAndHistoList(bei, out,
                          actionExecutor_->getCollationFlag() );    
  } 
  else if (requestID == "GlobalHistoList") {
    theActionFlag = NoAction;
    
    infoExtractor_->readGlobalHistoList(bei, out,
                          actionExecutor_->getCollationFlag() );    
  } 
  else if (requestID == "SummaryHistoList") {
    theActionFlag = NoAction;
    string sname = get_from_multimap(requestMap_, "StructureName");
   infoExtractor_->readSummaryHistoTree(bei, sname, out,
                           actionExecutor_->getCollationFlag());    
  } 
  else if (requestID == "AlarmList") {
    theActionFlag = NoAction;
    string sname = get_from_multimap(requestMap_, "StructureName");
    infoExtractor_->readAlarmTree(bei, sname, out,
                           actionExecutor_->getCollationFlag());    
  } 
  else if (requestID == "ReadQTestStatus") {
    theActionFlag = NoAction;
    string path = get_from_multimap(requestMap_, "Path");
    infoExtractor_->readStatusMessage(bei, path, out);
  } 
  else if (requestID == "PlotAsModule") {
    theActionFlag = PlotSingleModuleHistos;    
  }
  else if (requestID == "PlotGlobalHisto") {
    theActionFlag = PlotGlobalHistos;    
  }
  else if (requestID == "PlotHistogramFromPath") {
   theActionFlag = PlotHistogramFromPath;
  } 
  else if (requestID == "PlotTkMapHistogram") {
    theActionFlag = PlotTkMapHistogram;
  }
  else if (requestID == "PlotHistogramFromLayout") {
    theActionFlag = PlotHistogramFromLayout;
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
// -- Handles requests from WebElements submitting non-default requests 
//
void SiStripWebInterface::handleAnalyserRequest(xgi::Input* in,xgi::Output* out, int niter) {
  DaqMonitorBEInterface* bei = (*mui_p)->getBEInterface();
  // put the request information in a multimap...
  //  std::multimap<std::string, std::string> requestMap_;
  CgiReader reader(in);
  reader.read_form(requestMap_);
  std::string requestID = get_from_multimap(requestMap_, "RequestID");
  // get the string that identifies the request:
  cout << " requestID " << requestID << endl;
  if (requestID == "IsReady") {
    theActionFlag = NoAction;    
    if (niter > 2) {
      infoExtractor_->readLayoutNames(out);
    } else {
      returnReplyXml(out, "ReadyState", "wait");
    }
  }    
  else if (requestID == "CheckQTResults") {
    out->getHTTPResponseHeader().addHeader("Content-Type", "text/plain");
    std::string infoType = get_from_multimap(requestMap_, "InfoType");
    if (infoType == "Lite") *out <<  actionExecutor_->getQTestSummaryLite(bei) << endl;
    else *out <<  actionExecutor_->getQTestSummary(bei) << endl;
    theActionFlag = NoAction;
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
    
    infoExtractor_->readModuleAndHistoList(bei, out,
                          actionExecutor_->getCollationFlag() );    
  } 
  else if (requestID == "GlobalHistoList") {
    theActionFlag = NoAction;
    
    infoExtractor_->readGlobalHistoList(bei, out,
                          actionExecutor_->getCollationFlag() );    
  } 
  else if (requestID == "SummaryHistoList") {
    theActionFlag = NoAction;
    string sname = get_from_multimap(requestMap_, "StructureName");
   infoExtractor_->readSummaryHistoTree(bei, sname, out,
                           actionExecutor_->getCollationFlag());    
  } 
  else if (requestID == "AlarmList") {
    theActionFlag = NoAction;
    string sname = get_from_multimap(requestMap_, "StructureName");
    infoExtractor_->readAlarmTree(bei, sname, out,
                           actionExecutor_->getCollationFlag());    
  } 
  else if (requestID == "ReadQTestStatus") {
    theActionFlag = NoAction;
    string path = get_from_multimap(requestMap_, "Path");
    infoExtractor_->readStatusMessage(bei, path, out);
  } 
  else if (requestID == "PlotAsModule") {
    theActionFlag = PlotSingleModuleHistos;    
  }
  else if (requestID == "PlotGlobalHisto") {
    theActionFlag = PlotGlobalHistos;    
  }
  else if (requestID == "PlotHistogramFromPath") {
   theActionFlag = PlotHistogramFromPath;
  } 
  else if (requestID == "PlotTkMapHistogram") {
    theActionFlag = PlotTkMapHistogram;
  }
  else if (requestID == "PlotHistogramFromLayout") {
    theActionFlag = PlotHistogramFromLayout;
  } 
  else if (requestID == "UpdatePlot") {
    out->getHTTPResponseHeader().addHeader("Content-Type", "image/png");
    out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
    out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
    out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
    *out << infoExtractor_->getImage().str();
    theActionFlag = NoAction;    
  }
    
  performAction();
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
void SiStripWebInterface::readConfiguration(int& sum_freq){
  if (actionExecutor_)  {
    if (actionExecutor_->readConfiguration(sum_freq));
  } else {
    sum_freq   = -1;
  }
}
//
// -- Perform action
//
void SiStripWebInterface::performAction() {
  DaqMonitorBEInterface* bei = (*mui_p)->getBEInterface();
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
  //  case SiStripWebInterface::CreateTkMap :
  //    {
  //     if (createTkMap()) {
  //       tkMapCreated = true;
  //     }
  //      break;
  //    }
  case SiStripWebInterface::Summary :
    {
      actionExecutor_->createSummary(bei);
      break;
    }
  case SiStripWebInterface::SaveData :
    {
      cout << " Saving Monitoring Elements " << endl;
      actionExecutor_->saveMEs(bei, fileName_);
      break;
    }
  case SiStripWebInterface::PlotSingleModuleHistos :
    {
      infoExtractor_->plotSingleModuleHistos(bei, requestMap_);
      break;
    }
  case SiStripWebInterface::PlotGlobalHistos :
    {
      infoExtractor_->plotGlobalHistos(bei, requestMap_);
      break;
    }
  case SiStripWebInterface::PlotTkMapHistogram :
    {
      /*      vector<string> mes;  // commenting out
      int nval = actionExecutor_->getTkMapMENames(mes);
      if (nval == 0) break;
      for  (vector<string>::iterator it = mes.begin();
	    it != mes.end(); it++) {
	requestMap_.insert(pair<string,string>("histo",(*it)));  
      }
      infoExtractor_->plotSingleModuleHistos(bei, requestMap_);*/ 
      infoExtractor_->plotSingleModuleHistos(bei, requestMap_);
      break;
    }
  case SiStripWebInterface::PlotHistogramFromPath :
    {
      infoExtractor_->plotHistosFromPath(bei, requestMap_);
      break;
    }
  case SiStripWebInterface::PlotHistogramFromLayout :
    {
      infoExtractor_->plotHistosFromLayout(bei);
      break;
    }
  case SiStripWebInterface::NoAction :
    {
      break;
    }
  }
  setActionFlag(SiStripWebInterface::NoAction);
}
void SiStripWebInterface::returnReplyXml(xgi::Output * out, const std::string& name, const std::string& comment){
   out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  *out << "<?xml version=\"1.0\" ?>" << std::endl;
  *out << "<"<<name<<">" << endl;
  *out << " <Response>" << comment << "</Response>" << endl;
  *out << "</"<<name<<">" << endl;
  //  cout << "<"<<name<<">" << endl;
  //  cout << " <Response>" << comment << "</Response>" << endl;
  //  cout << "</"<<name<<">" << endl;
}
//bool SiStripWebInterface::createTkMap() {
//  if (theActionFlag == SiStripWebInterface::CreateTkMap) {
//    actionExecutor_->createTkMap(bei);
//    return true;
//  } else {
//    return false;
//  }
//}
