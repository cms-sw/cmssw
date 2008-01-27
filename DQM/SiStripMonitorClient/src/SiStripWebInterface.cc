#include "DQM/SiStripMonitorClient/interface/SiStripWebInterface.h"
#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutorQTest.h"
#include "DQM/SiStripMonitorClient/interface/SiStripInformationExtractor.h"
#include "DQMServices/WebComponents/interface/CgiWriter.h"
#include "DQMServices/WebComponents/interface/CgiReader.h"


#include <SealBase/Callback.h>
#include <map>
#include <iostream>
//
// -- Constructor
// 
SiStripWebInterface::SiStripWebInterface(DaqMonitorBEInterface* bei) : bei_(bei) {
  
  theActionFlag = NoAction;
  actionExecutor_ = 0;
  infoExtractor_  = 0;

  if (actionExecutor_ == 0) actionExecutor_ = new SiStripActionExecutorQTest();
  if (infoExtractor_ == 0) infoExtractor_ = new SiStripInformationExtractor();
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
// -- Read Configurations and access the frequency
//
bool SiStripWebInterface::readConfiguration(){
  if (actionExecutor_)
    return (actionExecutor_->readConfiguration());
  else return false;
}
// 
// -- Handles requests from WebElements submitting non-default requests 
//
void SiStripWebInterface::handleAnalyserRequest(xgi::Input* in,xgi::Output* out, int niter) {
  // put the request information in a multimap...
  //  std::multimap<std::string, std::string> requestMap_;
  CgiReader reader(in);
  reader.read_form(requestMap_);
  std::string requestID = get_from_multimap(requestMap_, "RequestID");
  // get the string that identifies the request:
  std::cout << " requestID " << requestID << std::endl;
  if (requestID == "IsReady") {
    theActionFlag = NoAction;    
    if (niter > 0) {
      infoExtractor_->readLayoutNames(requestMap_, out);
    } else {
      returnReplyXml(out, "ReadyState", "wait");
    }
  }    
  else if (requestID == "CheckQTResults") {
    out->getHTTPResponseHeader().addHeader("Content-Type", "text/plain");
    std::string infoType = get_from_multimap(requestMap_, "InfoType");
    if (infoType == "Lite") *out <<  actionExecutor_->getQTestSummaryLite(bei_) << std::endl;
    else *out <<  actionExecutor_->getQTestSummary(bei_) << std::endl;
    theActionFlag = NoAction;
  } 
  else if (requestID == "SingleModuleHistoList") {
    theActionFlag = NoAction;
    
    infoExtractor_->readModuleAndHistoList(bei_, out);
  } 
  else if (requestID == "GlobalHistoList") {
    theActionFlag = NoAction;
    
    infoExtractor_->readGlobalHistoList(bei_, out);
  } 
  else if (requestID == "SummaryHistoList") {
    theActionFlag = NoAction;
    std::string sname = get_from_multimap(requestMap_, "StructureName");
    infoExtractor_->readSummaryHistoTree(bei_, sname, out);
  } 
  else if (requestID == "AlarmList") {
    theActionFlag = NoAction;
    std::string sname = get_from_multimap(requestMap_, "StructureName");
    infoExtractor_->readAlarmTree(bei_, sname, out);
  } 
  else if (requestID == "ReadQTestStatus") {
    theActionFlag = NoAction;
    infoExtractor_->readStatusMessage(bei_, requestMap_, out);
  } 
   else if (requestID == "PlotAsModule") {
    theActionFlag = NoAction;  
    infoExtractor_->plotSingleModuleHistos(bei_, requestMap_, out);    
  }
  else if (requestID == "PlotGlobalHisto") {
    theActionFlag = NoAction;    
    infoExtractor_->plotGlobalHistos(bei_, requestMap_, out);    
  }
  else if (requestID == "PlotHistogramFromPath") {
   theActionFlag = NoAction;
   infoExtractor_->plotHistosFromPath(bei_, requestMap_, out);    
  } 
  else if (requestID == "PlotTkMapHistogram") {
    theActionFlag = NoAction;
    infoExtractor_->plotTrackerMapHistos(bei_, requestMap_, out);
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
  else if (requestID == "updateIMGCPlots") {
    theActionFlag = NoAction;    
    std::string MEFolder = get_from_multimap(requestMap_, "MEFolder");
    std::cout << "SiStripWebInterface::handleAnalyserRequest : "
         << "Collecting ME from folder " 
         << MEFolder
         << std::endl ;
    out->getHTTPResponseHeader().addHeader("Content-Type", "text/html");
    out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
    out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
    out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
    bei_->cd() ;
    bei_->cd(MEFolder) ;
    std::vector<std::string> meList = bei_->getMEs();
    bei_->cd() ;
    *out << MEFolder << " " ;
    std::cout << "SiStripWebInterface::handleAnalyserRequest "
         << "MEFolder: " << MEFolder << std::endl;
    std::cout << "SiSitrpWebInterface::handleCustomRequest ";
    for(std::vector<std::string>::iterator it=meList.begin(); it!=meList.end(); it++)
    {
     *out  << *it << " " ;
      std::cout << *it << " " ;
    }
    std::cout << std::endl;       
  }
  else if (requestID == "GetIMGCImage") { 
    theActionFlag = NoAction;    
    std::string imageName = get_from_multimap(requestMap_, "ImageName");
    std::cout << requestID << " " << imageName << std::endl;
    out->getHTTPResponseHeader().addHeader("Content-Type", "image/png");
    out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
    out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
    out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
    *out << infoExtractor_->getIMGCImage(bei_, requestMap_).str();
  }
    
  performAction();
}
//
// -- Perform action
//
void SiStripWebInterface::performAction() {
  switch (theActionFlag) {
  case SiStripWebInterface::Summary :
    {
      actionExecutor_->createSummary(bei_);
      break;
    }
  case SiStripWebInterface::PlotSingleModuleHistos :
    {
//      infoExtractor_->plotSingleModuleHistos(bei_, requestMap_, out);
      break;
    }
  case SiStripWebInterface::PlotGlobalHistos :
    {
//      infoExtractor_->plotGlobalHistos(bei_, requestMap_, out);
      break;
    }
  case SiStripWebInterface::PlotTkMapHistogram :
    {
//      infoExtractor_->plotTrackerMapHistos(bei_, requestMap_, out);
      break;
    }
  case SiStripWebInterface::PlotHistogramFromPath :
    {
//       infoExtractor_->plotHistosFromPath(bei_, requestMap_, out);
      break;
    }
  case SiStripWebInterface::PlotHistogramFromLayout :
    {
      infoExtractor_->plotHistosFromLayout(bei_);
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
  *out << "<"<<name<<">" << std::endl;
  *out << " <Response>" << comment << "</Response>" << std::endl;
  *out << "</"<<name<<">" << std::endl;
}
//
// Get the RequestId and tags 
//
std::string SiStripWebInterface::get_from_multimap(std::multimap<std::string, std::string> &mymap, std::string key)
{
  std::multimap<std::string, std::string>::iterator it;
  it = mymap.find(key);
  if (it != mymap.end())
    {
      return (it->second);
    }
  return "";
}
