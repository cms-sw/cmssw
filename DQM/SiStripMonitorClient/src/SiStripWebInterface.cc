#include "DQM/SiStripMonitorClient/interface/SiStripWebInterface.h"
#include "DQM/SiStripMonitorClient/interface/SiStripInformationExtractor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMScope.h"
#include "DQMServices/WebComponents/interface/CgiWriter.h"
#include "DQMServices/WebComponents/interface/CgiReader.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"


#include <SealBase/Callback.h>
#include <map>
#include <iostream>
#include <sstream>

#define BUF_SIZE 256
//
// -- Constructor
// 
SiStripWebInterface::SiStripWebInterface(DQMStore* dqm_store) : dqmStore_(dqm_store) {
  
  theActionFlag = NoAction;
  infoExtractor_  = 0;
  if (infoExtractor_ == 0) infoExtractor_ = new SiStripInformationExtractor();
}
//
// --  Destructor
// 
SiStripWebInterface::~SiStripWebInterface() {
  if (infoExtractor_) delete infoExtractor_;
  infoExtractor_ = 0; 
}
// 
// -- Handles requests from WebElements submitting non-default requests 
//
void SiStripWebInterface::handleAnalyserRequest(xgi::Input* in,xgi::Output* out, const edm::ESHandle<SiStripDetCabling>& detcabling, int niter) {
  DQMScope enter;
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
    std::string infoType = get_from_multimap(requestMap_, "InfoType");
    infoExtractor_->readQTestSummary(dqmStore_, infoType, detcabling, out);
    theActionFlag = NoAction;
  } 
  else if (requestID == "SingleModuleHistoList") {
    theActionFlag = NoAction;

    infoExtractor_->readModuleAndHistoList(dqmStore_, detcabling, out);
  } 
  else if (requestID == "GlobalHistoList") {
    theActionFlag = NoAction;
    std::string gname = get_from_multimap(requestMap_, "GlobalFolder");    
    infoExtractor_->readGlobalHistoList(dqmStore_, gname, out);
  } 
  else if (requestID == "SummaryHistoList") {
    theActionFlag = NoAction;
    std::string sname = get_from_multimap(requestMap_, "StructureName");
    infoExtractor_->readSummaryHistoTree(dqmStore_, sname, out);
  } 
  else if (requestID == "AlarmList") {
    theActionFlag = NoAction;
    std::string sname = get_from_multimap(requestMap_, "StructureName");
    infoExtractor_->readAlarmTree(dqmStore_, sname, out);
  } 
  else if (requestID == "ReadQTestStatus") {
    theActionFlag = NoAction;
    infoExtractor_->readStatusMessage(dqmStore_, requestMap_, out);
  } 
   else if (requestID == "PlotAsModule") {
    theActionFlag = NoAction;  
    infoExtractor_->getSingleModuleHistos(dqmStore_, requestMap_, out);    
  }
  else if (requestID == "PlotGlobalHisto") {
    theActionFlag = NoAction;
    infoExtractor_->getGlobalHistos(dqmStore_, requestMap_, out);    
  }
  else if (requestID == "PlotHistogramFromPath") {
   theActionFlag = NoAction;
   infoExtractor_->getHistosFromPath(dqmStore_, requestMap_, out);    
  } 
  else if (requestID == "PlotTkMapHistogram") {
    theActionFlag = NoAction;
    infoExtractor_->getTrackerMapHistos(dqmStore_, requestMap_, out);
  }
  else if (requestID == "PlotHistogramFromLayout") {
    theActionFlag = PlotHistogramFromLayout;
  } 
  else if (requestID == "GetIMGCImage") { 
   infoExtractor_->getIMGCImage(requestMap_, out);
  }
  else if (requestID == "GetTkMap") { 
    theActionFlag = NoAction;    
    
    ifstream fin("dqmtmapviewer.xhtml");
    char buf[BUF_SIZE];
    std::ostringstream html_out;
    if (!fin) {
      std::cerr << "Input File: dqmtmapviewer.xhtml "<< " could not be opened!" << std::endl;
      return;
    }
    while (fin.getline(buf, BUF_SIZE, '\n')) { // pops off the newline character 
      html_out << buf << std::endl;
    }
    fin.close();
    
    out->getHTTPResponseHeader().addHeader("Content-type","application/xhtml+xml");
    *out << html_out.str();
  }
  else if (requestID == "NonGeomHistoList") {
    theActionFlag = NoAction;
    std::string fname = get_from_multimap(requestMap_, "FolderName");
    infoExtractor_->readNonGeomHistoTree(dqmStore_, fname, out);
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
      //      actionExecutor_->createSummary(dqmStore_);
      break;
    }
  case SiStripWebInterface::PlotSingleModuleHistos :
    {
//      infoExtractor_->plotSingleModuleHistos(dqmStore_, requestMap_, out);
      break;
    }
  case SiStripWebInterface::PlotGlobalHistos :
    {
//      infoExtractor_->plotGlobalHistos(dqmStore_, requestMap_, out);
      break;
    }
  case SiStripWebInterface::PlotTkMapHistogram :
    {
//      infoExtractor_->plotTrackerMapHistos(dqmStore_, requestMap_, out);
      break;
    }
  case SiStripWebInterface::PlotHistogramFromPath :
    {
//       infoExtractor_->plotHistosFromPath(dqmStore_, requestMap_, out);
      break;
    }
  case SiStripWebInterface::PlotHistogramFromLayout :
    {
      infoExtractor_->plotHistosFromLayout(dqmStore_);
      break;
    }
  case SiStripWebInterface::CreatePlots :
    {
      infoExtractor_->createImages(dqmStore_);
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
