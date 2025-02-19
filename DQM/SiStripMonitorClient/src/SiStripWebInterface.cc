#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiStripMonitorClient/interface/SiStripWebInterface.h"
#include "DQM/SiStripMonitorClient/interface/SiStripInformationExtractor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMScope.h"
#include "DQM/TrackerCommon/interface/CgiWriter.h"
#include "DQM/TrackerCommon/interface/CgiReader.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"


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
  TkMapType = "QTestAlarm";
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
  if (niter < 0) return;
  std::string requestID = get_from_multimap(requestMap_, "RequestID");
  // get the string that identifies the request:
  edm::LogInfo ("SiStripWebInterface") << "SiStripWebInterface::handleAnalyserRequest RequestID = " << requestID ;
  if (requestID == "IsReady") {
    theActionFlag = NoAction;    
    if (niter > 0) {
      infoExtractor_->readLayoutNames(dqmStore_, out);
    } else {
      returnReplyXml(out, "ReadyState", "wait");
    }
  }    
  else if (requestID == "CheckQTResults") {
    std::string infoType = get_from_multimap(requestMap_, "InfoType");
    infoExtractor_->readQTestSummary(dqmStore_, infoType, out);
    theActionFlag = NoAction;
  }
  else if (requestID == "SingleModuleHistoList") {
    theActionFlag = NoAction;
    std::string sname = get_from_multimap(requestMap_, "FolderName");    
    infoExtractor_->readModuleAndHistoList(dqmStore_, sname, detcabling, out);
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
    requestMap_.insert(std::pair<std::string,std::string>("TkMapType", TkMapType));
    infoExtractor_->getTrackerMapHistos(dqmStore_, requestMap_, out);
  }
  else if (requestID == "PlotHistogramFromLayout") {
    theActionFlag = PlotHistogramFromLayout;
  } 
  else if (requestID == "GetImage") { 
   infoExtractor_->getImage(requestMap_, out);
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
  else if (requestID == "PlotModuleCondDBHistos") {    
    theActionFlag = NoAction;
    CondDBPlotParameter local_par;
    uint32_t detId = atoi(get_from_multimap(requestMap_,"ModId").c_str());
    local_par.detId   = detId;
    local_par.type    = "";
    local_par.side    = 999;
    local_par.layer   = 999;   
    bool create_plot;   
    infoExtractor_->getCondDBHistos(dqmStore_, create_plot, requestMap_, out);      
    if (create_plot)  condDBRequestList_.push_back(local_par);
  }
  else if (requestID == "PlotLayerCondDBHistos") {

    theActionFlag = NoAction;
    CondDBPlotParameter local_par;
    std::string sname = get_from_multimap(requestMap_,"StructureName");
    local_par.detId   = 999;
    local_par.type    = sname.substr(sname.find_first_of("/")+1,3);
    if (sname.find("side_")!=std::string::npos) 
            local_par.side = atoi((sname.substr(sname.find("side_")+5,1)).c_str());
    else local_par.side = 999;
    local_par.layer   = atoi((sname.substr(sname.find_last_of("_")+1)).c_str());
    bool create_plot;
    infoExtractor_->getCondDBHistos(dqmStore_, create_plot, requestMap_, out);      
    if (create_plot)  condDBRequestList_.push_back(local_par);

  }  else if (requestID == "UpdateTrackerMapOption") { 
    theActionFlag = NoAction;
    TkMapType = get_from_multimap(requestMap_,"Option");
    returnReplyXml(out, "TkMapOption", TkMapType);
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
//
// -- Get CondDB Parameters
//
void SiStripWebInterface::getConDBPlotParameters(unsigned int ival, uint32_t &det_id, 
		   std::string& subdet_type, uint32_t& subdet_side, uint32_t& layer_number) {
  if (condDBRequestList_.size() > ival) {
    det_id       = condDBRequestList_[ival].detId;
    subdet_type  = condDBRequestList_[ival].type;
    subdet_side  = condDBRequestList_[ival].side;
    layer_number = condDBRequestList_[ival].layer;
  } else {
    det_id       = 999;
    subdet_type  = "";
    subdet_side  = 999;
    layer_number = 999;
  }

}
