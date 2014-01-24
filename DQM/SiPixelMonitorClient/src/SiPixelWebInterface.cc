#include "DQM/SiPixelMonitorClient/interface/SiPixelWebInterface.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelActionExecutor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelInformationExtractor.h"
#include "DQM/SiPixelMonitorClient/interface/ANSIColors.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMScope.h"
#include "DQM/TrackerCommon/interface/CgiWriter.h"
#include "DQM/TrackerCommon/interface/CgiReader.h"

#include <map>
#include <iostream>
#include <sstream>

#define BUF_SIZE 256

using namespace std ;


//____________________________________________________________________________________________________
SiPixelWebInterface::SiPixelWebInterface(DQMStore* bei, 
                                         bool offlineXMLfile, 
					 bool Tier0Flag) : 
					 bei_(bei), 
					 offlineXMLfile_(offlineXMLfile), 
					 Tier0Flag_(Tier0Flag) {
  
  theActionFlag = NoAction;
  actionExecutor_ = 0;
  infoExtractor_  = 0;
  tkMapOptions_.push_back("Persistant");
  tkMapOptions_.push_back("Temporary");
  tkMapCreated = false;
  if (actionExecutor_ == 0) actionExecutor_ = new SiPixelActionExecutor(offlineXMLfile_, Tier0Flag_);
  if (infoExtractor_ == 0) infoExtractor_ = new SiPixelInformationExtractor(offlineXMLfile_);
}

//____________________________________________________________________________________________________
//
// --  Destructor
// 
SiPixelWebInterface::~SiPixelWebInterface() {
  if (actionExecutor_) delete actionExecutor_;
  actionExecutor_ = 0;
  if (infoExtractor_)  delete infoExtractor_;
  infoExtractor_  = 0; 
}

//____________________________________________________________________________________________________
// 
// -- Handles requests from WebElements submitting non-default requests 
//
void SiPixelWebInterface::handleEDARequest(xgi::Input* in,xgi::Output* out, int niter, bool isUpgrade) {
  DQMScope enter;
  //DQMStore* bei = (*mui_p)->getBEInterface();
  CgiReader reader(in);
  reader.read_form(requestMap_);
  // get the string that identifies the request:
  std::string requestID = get_from_multimap(requestMap_, "RequestID");
  cout << ACGreen << ACBold << ACReverse 
       << "[SiPixelWebInterface::handleEDARequest]"
       << ACCyan 
       << " requestID " 
       << ACPlain
       << requestID << endl;

  if (requestID == "IsReady") {
    theActionFlag = NoAction;    
    returnReplyXml(out, "ReadyState", "wait");
  } else if (requestID == "CheckQTResults") {
   theActionFlag = QTestResult;
  } else if (requestID == "updateIMGCPlots") {	  // <-----------------
    theActionFlag = NoAction;	 
    std::string MEFolder = get_from_multimap(requestMap_, "MEFolder");
    //cout << ACYellow << ACBold
    //     << "[SiPixelWebInterface::handleEDARequest] "
    //	 << ACPlain
    //	 << "Collecting ME from folder " 
    //	 << MEFolder
    //	 << endl ;
    out->getHTTPResponseHeader().addHeader("Content-Type", "text/html");
    out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
    out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
    out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
    bei_->cd() ;
    bei_->cd(MEFolder) ;

    vector<std::string> meList = bei_->getMEs() ;
    
    *out << MEFolder << " " ;
    bei_->cd() ;
    for(vector<std::string>::iterator it=meList.begin(); it!=meList.end(); it++)
    {
     *out << *it << " " ;
    }

  } else if (requestID == "getIMGCPlot") {	  // <-----------------
   infoExtractor_->getIMGCImage(requestMap_, out);

  } else if (requestID == "SetupQTest") {	  // <-----------------
    theActionFlag = setupQTest;

  } else if (requestID == "CreateSummary") {	  // <-----------------
    theActionFlag = Summary;

  } else if (requestID == "CreateTkMap") {
    std::string name = "TkMap";
    std::string comment;
    if (tkMapCreated) comment = "Successful";
    else  comment = "Failed";
    returnReplyXml(out, name, comment);
    theActionFlag = CreateTkMap;    
  } else if (requestID == "SingleModuleHistoList") {
    theActionFlag = NoAction;
    infoExtractor_->readModuleAndHistoList(bei_, out);    
  } else if (requestID == "ModuleHistoList") {
    theActionFlag = NoAction;
    string sname = get_from_multimap(requestMap_, "StructureName");
    //cout<<"in EDARequest: structure name= "<<sname<<endl;
    infoExtractor_->readModuleHistoTree(bei_, sname, out);    
  } else if (requestID == "SummaryHistoList") {
    theActionFlag = NoAction;
    string sname = get_from_multimap(requestMap_, "StructureName");
    infoExtractor_->readSummaryHistoTree(bei_, sname, out);    
  } else if (requestID == "AlarmList") {
    theActionFlag = NoAction;
    string sname = get_from_multimap(requestMap_, "StructureName");
    infoExtractor_->readAlarmTree(bei_, sname, out);    
  } else if (requestID == "ReadQTestStatus") {
    theActionFlag = NoAction;
    //string path = get_from_multimap(requestMap_, "Path");
    //infoExtractor_->readStatusMessage(bei_, path, out);
    infoExtractor_->readStatusMessage(bei_, requestMap_, out);
  } else if (requestID == "PlotAsModule") {
    //theActionFlag = PlotSingleModuleHistos;    
    theActionFlag = NoAction;  
    infoExtractor_->getSingleModuleHistos(bei_, requestMap_, out, isUpgrade);    
  } else if (requestID == "PlotHistogramFromPath") {
   //theActionFlag = PlotHistogramFromPath;
   theActionFlag = NoAction;
   infoExtractor_->getHistosFromPath(bei_, requestMap_, out);    
  //} else if (requestID == "PlotSingleHistogram") {
  //  theActionFlag = PlotSingleHistogram;

  } else if (requestID == "PlotTkMapHistogram") {
    //string theMEName = get_from_multimap(requestMap_, "MEName");
    //string theModId  = get_from_multimap(requestMap_, "ModId");
    //infoExtractor_->plotTkMapHisto(bei, theModId, theMEName);
    //out->getHTTPResponseHeader().addHeader("Content-Type", "image/png");
    //out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
    //out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
    //out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
    //*out << infoExtractor_->getNamedImage(theMEName).str();
    theActionFlag = NoAction;    
    infoExtractor_->getTrackerMapHistos(bei_, requestMap_, out, isUpgrade);
  //} else if (requestID == "UpdatePlot") {
  //  string theMEName = get_from_multimap(requestMap_, "MEName");
  //  out->getHTTPResponseHeader().addHeader("Content-Type", "image/png");
  //  out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
  //  out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
 //   out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
  //  *out << infoExtractor_->getImage().str();
  //  theActionFlag = NoAction;    
 // } else if (requestID == "UpdateTkMapPlot") {
 //   string theMEName = get_from_multimap(requestMap_, "MEName");
 //   out->getHTTPResponseHeader().addHeader("Content-Type", "image/png");
 //   out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
 //   out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
 //   out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
 //   *out << infoExtractor_->getNamedImage(theMEName).str();
 //    theActionFlag = NoAction;    

  } else if (requestID == "GetMEList") {
    theActionFlag = NoAction;
    infoExtractor_->readModuleAndHistoList(bei_, out);    

  } else if (requestID == "GetTkMap") { 
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
  } else if (requestID == "periodicTrackerMapUpdate") {
   theActionFlag = NoAction;
   periodicTkMapUpdate(out) ;

  } else if  (requestID == "globalQFlag") {
    //cout << ACYellow << ACBold << "[SiPixelWebInterface::handleEDARequest]  " 
    //     << ACPlain << "Compute global Pixel quality flag" << endl;
    theActionFlag = ComputeGlobalQualityFlag;
  } else if (requestID == "dumpModIds") {
    theActionFlag = dumpModIds;
  }
    
  performAction(isUpgrade);
}

//____________________________________________________________________________________________________
// -- Perform action
//
void SiPixelWebInterface::performAction(bool isUpgrade) {
  //cout<<"entering performAction..."<<endl;
  //DQMStore * bei_ = (*mui_p)->getBEInterface();
  switch (theActionFlag) {
  case SiPixelWebInterface::CreateTkMap :
    {
     if (createTkMap()) {
//     cout<<"calling the dtor before this?"<<endl;
       tkMapCreated = true;
//       theOut->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
//      *theOut << "<?xml version=\"1.0\" ?>"	     << endl;
//      *theOut << "<TkMap>"			     << endl;
//      *theOut << " <Response>Successfull</Response>" << endl;
//      *theOut << "</TkMap>"			     << endl;
//cout<<"Done creating the TkMap in WI::performAction, what now?!"<<endl;
     }
      break;
    }
  case SiPixelWebInterface::Summary :
    {
      actionExecutor_->createSummary(bei_, isUpgrade);
      break;
    }
  case SiPixelWebInterface::Occupancy :
    {
      actionExecutor_->createOccupancy(bei_);
      break;
    }
  case SiPixelWebInterface::setupQTest :
    {
      actionExecutor_->setupQTests(bei_);
      break;
    }
  case SiPixelWebInterface::QTestResult :
    {
      actionExecutor_->checkQTestResults(bei_);
      break;
    }
  case SiPixelWebInterface::PlotSingleModuleHistos :
    {
//      infoExtractor_->plotSingleModuleHistos(bei_, requestMap_);
      break;
    }
  case SiPixelWebInterface::PlotTkMapHistogram :
    {
//      string theMEName = get_from_multimap(requestMap_, "MEName");
//      string theModId  = get_from_multimap(requestMap_, "ModId");
//      infoExtractor_->plotTkMapHisto(bei_, theModId, theMEName);
//
//      cout << ACYellow << ACBold 
//           << "[SiPixelWebInterface::PlotTkMapHistogram()]"
//           << ACPlain
//           << " Buffer for "
// 	   <<  theMEName
// 	   << " stored away: shipping back header (" << theOut << ")" 
//           << endl ;
///      theOut->getHTTPResponseHeader().addHeader("Content-Type", "image/png");
///      theOut->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
///      theOut->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
///      theOut->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
///      cout << ACYellow << ACBold 
///           << "[SiPixelWebInterface::PlotTkMapHistogram()]"
///           << ACPlain
///           << " Buffer for "
/// 	   <<  theMEName
/// 	   << " stored away: shipping back data (" << theOut << ")" 
///           << endl ;
///      *theOut << infoExtractor_->getNamedImage(theMEName).str();
      break;
    }
  case SiPixelWebInterface::PlotSingleHistogram :
    {
//      infoExtractor_->plotSingleHistogram(bei_, requestMap_);
      break;
    }
  case SiPixelWebInterface::periodicTrackerMapUpdate :
    {
      break;
    }
  case SiPixelWebInterface::PlotHistogramFromPath :
    {
//      infoExtractor_->getHistosFromPath(bei_, requestMap_);
      break;
    }
  case SiPixelWebInterface::CreatePlots :
    {
      infoExtractor_->createImages(bei_);
      break;
    }
  case SiPixelWebInterface::dumpModIds  :
    {
      break;
    }
  case SiPixelWebInterface::ComputeGlobalQualityFlag :
    {
      break;
    }
  case SiPixelWebInterface::NoAction :
    {
      break;
    }
  }
  setActionFlag(SiPixelWebInterface::NoAction);
//  cout<<"leaving performAction..."<<endl;
}


//____________________________________________________________________________________________________
void SiPixelWebInterface::returnReplyXml(xgi::Output * out, 
                                         const std::string& name, 
					 const std::string& comment){
   out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  *out << "<?xml version=\"1.0\" ?>" << std::endl;
  *out << "<TkMap>" << endl;
  *out << " <Response>" << comment << "</Response>" << endl;
  *out << "</TkMap>" << endl;

}


//____________________________________________________________________________________________________
bool SiPixelWebInterface::createTkMap() {
  //DQMStore * bei = (*mui_p)->getBEInterface();
  if (theActionFlag == SiPixelWebInterface::CreateTkMap) {
    string sname     = get_from_multimap(requestMap_, "MEName");
    string theTKType = get_from_multimap(requestMap_, "TKMapType");
    actionExecutor_->createTkMap(bei_, sname, theTKType);
    return true;
  } else {
    return false;
  }
}


//____________________________________________________________________________________________________
void SiPixelWebInterface::periodicTkMapUpdate(xgi::Output * out)
{
  //DQMStore * bei = (*mui_p)->getBEInterface();
  string sname     = get_from_multimap(requestMap_, "MEName");
  string theTKType = get_from_multimap(requestMap_, "TKMapType");
  infoExtractor_->sendTkUpdatedStatus(bei_, out, sname, theTKType) ;
}

//____________________________________________________________________________________________________
bool SiPixelWebInterface::readConfiguration(int& tkmap_freq,int& summary_freq){
  bool success=false;
  tkmap_freq = -1;
  summary_freq = -1;
  if (actionExecutor_)  {
    actionExecutor_->readConfiguration(tkmap_freq,summary_freq);
    if(tkmap_freq!=-1 && summary_freq!=-1) success=true;
  }
  return success;
}

//
// Get the RequestId and tags 
//
std::string SiPixelWebInterface::get_from_multimap(std::multimap<std::string, std::string> &mymap, std::string key)
{
  std::multimap<std::string, std::string>::iterator it;
  it = mymap.find(key);
  if (it != mymap.end())
    {
      return (it->second);
    }
  return "";
}
