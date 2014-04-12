/*! \file SiPixelInformationExtractor.cc
 *  \brief This class represents ...
 *  
 *  (Documentation under development)
 *  
 */
#include "DQM/SiPixelMonitorClient/interface/SiPixelInformationExtractor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelUtility.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelEDAClient.h"
#include "DQM/SiPixelMonitorClient/interface/ANSIColors.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelHistoPlotter.h"
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "CondFormats/SiPixelObjects/interface/DetectorIndex.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "TClass.h"
#include "TText.h"
#include "TROOT.h"
#include "TPad.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TString.h"
#include "TImage.h"
#include "TPaveText.h"
#include "TImageDump.h"
#include "TRandom.h"
#include "TStopwatch.h"
#include "TAxis.h"
#include "TPaveLabel.h"
#include "Rtypes.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"

#include <iostream>
#include <math.h>
#include <map>

#include <cstdlib> // for free() - Root can allocate with malloc() - sigh...
 
using namespace std;
using namespace edm;

//------------------------------------------------------------------------------
/*! \brief Constructor of the SiPixelInformationExtractor class.
 *  
 */
SiPixelInformationExtractor::SiPixelInformationExtractor(bool offlineXMLfile) : offlineXMLfile_(offlineXMLfile) {
  edm::LogInfo("SiPixelInformationExtractor") << 
    " Creating SiPixelInformationExtractor " << "\n" ;
  
  readReference_ = false;
  histoPlotter_=0;
  histoPlotter_ = new SiPixelHistoPlotter();
}

//------------------------------------------------------------------------------
/*! \brief Destructor of the SiPixelInformationExtractor class.
 *  
 */
SiPixelInformationExtractor::~SiPixelInformationExtractor() {
  edm::LogInfo("SiPixelInformationExtractor") << 
    " Deleting SiPixelInformationExtractor " << "\n" ;
  
  if (histoPlotter_) delete histoPlotter_;
}

//------------------------------------------------------------------------------
/*! \brief Read Configuration File
 *
 */
void SiPixelInformationExtractor::readConfiguration() { }

//
// -- Select Histograms for a given module
//
/*Removing xdaq deps
void SiPixelInformationExtractor::getSingleModuleHistos(DQMStore * bei, 
                                                        const multimap<string, string>& req_map, 
							xgi::Output * out,
							bool isUpgrade){
  //cout<<"In SiPixelInformationExtractor::getSingleModuleHistos: "<<endl;
  vector<string> hlist;
  getItemList(req_map,"histo", hlist);

  uint32_t detId = atoi(getItemValue(req_map,"ModId").c_str());
 
  int width  = atoi(getItemValue(req_map, "width").c_str());
  int height = atoi(getItemValue(req_map, "height").c_str());

  string opt =" ";
  
  SiPixelFolderOrganizer folder_organizer;
  string path;
  folder_organizer.getModuleFolder(detId,path,isUpgrade);   

  if((bei->pwd()).find("Module_") == string::npos &&
     (bei->pwd()).find("FED_") == string::npos){
    cout<<"This is not a pixel module or FED!"<<endl;
    return;
  }
 
  vector<MonitorElement*> all_mes = bei->getContents(path);
  setHTMLHeader(out);
  *out << path << " ";

  string theME ;
  for (vector<string>::const_iterator ih = hlist.begin();
       ih != hlist.end(); ih++) {
    for (vector<MonitorElement *>::const_iterator it = all_mes.begin();
	 it!= all_mes.end(); it++) {
      MonitorElement * me = (*it);
      if (!me) continue;
      theME = me->getName();
      string temp_s ; 
      if(theME.find("siPixel")!=string::npos || theME.find("ctfWithMaterialTracks")!=string::npos) { temp_s = theME.substr(0,theME.find_first_of("_")); }
      //cout<<"should be the variable name: temp_s= "<<temp_s<<endl;
      if (temp_s == (*ih)) {
	string full_path = path + "/" + me->getName();
	histoPlotter_->setNewPlot(full_path, opt, width, height);
	*out << me->getName() << " " ;
      }
    }
  }
}
*/
//
// -- Plot Tracker Map MEs
//
/* removing xdaq deps
void SiPixelInformationExtractor::getTrackerMapHistos(DQMStore* bei, 
                                                      const std::multimap<std::string, std::string>& req_map, 
						      xgi::Output * out,
						      bool isUpgrade) {

//  cout << __LINE__ << ACYellow << ACBold 
//       << "[SiPixelInformationExtractor::getTrackerMapHistos] " << ACPlain << endl ;
//  cout<<"I am in this dir: "<<bei->pwd()<<endl;
  vector<string> hlist;
  string tkmap_name;
  SiPixelConfigParser config_parser;
  string localPath;
  if(offlineXMLfile_) localPath = string("DQM/SiPixelMonitorClient/test/sipixel_tier0_config.xml");
  else localPath = string("DQM/SiPixelMonitorClient/test/sipixel_monitorelement_config.xml");
  config_parser.getDocument(edm::FileInPath(localPath).fullPath());
//  if (!config_parser.getMENamesForTrackerMap(tkmap_name, hlist)) return;
//  if (hlist.size() == 0) return;
  if (!config_parser.getMENamesForTrackerMap(tkmap_name, hlist)) 
  {
   cout << __LINE__ << ACYellow << ACBold 
        << "[SiPixelInformationExtractor::getTrackerMapHistos] " 
	<< ACPlain << ACRed << ACPlain 
	<< "getMENamesForTrackerMap return false " 
        << ACPlain << endl ; assert(0) ;
   return;
  }
  if (hlist.size() == 0) 
  {
   cout << __LINE__ << ACYellow << ACBold 
        << "[SiPixelInformationExtractor::getTrackerMapHistos] " 
	<< ACPlain << ACRed << ACPlain 
	<< "hlist.size() == 0 " 
        << ACPlain << endl ;  assert(0) ;
   return;
  }


  uint32_t detId = atoi(getItemValue(req_map,"ModId").c_str());
 
  int width  = atoi(getItemValue(req_map, "width").c_str());
  int height = atoi(getItemValue(req_map, "height").c_str());

  string opt =" ";
  
  SiPixelFolderOrganizer folder_organizer;
  string path;
  
  folder_organizer.getModuleFolder(detId,path,isUpgrade);
  string currDir = bei->pwd();   
//  cout<<"detId= "<<detId<<" , path= "<<path<<" , and now I am in "<<currDir<<endl;
  

  if((bei->pwd()).find("Module_") == string::npos &&
     (bei->pwd()).find("FED_") == string::npos){
    cout<<"This is not a pixel module or FED!"<<endl;
   cout << __LINE__ << ACYellow << ACBold 
        << "[SiPixelInformationExtractor::getTrackerMapHistos] " 
	<< ACPlain << ACRed << ACPlain 
	<< "This is not a pixel module or FED!" 
        << ACPlain << endl ; assert(0) ;
    return;
  }

  vector<MonitorElement*> all_mes = bei->getContents(path);
  setXMLHeader(out);

  cout << __LINE__ << ACCyan << ACBold 
       << " [SiPixelInformationExtractor::getTrackerMapHistos()] path "
       << ACPlain << path << endl ; 
  cout << __LINE__ << ACCyan << ACBold 
       << " [SiPixelInformationExtractor::getTrackerMapHistos()] all_mes.size() "
       << ACPlain << all_mes.size() << endl ; 

  string theME ;
  *out << "<pathList>" << endl ;
  for (vector<string>::iterator ih = hlist.begin();
       ih != hlist.end(); ih++) {
       //cout<<"ih iterator (hlist): "<<(*ih)<<endl;
    for (vector<MonitorElement *>::const_iterator it = all_mes.begin();
	 it!= all_mes.end(); it++) {
      MonitorElement * me = (*it);
      if (!me) 
      { 
       cout << __LINE__ << ACCyan << ACBold 
            << " [SiPixelInformationExtractor::getTrackerMapHistos()] skipping "
        	       << ACPlain << *ih << endl ; 
       continue;
      }
      theME = me->getName();
      //cout<<"ME iterator (all_mes): "<<theME<<endl; 
      string temp_s ; 
      if(theME.find("siPixel")!=string::npos || theME.find("ctfWithMaterialTracks")!=string::npos) { temp_s = theME.substr(0,theME.find_first_of("_")); }
      //cout << __LINE__ << ACCyan << ACBold 
      //     << " [SiPixelInformationExtractor::getTrackerMapHistos()] temp_s "
      //     << ACPlain << temp_s << " <--> " << *ih << " |" << theME << "|" << endl ; 
      if (temp_s == (*ih)) {
	string full_path = path + "/" + me->getName();
	histoPlotter_->setNewPlot(full_path, opt, width, height);
//cout << __LINE__ << ACRed << ACBold 
//     << " [SiPixelInformationExtractor::getTrackerMapHistos()] fullPath: "
//     << ACPlain << full_path << endl ; 
	*out << " <pathElement path='" << full_path << "' />" << endl ;
      }      
    }
  }   
  *out << "</pathList>" << endl ;
//cout << __LINE__ << " [SiPixelInformationExtractor::getTrackerMapHistos()] endlist: " << endl ;
}
*/
//============================================================================================================
// --  Return type of ME
//
std::string  SiPixelInformationExtractor::getMEType(MonitorElement * theMe)
{
  string qtype = theMe->getRootObject()->IsA()->GetName() ;
  if(         qtype.find("TH1") != string::npos )
  {
    return "TH1" ;
  } else if ( qtype.find("TH2") != string::npos  ) {
    return "TH2" ;
  } else if ( qtype.find("TH3") != string::npos ) {
    return "TH3" ;
  }
  return "TH1" ;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
/*removing xdaq deps
void SiPixelInformationExtractor::readModuleAndHistoList(DQMStore* bei, 
                                                         xgi::Output * out) {
//cout<<"entering SiPixelInformationExtractor::readModuleAndHistoList"<<endl;
   bei->cd("Pixel");
   std::map<std::string,std::string> hnames;
   std::vector<std::string> mod_names;
   fillModuleAndHistoList(bei, mod_names, hnames);
   out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  *out << "<?xml version=\"1.0\" ?>" << std::endl;
  *out << "<ModuleAndHistoList>" << endl;
  *out << "<ModuleList>" << endl;
   for (std::vector<std::string>::iterator im = mod_names.begin();
        im != mod_names.end(); im++) {
     *out << "<ModuleNum>" << *im << "</ModuleNum>" << endl;     
   }
   *out << "</ModuleList>" << endl;
   *out << "<HistoList>" << endl;

   for (std::map<std::string,std::string>::iterator ih = hnames.begin();
        ih != hnames.end(); ih++) {
     *out << "<Histo type=\"" 
          << ih->second
	  << "\">" 
	  << ih->first 
	  << "</Histo>" 
	  << endl;     
   }
   *out << "</HistoList>" << endl;
   *out << "</ModuleAndHistoList>" << endl;
//cout<<"leaving SiPixelInformationExtractor::readModuleAndHistoList"<<endl;
}
*/

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
void SiPixelInformationExtractor::fillModuleAndHistoList(DQMStore * bei, 
                                                         vector<string>        & modules,
							 map<string,string>    & histos) {
//cout<<"entering SiPixelInformationExtractor::fillModuleAndHistoList"<<endl;
  string currDir = bei->pwd();
  //cout<<"currDir= "<<currDir<<endl;
  if(currDir.find("Module_") != string::npos){
    if(histos.size() == 0){
      vector<string> contents = bei->getMEs();
      for (vector<string>::const_iterator it = contents.begin(); it != contents.end(); it++) {
	string hname          = (*it).substr(0, (*it).find("_siPixel"));
	if(hname==" ") hname = (*it).substr(0, (*it).find("_generalTracks"));
        string fullpathname   = bei->pwd() + "/" + (*it); 
       // cout<<"fullpathname="<<fullpathname<<endl;
        MonitorElement * me   = bei->get(fullpathname);
        string htype          = "undefined" ;
        if(me) htype = me->getRootObject()->IsA()->GetName() ;
	//cout<<"hname="<<hname<<endl;
	//if(htype=="TH1F" || htype=="TH1D"){
        histos[hname] = htype ;
        string mId=" ";
	if(hname.find("ndigis") 	       !=string::npos) mId = (*it).substr((*it).find("ndigis_siPixelDigis_")+20, 9);
	if(mId==" " && hname.find("nclusters") !=string::npos) mId = (*it).substr((*it).find("nclusters_siPixelClusters_")+26, 9);
        if(mId==" " && hname.find("residualX") !=string::npos) mId = (*it).substr((*it).find("residualX_ctfWithMaterialTracks_")+32, 9);
        if(mId==" " && hname.find("NErrors") !=string::npos) mId = (*it).substr((*it).find("NErrors_siPixelDigis_")+21, 9);
        if(mId==" " && hname.find("ClustX") !=string::npos) mId = (*it).substr((*it).find("ClustX_siPixelRecHit_")+21, 9);
        if(mId==" " && hname.find("pixelAlive") !=string::npos) mId = (*it).substr((*it).find("pixelAlive_siPixelCalibDigis_")+29, 9);
        if(mId==" " && hname.find("Gain1d") !=string::npos) mId = (*it).substr((*it).find("Gain1d_siPixelCalibDigis_")+25, 9);
        if(mId!=" ") modules.push_back(mId);
        //cout<<"mId="<<mId<<endl;
	//}
      }    
    }
  } else {  
    vector<string> subdirs = bei->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin(); it != subdirs.end(); it++) {
      if((bei->pwd()).find("Barrel")==string::npos && (bei->pwd()).find("Endcap")==string::npos) bei->goUp();
      bei->cd(*it);
      fillModuleAndHistoList(bei, modules, histos);
      bei->goUp();
    }
  }
//  fillBarrelList(bei, modules, histos);
//cout<<"leaving SiPixelInformationExtractor::fillModuleAndHistoList"<<endl;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
/* removing xdaq deps
void SiPixelInformationExtractor::readModuleHistoTree(DQMStore* bei, 
                                                      string& str_name, 
						      xgi::Output * out) {
//cout<<"entering  SiPixelInformationExtractor::readModuleHistoTree"<<endl;
  ostringstream modtree;
  if (goToDir(bei, str_name)) {
    modtree << "<form name=\"IMGCanvasItemsSelection\" "
            << "action=\"javascript:void%200\">" 
	    << endl ;
    modtree << "<ul id=\"dhtmlgoodies_tree\" class=\"dhtmlgoodies_tree\">" << endl;
    printModuleHistoList(bei,modtree);
    modtree <<"</ul>" << endl;   
    modtree <<"</form>" << endl;   
  } else {
    modtree << "Desired Directory does not exist";
  }
  cout << ACYellow << ACBold
       << "[SiPixelInformationExtractor::readModuleHistoTree()]"
       << ACPlain << endl ;
  //     << "html string follows: " << endl ;
  //cout << modtree.str() << endl ;
  //cout << ACYellow << ACBold
  //     << "[SiPixelInformationExtractor::readModuleHistoTree()]"
  //     << ACPlain
  //     << "String complete " << endl ;
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/plain");
  *out << modtree.str();
   bei->cd();
//cout<<"leaving  SiPixelInformationExtractor::readModuleHistoTree"<<endl;
}
*/

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
void SiPixelInformationExtractor::printModuleHistoList(DQMStore * bei, 
                                                       ostringstream& str_val){
//cout<<"entering SiPixelInformationExtractor::printModuleHistoList"<<endl;
  string currDir = bei->pwd();
  string dname = currDir.substr(currDir.find_last_of("/")+1);
  str_val << " <li>\n"
	  << "  <a href=\"#\" id=\"" << currDir << "\">\n   " 
	  <<     dname << "\n"
	  << "  </a>\n"
	  << endl << endl;

  vector<string> meVec     = bei->getMEs(); 
  
  vector<string> subDirVec = bei->getSubdirs();
  if ( meVec.size()== 0  && subDirVec.size() == 0 ) {
    str_val << " </li>" << endl;    
    return;
  }
  str_val << "\n   <ul>" << endl; 
  for (vector<string>::const_iterator it  = meVec.begin();
                                      it != meVec.end(); it++) {
    if ((*it).find("_siPixel")!=string::npos || 
        (*it).find("_ctfWithMaterialTracks")!=string::npos) {
      string qit = (*it) ;
      string temp_s;
      if(qit.find("siPixel")!=string::npos || qit.find("ctfWithMaterialTracks")!=string::npos) { temp_s = qit.substr(0,qit.find_first_of("_")); }
      str_val << "    <li class=\"dhtmlgoodies_sheet.gif\">\n"
	      << "     <input id      = \"selectedME\""
	      << "            folder  = \"" << currDir << "\""
	      << "            type    = \"checkbox\""
	      << "            name    = \"selected\""
	      << "            class   = \"smallCheckBox\""
	      << "            value   = \"" << (*it) << "\""
	      << "            onclick = \"javascript:IMGC.selectedIMGCItems()\" />\n"
//	      << "     <a href=\"javascript:IMGC.updateIMGC('" << currDir << "')\">\n       " 
	      << "     <a href=\"javascript:IMGC.plotFromPath('" << currDir << "')\">\n       " 
//	      <<        temp_s << "\n"
	      <<        (*it) << "\n"
	      << "     </a>\n"
	      << "    </li>" 
	      << endl;
    }
  }
  for (vector<string>::const_iterator ic  = subDirVec.begin();
                                      ic != subDirVec.end(); ic++) {
    bei->cd(*ic);
    printModuleHistoList(bei, str_val);
    bei->goUp();
  }
  str_val << "   </ul>" << endl;  
  str_val << "  </li>"  << endl;  
//cout<<"leaving SiPixelInformationExtractor::printModuleHistoList"<<endl;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
/* removing xdaq deps
void SiPixelInformationExtractor::readSummaryHistoTree(DQMStore* bei, 
                                                       string& str_name, 
						       xgi::Output * out) {
//cout<<"entering  SiPixelInformationExtractor::readSummaryHistoTree"<<endl;
  ostringstream sumtree;
  if (goToDir(bei, str_name)) {
    sumtree << "<ul id=\"dhtmlgoodies_tree\" class=\"dhtmlgoodies_tree\">" << endl;
    printSummaryHistoList(bei,sumtree);
    sumtree <<"</ul>" << endl;   
  } else {
    sumtree << "Desired Directory does not exist";
  }
  cout << ACYellow << ACBold
       << "[SiPixelInformationExtractor::readSummaryHistoTree()]"
       << ACPlain << endl ;
  //     << "html string follows: " << endl ;
  //cout << sumtree.str() << endl ;
  //cout << ACYellow << ACBold
  //     << "[SiPixelInformationExtractor::readSummaryHistoTree()]"
  //     << ACPlain
  //     << "String complete " << endl ;
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/plain");
  *out << sumtree.str();
   bei->cd();
//cout<<"leaving  SiPixelInformationExtractor::readSummaryHistoTree"<<endl;
}
*/

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  Returns a stringstream containing an HTML-formatted list of ME in the current
 *  directory. 
 *  This is a recursive method.
 */
void SiPixelInformationExtractor::printSummaryHistoList(DQMStore * bei, 
                                                        ostringstream& str_val){
//cout<<"entering SiPixelInformationExtractor::printSummaryHistoList"<<endl;
  string currDir = bei->pwd();
  string dname = currDir.substr(currDir.find_last_of("/")+1);
  if (dname.find("Module_") ==0 || dname.find("FED_")==0) return;
  str_val << " <li>\n"
          << "  <a href=\"#\" id=\"" << currDir << "\">\n   " 
	  <<     dname 
	  << "  </a>" 
	  << endl;

  vector<string> meVec     = bei->getMEs(); 
  
  vector<string> subDirVec = bei->getSubdirs();
  if ( meVec.size()== 0  && subDirVec.size() == 0 ) {
    str_val << " </li> "<< endl;    
    return;
  }
  str_val << "\n   <ul>" << endl;      
  for (vector<string>::const_iterator it = meVec.begin();
       it != meVec.end(); it++) {
    if ((*it).find("SUM") == 0) {
      str_val << "    <li class=\"dhtmlgoodies_sheet.gif\">\n"
	      << "     <input id      = \"selectedME\""
	      << "            folder  = \"" << currDir << "\""
	      << "            type    = \"checkbox\""
	      << "            name    = \"selected\""
	      << "            class   = \"smallCheckBox\""
	      << "            value   = \"" << (*it) << "\""
	      << "            onclick = \"javascript:IMGC.selectedIMGCItems()\" />\n"
//              << "     <a href=\"javascript:IMGC.updateIMGC('" << currDir << "')\">\n       " 
              << "     <a href=\"javascript:IMGC.plotFromPath('" << currDir << "')\">\n       " 
	      <<       (*it) << "\n"
	      << "     </a>\n"
	      << "    </li>" 
	      << endl;
    }
  }

  for (vector<string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    bei->cd(*ic);
    printSummaryHistoList(bei, str_val);
    bei->goUp();
  }
  str_val << "   </ul> "<< endl;  
  str_val << "  </li> "<< endl;  
//cout<<"leaving SiPixelInformationExtractor::printSummaryHistoList"<<endl;
}


//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
/* removing xdaq deps
void SiPixelInformationExtractor::readAlarmTree(DQMStore* bei, 
                                                string& str_name, 
						xgi::Output * out){
//cout<<"entering SiPixelInformationExtractor::readAlarmTree"<<endl;
  ostringstream alarmtree;
  if (goToDir(bei, str_name)) {
    alarmtree << "<ul id=\"dhtmlgoodies_tree\" class=\"dhtmlgoodies_tree\">" << endl;
    alarmCounter_=0;
    printAlarmList(bei,alarmtree);
    if(alarmCounter_==0) alarmtree <<"<li>No problematic modules found, all ok!</li>" << endl;
    alarmtree <<"</ul>" << endl; 
  } else {
    alarmtree << "Desired Directory does not exist";
  }
  cout << ACYellow << ACBold
       << "[SiPixelInformationExtractor::readAlarmTree()]"
       << ACPlain << endl ;
  //     << "html string follows: " << endl ;
  //cout << alarmtree.str() << endl ;
  //cout << ACYellow << ACBold
  //     << "[SiPixelInformationExtractor::readAlarmTree()]"
  //     << ACPlain
  //     << "String complete " << endl ;
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/plain");
 *out << alarmtree.str();
  bei->cd();
  cout << ACYellow << ACBold
       << "[SiPixelInformationExtractor::readAlarmTree()]"
       << ACPlain 
       << " Done!"
       << endl ;
//cout<<"leaving SiPixelInformationExtractor::readAlarmTree"<<endl;
}
*/
//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *  
 *  Returns a stringstream containing an HTML-formatted list of alarms for the current
 *  directory. 
 *  This is a recursive method.
 */
void SiPixelInformationExtractor::printAlarmList(DQMStore * bei, 
                                                 ostringstream& str_val){
//cout<<"entering SiPixelInformationExtractor::printAlarmList"<<endl;
//   cout << ACRed << ACBold
//        << "[SiPixelInformationExtractor::printAlarmList()]"
//        << ACPlain
//        << " Enter" 
//        << endl ;
  string currDir = bei->pwd();
  string dname = currDir.substr(currDir.find_last_of("/")+1);
  string image_name;
  selectImage(image_name,bei->getStatus(currDir));
  if(image_name!="images/LI_green.gif")
    str_val << " <li>\n"
            << "  <a href=\"#\" id=\"" << currDir << "\">\n   " 
	    <<     dname 
	    << "  </a>\n"
	    << "  <img src=\"" 
            <<     image_name 
	    << "\">" << endl;
  vector<string> subDirVec = bei->getSubdirs();

  vector<string> meVec = bei->getMEs();
   
  if (subDirVec.size() == 0 && meVec.size() == 0) {
    str_val <<  "</li> "<< endl;    
    return;
  }
  str_val << "<ul>" << endl;
  for (vector<string>::const_iterator it = meVec.begin();
	   it != meVec.end(); it++) {
    string full_path = currDir + "/" + (*it);

    MonitorElement * me = bei->get(full_path);
    
    if (!me) continue;
    std::vector<QReport *> my_map = me->getQReports();
    if (my_map.size() > 0) {
      string image_name1;
      selectImage(image_name1,my_map);
      if(image_name1!="images/LI_green.gif") {
        alarmCounter_++;
        str_val << "	<li class=\"dhtmlgoodies_sheet.gif\">\n"
        	<< "	 <input id	= \"selectedME\""
        	<< "		folder  = \"" << currDir << "\""
        	<< "		type	= \"checkbox\""
        	<< "		name	= \"selected\""
        	<< "		class	= \"smallCheckBox\""
        	<< "		value	= \"" << (*it) << "\""
        	<< "		onclick = \"javascript:IMGC.selectedIMGCItems()\" />\n"
//        	<< "	 <a href=\"javascript:IMGC.updateIMGC('" << currDir << "')\">\n       " 
        	<< "	 <a href=\"javascript:IMGC.plotFromPath('" << currDir << "')\">\n       " 
        	<<	  (*it) << "\n"
        	<< "	 </a>\n"
		<< "     <img src=\""
		<<        image_name1 
		<< "\">"
        	<< "	</li>" 
        	<< endl;
	}	
    }
  }
  for (vector<string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    bei->cd(*ic);
    printAlarmList(bei, str_val);
    bei->goUp();
  }
  str_val << "</ul> "<< endl;  
  str_val << "</li> "<< endl;  
//   cout << ACGreen << ACBold
//        << "[SiPixelInformationExtractor::printAlarmList()]"
//        << ACPlain
//        << " Done" 
//        << endl ;
//cout<<"leaving SiPixelInformationExtractor::printAlarmList"<<endl;
}


//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
void SiPixelInformationExtractor::getItemList(const multimap<string, string>& req_map, 
                                              string item_name,
					      vector<string>& items) {
//cout<<"entering SiPixelInformationExtractor::getItemList"<<endl;
  items.clear();
  for (multimap<string, string>::const_iterator it = req_map.begin();
       it != req_map.end(); it++) {
    if (it->first == item_name) {
      items.push_back(it->second);
    }
  }
//cout<<"leaving SiPixelInformationExtractor::getItemList"<<endl;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
bool SiPixelInformationExtractor::hasItem(multimap<string,string>& req_map,
					  string item_name){
//cout<<"entering SiPixelInformationExtractor::hasItem"<<endl;
  multimap<string,string>::iterator pos = req_map.find(item_name);
  if (pos != req_map.end()) return true;
  return false;  
//cout<<"leaving SiPixelInformationExtractor::hasItem"<<endl;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
std::string SiPixelInformationExtractor::getItemValue(const std::multimap<std::string,std::string>& req_map,
						 std::string item_name){
//cout<<"entering SiPixelInformationExtractor::getItemValue for item: "<<item_name<<endl;
  std::multimap<std::string,std::string>::const_iterator pos = req_map.find(item_name);
  std::string value = " ";
  if (pos != req_map.end()) {
    value = pos->second;
  }
//  cout<<"value = "<<value<<endl;
  return value;
//cout<<"leaving SiPixelInformationExtractor::getItemValue"<<endl;
}
std::string SiPixelInformationExtractor::getItemValue(std::multimap<std::string,std::string>& req_map,
						 std::string item_name){
//cout<<"entering SiPixelInformationExtractor::getItemValue for item: "<<item_name<<endl;
  std::multimap<std::string,std::string>::iterator pos = req_map.find(item_name);
  std::string value = " ";
  if (pos != req_map.end()) {
//  cout<<"item found!"<<endl;
    value = pos->second;
  }
//  cout<<"value = "<<value<<endl;
  return value;
//cout<<"leaving SiPixelInformationExtractor::getItemValue"<<endl;
}

//
// -- Get color  name from status
//
void SiPixelInformationExtractor::selectColor(string& col, int status){
  if (status == dqm::qstatus::STATUS_OK)    col = "#00ff00";
  else if (status == dqm::qstatus::WARNING) col = "#ffff00";
  else if (status == dqm::qstatus::ERROR)   col = "#ff0000";
  else if (status == dqm::qstatus::OTHER)   col = "#ffa500";
  else  col = "#0000ff";
}
//
// -- Get Image name from ME
//
void SiPixelInformationExtractor::selectColor(string& col, vector<QReport*>& reports){
  int istat = 999;
  int status = 0;
  for (vector<QReport*>::const_iterator it = reports.begin(); it != reports.end();
       it++) {
    status = (*it)->getStatus();
    if (status > istat) istat = status;
  }
  selectColor(col, status);
}
//
// -- Get Image name from status
//
void SiPixelInformationExtractor::selectImage(string& name, int status){
  if (status == dqm::qstatus::STATUS_OK) name="images/LI_green.gif";
  else if (status == dqm::qstatus::WARNING) name="images/LI_yellow.gif";
  else if (status == dqm::qstatus::ERROR) name="images/LI_red.gif";
  else if (status == dqm::qstatus::OTHER) name="images/LI_orange.gif";
  else  name="images/LI_blue.gif";
}
//
// -- Get Image name from ME
//
void SiPixelInformationExtractor::selectImage(string& name, vector<QReport*>& reports){
  int istat = 999;
  int status = 0;
  for (vector<QReport*>::const_iterator it = reports.begin(); it != reports.end();
       it++) {
    status = (*it)->getStatus();
    if (status > istat) istat = status;
  }
  selectImage(name, status);
}

//
// -- Get a tagged image 
//
/* removing xdaq deps
void SiPixelInformationExtractor::getIMGCImage(const multimap<string, string>& req_map, 
                                               xgi::Output * out){
  string path = getItemValue(req_map,"Path");
  string image;
  histoPlotter_->getNamedImageBuffer(path, image);

  out->getHTTPResponseHeader().addHeader("Content-Type", "image/png");
  out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
  out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
  out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
  *out << image;
}
*/
/* removing xdaq deps
void SiPixelInformationExtractor::getIMGCImage(multimap<string, string>& req_map, 
                                               xgi::Output * out){
  
  string path = getItemValue(req_map,"Path");
  string image;
  histoPlotter_->getNamedImageBuffer(path, image);

  out->getHTTPResponseHeader().addHeader("Content-Type", "image/png");
  out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
  out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
  out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
  *out << image;

}
*/

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
bool SiPixelInformationExtractor::goToDir(DQMStore* bei, 
                                          string& sname){ 
//cout<<"entering SiPixelInformationExtractor::goToDir"<<endl;
  bei->cd();
  //if(flg) bei->cd("Collector/Collated");
  bei->cd(sname);
  string dirName = bei->pwd();
  if (dirName.find(sname) != string::npos) return true;
  else return false;  
//cout<<"leaving SiPixelInformationExtractor::goToDir"<<endl;
}

//
// -- Get Warning/Error Messages
//
/* removing xdaq deps
void SiPixelInformationExtractor::readStatusMessage(DQMStore* bei, 
                                                    std::multimap<std::string, std::string>& req_map, 
						    xgi::Output * out){

  string path = getItemValue(req_map,"Path");

  int width  = atoi(getItemValue(req_map, "width").c_str());
  int height = atoi(getItemValue(req_map, "height").c_str());

  string opt =" ";

  ostringstream test_status;
  
  setXMLHeader(out);
  *out << "<StatusAndPath>" << endl;
  *out << "<PathList>" << endl;
  if (path.size() == 0) {
    *out << "<HPath>" << "NONE" << "</HPath>" << endl;     
    test_status << " ME Does not exist ! " << endl;
  } else {
    vector<MonitorElement*> all_mes = bei->getContents(path);
    *out << "<HPath>" << path << "</HPath>" << endl;     
    for(vector<MonitorElement*>::iterator it=all_mes.begin(); it!=all_mes.end(); it++){
      MonitorElement* me = (*it);
      if (!me) continue;
      string name = me->getName();  

      vector<QReport*> q_reports = me->getQReports();
      if (q_reports.size() == 0) continue;
      string full_path = path + "/" + name;
      histoPlotter_->setNewPlot(full_path, opt, width, height);

      if (q_reports.size() != 0) {
        test_status << " QTest Status for " << name << " : " << endl;
        test_status << " ======================================================== " << endl; 
        for (vector<QReport*>::const_iterator it = q_reports.begin(); it != q_reports.end();
	     it++) {
	  int status = (*it)->getStatus();
	  if (status == dqm::qstatus::WARNING) test_status << " Warning ";
	  else if (status == dqm::qstatus::ERROR) test_status << " Error  ";
	  else if (status == dqm::qstatus::STATUS_OK) test_status << " Ok  ";
	  else if (status == dqm::qstatus::OTHER) test_status << " Other(" << status << ") ";
	  string mess_str = (*it)->getMessage();
	  test_status <<  "&lt;br/&gt;";
	  mess_str = mess_str.substr(mess_str.find(" Test")+5);
	  test_status <<  " QTest Name  : " << mess_str.substr(0, mess_str.find(")")+1) << endl;
	  test_status << "&lt;br/&gt;";
	  test_status <<  " QTest Detail  : " << mess_str.substr(mess_str.find(")")+2) << endl;
	} 
	test_status << " ======================================================== " << endl;
      }
      *out << "<HPath>" << name << "</HPath>" << endl;         
    }    
  }
  *out << "</PathList>" << endl;
  *out << "<StatusList>" << endl;
  *out << "<Status>" << test_status.str() << "</Status>" << endl;      
  *out << "</StatusList>" << endl;
  *out << "</StatusAndPath>" << endl;
}
*/

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *  
 */
void SiPixelInformationExtractor::computeStatus(MonitorElement      * theME,
                                                double              & colorValue,
						pair<double,double> & norm) 
{
  double normalizationX = 1 ;
  double normalizationY = 1 ;
  double meanX          = 0 ;
  double meanY          = 0 ;
  
  colorValue = 0 ;

  pair<double,double> normX ;
  pair<double,double> normY ;

  string theMEType = getMEType(theME) ;

//   cout << ACRed << ACReverse
//        << "[SiPixelInformationExtractor::computeStatus()]"
//        << ACPlain
//        << " Computing average for "
//        << theME->getName()
//        << endl ;

  if( theMEType.find("TH1") != string::npos)
  {
   meanX = (double)theME->getMean();
   getNormalization(theME, normX, "TH1") ;
   normalizationX = fabs( normX.second - normX.first) ;
   if( normalizationX == 0 ) {normalizationX=1.E-20;}
   colorValue  = meanX / normalizationX ;
   norm.first  = normX.first ;
   norm.second = normX.second ;
  }
  
  if( theMEType.find("TH2") != string::npos)
  {
   meanX = (double)theME->getMean(1);
   meanY = (double)theME->getMean(2);
   getNormalization2D(theME, normX, normY, "TH2") ;
   normalizationX = fabs( normX.second - normX.first) ;
   normalizationY = fabs( normY.second - normY.first) ;
   if( normalizationX == 0 ) {normalizationX=1.E-20;}
   if( normalizationY == 0 ) {normalizationY=1.E-20;}
   double cVX = meanX / normalizationX ;
   double cVY = meanY / normalizationY ;
   colorValue = sqrt(cVX*cVX + cVY*cVY) ;
   if( normalizationX >= normalizationY )
   { 
    norm.first  = normX.first;
    norm.second = normX.second ;
   } else { 
    norm.first  = normY.first;
    norm.second = normY.second ;
   }
//   cout << ACBlue << ACBold << ACReverse
//        << "[SiPixelInformationExtractor::computeStatus()]"
//	<< ACPlain << "    "
//	<< theME->getName()
//	<< " meanX:Y "
//	<< meanX << ":" << meanY
//	<< " normX:Y " 
//	<< norm.first << ":" << norm.second
//	<< endl ;
  } 
 
  return ;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *  
 */
void SiPixelInformationExtractor::getNormalization(MonitorElement     * theME, 
                                                   pair<double,double>& norm,
						   std::string          theMEType) 
{
  double normLow  = 0 ;
  double normHigh = 0 ;

  if( theMEType.find("TH1") != string::npos)
  {
   normHigh    = (double)theME->getNbinsX() ;
   norm.first  = normLow  ;
   norm.second = normHigh ;
  }
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *  
 */
void SiPixelInformationExtractor::getNormalization2D(MonitorElement     * theME, 
                                                     pair<double,double>& normX,
                                                     pair<double,double>& normY,
						     std::string          theMEType) 
{
  double normLow  = 0 ;
  double normHigh = 0 ;

  if( theMEType.find("TH2") != string::npos )
  {
   normHigh    = (double)theME->getNbinsX() ;
   normX.first  = normLow  ;
   normX.second = normHigh ;
   normHigh    = (double)theME->getNbinsY() ;
   normY.first  = normLow  ;
   normY.second = normHigh ;
//   cout << ACCyan << ACBold << ACReverse
//        << "[SiPixelInformationExtractor::getNormalization2D()]"
//	<< ACPlain << " "
//	<< theME->getName()
//	<< " normX: " 
//	<< normX.first << ":" << normX.second
//	<< " normY: " 
//	<< normY.first << ":" << normY.second
//	<< endl ;
  }
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *   
 */
void SiPixelInformationExtractor::selectMEList(DQMStore   * bei,  
					       string	               & theMEName,
					       vector<MonitorElement*> & mes) 
{  
//  cout<<"In SiPixelInformationExtractor::selectMEList: "<<endl;
  string currDir = bei->pwd();
   
  string theME ;
   
  // Get ME from Collector/FU0/Tracker/PixelEndcap/HalfCylinder_pX/Disk_X/Blade_XX/Panel_XX/Module_XX
  if (currDir.find("Module_") != string::npos ||
      currDir.find("FED_") != string::npos)  
  {
    vector<string> contents = bei->getMEs(); 
       
    for (vector<string>::const_iterator it = contents.begin(); it != contents.end(); it++) 
    {
      theME = (*it) ;
      if(theME.find("siPixel")==string::npos && theME.find("ctfWithMaterialTracks")==string::npos) {continue ;} // If the ME is not a siPixel or ctfWithMaterialTrack one, skip
      string temp_s = theME.substr(0,theME.find_first_of("_"));
      //cout<<"should be the variable name: temp_s= "<<temp_s<<endl;
      if (temp_s == theMEName)  
      {
        string full_path = currDir + "/" + (*it);

        MonitorElement * me = bei->get(full_path.c_str());
	
        if (me) {mes.push_back(me);}
      }
    }
    return;
  } else {  // If not yet reached the desired level in the directory tree, recursively go down one level more
    vector<string> subdirs = bei->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin(); it != subdirs.end(); it++) 
    {
      bei->cd(*it);
      selectMEList(bei, theMEName, mes);
      bei->goUp();
    }
  }
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *  
 */
/* removing xdaq dependencies
void SiPixelInformationExtractor::sendTkUpdatedStatus(DQMStore  * bei, 
                                                      xgi::Output            * out,
						      std::string            & theMEName,
						      std::string            & theTKType) 
{
//  cout<<"In SiPixelInformationExtractor::sendTkUpdatedStatus: "<<endl;
  int rval, gval, bval;
  vector<string>          colorMap ;
  vector<MonitorElement*> me_list;
  pair<double,double>     norm ;
  double sts ;
    
  bei->cd();
  selectMEList(bei, theMEName, me_list) ;
  bei->cd();

  string detId = "undefined";

   cout << ACYellow << ACBold
	<< "[SiPixelInformationExtractor::sendTkUpdatedStatus()] "
	<< ACPlain
	<< "Preparing color map update for " 
	<< theMEName
	<< " type "
	<< theTKType
	<< " - List size: "
	<< me_list.size() 
	<< endl ;
  
  int maxEntries = 0 ;
  if( theTKType == "Entries") // In this case find the ME with the highest number of entries
  {			      // first and use that as a vertical scale normalization
   for(vector<MonitorElement*>::iterator it=me_list.begin(); it!=me_list.end(); it++)
   {
    int entries = (int)(*it)->getEntries() ;
    if( entries > maxEntries ) maxEntries = entries ;
   }
  }
  
  int entries = 0 ;
  stringstream jsSnippet ;
  for(vector<MonitorElement*>::iterator it=me_list.begin(); it!=me_list.end(); it++)
  {
    string meName    = (*it)->getName();
    string theMEType = getMEType(*it);
    if( meName.find("_3") != string::npos ) 
    {
     string detIdString = meName.substr(meName.find_last_of("_")+1,9);
     std::istringstream isst;
     isst.str(detIdString);
     isst>>detId;
     entries = (int)(*it)->getEntries() ;
     if( theTKType == "Averages") 
     {
      computeStatus(*it, sts, norm) ;
      SiPixelUtility::getStatusColor(sts, rval, gval, bval);
     } else if( theTKType == "Entries") {
      sts = (double)entries / (double)maxEntries ;
      SiPixelUtility::getStatusColor(sts, rval, gval, bval);
      if( entries > maxEntries ) maxEntries = entries ;
      norm.first  = 0 ;
      norm.second = maxEntries ;
     } else {
      int status  =  SiPixelUtility::getStatus((*it));
      if(        status == dqm::qstatus::ERROR ) 
      {
       rval = 255; gval =   0; bval =   0;
      } else if (status == dqm::qstatus::WARNING )  {
       rval = 255; gval = 255; bval =   0; 
      } else if (status == dqm::qstatus::OTHER)     {
       rval =   0; gval =   0; bval = 255;
      } else if (status == dqm::qstatus::STATUS_OK) {
       rval =   0; gval = 255; bval =   0;
      } else {  
       rval = 255; gval = 255; bval = 255;
      }
     }
     jsSnippet.str("") ;
     jsSnippet << " <DetInfo DetId='"
     	       << detId
     	       << "' red='"
     	       << rval
     	       << "' green='"
     	       << gval
     	       << "' blue='"
     	       << bval
     	       << "' entries='"
     	       << entries
     	       << "'/>" ;
     colorMap.push_back(jsSnippet.str()) ;
//      if( it == me_list.begin()) // The first should be equal to all others...
//      {
//       getNormalization((*it), norm, theMEType.latin1()) ;
//      }
    }
  }

//  delete random ;
  
   cout << ACYellow << ACBold
	<< "[SiPixelInformationExtractor::sendTkUpdatedStatus()] "
	<< ACPlain
	<< "Color map consists of "
	<< colorMap.size()
	<< " snippets: start shipping back"
	<< endl ;

  out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  *out << "<?xml version=\"1.0\" ?>" << endl;
  *out << "<TrackerMapUpdate>"       << endl;

  for(vector<string>::iterator it=colorMap.begin(); it!=colorMap.end(); it++)
  {
   *out << *it << endl;
  }
 
  *out << " <theLimits id=\"normalizationLimits\" normLow=\"" 
       << norm.first 
       << "\" normHigh=\""
       << norm.second 
       << "\" />"
       << endl;
  *out << "</TrackerMapUpdate>"              
       << endl;

   cout << ACYellow << ACBold
	<< "[SiPixelInformationExtractor::sendTkUpdatedStatus()] "
	<< ACPlain
	<< "Color map updated within range " 
	<< norm.first
	<< "-"
	<< norm.second
	<< endl ;

}
*/
//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *  
 *  Given a pointer to ME returns the associated detId 
 */
int SiPixelInformationExtractor::getDetId(MonitorElement * mE) 
{
//cout<<"In SiPixelInformationExtractor::getDetId: for ME= "<<mE->getName()<<endl;
 string mEName = mE->getName();

 int detId = 0;
 
 if( mEName.find("_3") != string::npos )
 {
  string detIdString = mEName.substr((mEName.find_last_of("_"))+1,9);
  //cout<<"string: "<<detIdString<<endl;
  std::istringstream isst;
  isst.str(detIdString);
  isst>>detId;
// } else {
//  cout << ACYellow << ACBold
//       << "[SiPixelInformationExtractor::getDetId()] "
//       << ACPlain
//       << "Could not extract detId from "
//       << mEName
//       << endl ;
 }
  //cout<<"returning with: "<<detId<<endl;
  return detId ;
  
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *  
 */
void SiPixelInformationExtractor::getMEList(DQMStore    * bei,  
					    map<string, int>         & mEHash)
{
  string currDir = bei->pwd();
   
//   cout << ACRed << ACBold
//        << "[SiPixelInformationExtractor::getMEList()]"
//        << ACPlain
//        << " Requesting ME list in " 
//        << currDir
//        << endl ;
       
  string theME ;
   
  // Get ME from Collector/FU0/Tracker/PixelEndcap/HalfCylinder_pX/Disk_X/Blade_XX/Panel_XX/Module_XX
  if (currDir.find("Module_") != string::npos ||
      currDir.find("FED_") != string::npos)  
  {
    vector<string> contents = bei->getMEs(); 
       
    for (vector<string>::const_iterator it = contents.begin(); it != contents.end(); it++) 
    {
      theME = (*it) ;
//       cout << ACRed << ACReverse
//            << "[SiPixelInformationExtractor::getMEList()]"
//            << ACPlain
//            << " ME: " 
//            << (*it)
//            << endl ;
      if(theME.find("siPixel")==string::npos && theME.find("ctfWithMaterialTracks")==string::npos) 
      {
       cout << ACRed << ACBold
            << "[SiPixelInformationExtractor::getMEList()]"
	    << ACPlain
	    << " ----> Skipping " 
	    << (*it)
	    << endl ;
       continue ;
      } // If the ME is not a Pixel one, skip
      string full_path = currDir + "/" + (*it);
      string mEName = theME.substr(0,theME.find_first_of("_"));
      mEHash[mEName]++ ;
    }
    
    return;
  } else {  // If not yet reached the desired level in the directory tree, recursively go down one level more
    vector<string> subdirs = bei->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin(); it != subdirs.end(); it++) 
    {
      bei->cd(*it);
      getMEList(bei, mEHash);
      bei->goUp();
    }
  }
}

//
// -- Get All histograms from a Path
//
/* removing xdaq deps
void SiPixelInformationExtractor::getHistosFromPath(DQMStore * bei, 
                                                    const std::multimap<std::string, std::string>& req_map, 
						    xgi::Output * out){
//cout<<"Entering SiPixelInformationExtractor::getHistosFromPath: "<<endl;
  string path = getItemValue(req_map,"Path");
//cout<<"Path is: "<<path<<endl;
  if (path.size() == 0) return;

  int width  = atoi(getItemValue(req_map, "width").c_str());
  int height = atoi(getItemValue(req_map, "height").c_str());

  string opt =" ";

  setHTMLHeader(out);
  vector<MonitorElement*> all_mes = bei->getContents(path);
  *out << path << " " ;
  for(vector<MonitorElement*>::iterator it=all_mes.begin(); it!=all_mes.end(); it++){
    MonitorElement* me = (*it);
    //cout<<"I'm in the loop now..."<<endl;
    if (!me) continue;
    string name = me->getName();
    string full_path = path + "/" + name;
//cout<<"Calling HP::setNewPlot now for "<<full_path<<endl;
    histoPlotter_->setNewPlot(full_path, opt, width, height);
    *out << name << " ";
  }
//  cout<<"... leaving SiPixelInformationExtractor::getHistosFromPath!"<<endl;
}
*/
/////////////////////////////////////////////////////////////////////////////////////////////////////

void SiPixelInformationExtractor::bookNoisyPixels(DQMStore * bei, float noiseRate_,bool Tier0Flag) {
//std::cout<<"BOOK NOISY PIXEL MEs!"<<std::endl;
  bei->cd();
  if(noiseRate_>=0.){
    bei->setCurrentFolder("Pixel/Barrel");
    EventRateBarrelPixels = bei->book1D("barrelEventRate","Digi event rate for all Barrel pixels",1000,0.,0.01);
    EventRateBarrelPixels->setAxisTitle("Event Rate",1);
    EventRateBarrelPixels->setAxisTitle("Number of Pixels",2);
    bei->cd();  
    bei->setCurrentFolder("Pixel/Endcap");
    EventRateEndcapPixels = bei->book1D("endcapEventRate","Digi event rate for all Endcap pixels",1000,0.,0.01);
    EventRateEndcapPixels->setAxisTitle("Event Rate",1);
    EventRateEndcapPixels->setAxisTitle("Number of Pixels",2);
  }
}

          
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SiPixelInformationExtractor::findNoisyPixels(DQMStore * bei, bool init, float noiseRate_, int noiseRateDenominator_, edm::EventSetup const& eSetup)
{
//cout<<"Entering SiPixelInformationExtractor::findNoisyPixels with noiseRate set to "<<noiseRate_<<endl;

  
  if(init){
    endOfModules_=false;
    nevents_=noiseRateDenominator_;
    if(nevents_ == -1){
      bei->cd();
      bei->setCurrentFolder("Pixel/EventInfo");
      nevents_ = (bei->get("Pixel/EventInfo/processedEvents"))->getIntValue();
    }
    bei->cd();  
    myfile_.open ("NoisyPixelList.txt", ios::app);
    myfile_ << "Noise summary, ran over " << nevents_ << " events, threshold was set to " << noiseRate_ <<  std::endl;
  }
  string currDir = bei->pwd();
  string dname = currDir.substr(currDir.find_last_of("/")+1);


  if(dname.find("Module_")!=string::npos){
    vector<string> meVec = bei->getMEs();
    for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end(); it++) {
      string full_path = currDir + "/" + (*it);
      if(full_path.find("hitmap_siPixelDigis")!=string::npos){
        //broken HV bond:
	//if(currDir.find("HalfCylinder_mI/Disk_1/Blade_01/Panel_2/Module_2")!=string::npos) continue;
        //?noisy?
	//if(currDir.find("HalfCylinder_mI/Disk_1/Blade_12/Panel_1/Module_4")!=string::npos) continue;
        //ROG with HV problem (short?):
	//if(currDir.find("HalfCylinder_mI/Disk_1/Blade_10/Panel_1/Module_3")!=string::npos) continue;
	//if(currDir.find("HalfCylinder_mI/Disk_1/Blade_10/Panel_1/Module_4")!=string::npos) continue;
	//if(currDir.find("HalfCylinder_mI/Disk_1/Blade_10/Panel_2/Module_2")!=string::npos) continue;
	//if(currDir.find("HalfCylinder_mI/Disk_1/Blade_10/Panel_2/Module_3")!=string::npos) continue;
	//if(currDir.find("HalfCylinder_mI/Disk_1/Blade_11/Panel_1/Module_3")!=string::npos) continue;
	//if(currDir.find("HalfCylinder_mI/Disk_1/Blade_11/Panel_1/Module_4")!=string::npos) continue;
	//if(currDir.find("HalfCylinder_mI/Disk_1/Blade_11/Panel_2/Module_2")!=string::npos) continue;
	//if(currDir.find("HalfCylinder_mI/Disk_1/Blade_11/Panel_2/Module_3")!=string::npos) continue;
	//if(currDir.find("HalfCylinder_mI/Disk_1/Blade_12/Panel_1/Module_3")!=string::npos) continue;
	//if(currDir.find("HalfCylinder_mI/Disk_1/Blade_12/Panel_1/Module_4")!=string::npos) continue;
	//if(currDir.find("HalfCylinder_mI/Disk_1/Blade_12/Panel_2/Module_2")!=string::npos) continue;
	//if(currDir.find("HalfCylinder_mI/Disk_1/Blade_12/Panel_2/Module_3")!=string::npos) continue;
        MonitorElement * me = bei->get(full_path);
        if (!me) continue;
	int detid=getDetId(me); int pixcol=-1; int pixrow=-1; 

	//cout<<"detid= "<<detid<<endl;
	std::vector<std::pair<std::pair<int, int>, float> > noisyPixelsInModule;
	TH2F * hothisto = me->getTH2F();
	if(hothisto){
	  for(int i=1; i!=hothisto->GetNbinsX()+1; i++){
	    for(int j=1; j!=hothisto->GetNbinsY()+1; j++){
	      float value = (hothisto->GetBinContent(i,j))/float(nevents_);
	      if(me->getPathname().find("Barrel")!=string::npos){
        	EventRateBarrelPixels = bei->get("Pixel/Barrel/barrelEventRate");
        	if(EventRateBarrelPixels) EventRateBarrelPixels->Fill(value);
	      }else if(me->getPathname().find("Endcap")!=string::npos){
        	EventRateEndcapPixels = bei->get("Pixel/Endcap/endcapEventRate");
        	if(EventRateEndcapPixels) EventRateEndcapPixels->Fill(value);
	      }
	      if(value > noiseRate_){
	        pixcol = i-1;
	        pixrow = j-1;
		//cout<<"pixcol= "<<pixcol<<" , pixrow= "<<pixrow<<" , value= "<<value<<endl;
 
	        std::pair<int, int> address(pixcol, pixrow);
	        std::pair<std::pair<int, int>, float>  PixelStats(address, value);
	        noisyPixelsInModule.push_back(PixelStats);
	      }
            }
	  }
	}
	noisyDetIds_[detid] = noisyPixelsInModule;
	//if(noisyPixelsInModule.size()>=20) cout<<"This module has 20 or more hot pixels: "<<detid<<","<<bei->pwd()<<","<<noisyPixelsInModule.size()<<endl;
      }
    }
  }
  vector<string> subDirVec = bei->getSubdirs();  
  for (vector<string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    if((*ic).find("AdditionalPixelErrors")!=string::npos) continue;
    bei->cd(*ic);
    init=false;
    findNoisyPixels(bei,init,noiseRate_,noiseRateDenominator_,eSetup);
    bei->goUp();
  }

  if(bei->pwd().find("EventInfo")!=string::npos) endOfModules_ = true;
  
  if(!endOfModules_) return;
  // myfile_ <<"am in "<<bei->pwd()<<" now!"<<endl;
  if(currDir == "Pixel/EventInfo/reportSummaryContents"){
    eSetup.get<SiPixelFedCablingMapRcd>().get(theCablingMap);
    std::vector<std::pair<sipixelobjects::DetectorIndex,double> > pixelvec;
    std::map<uint32_t,int> myfedmap;
    std::map<uint32_t,std::string> mynamemap;
    int realfedID = -1;
    //int Nnoisies = noisyDetIds_.size();
    //cout<<"Number of noisy modules: "<<Nnoisies<<endl;
    int counter = 0;
    int n_noisyrocs_all = 0;
    int n_noisyrocs_barrel = 0;
    int n_noisyrocs_endcap = 0;
    int n_verynoisyrocs_all = 0;
    int n_verynoisyrocs_barrel = 0;
    int n_verynoisyrocs_endcap = 0;

    for(int fid = 0; fid < 40; fid++){
    for(std::map<uint32_t, std::vector< std::pair<std::pair<int, int>, float> > >::const_iterator it = noisyDetIds_.begin(); 
        it != noisyDetIds_.end(); it++){
      uint32_t detid = (*it).first;
      std::vector< std::pair<std::pair<int, int>, float> > noisyPixels = (*it).second;
      //cout<<noisyPixels.size()<<" noisy pixels in a module: "<<detid<<endl;
      // now convert into online conventions:
      for(int fedid=0; fedid<=40; ++fedid){
	SiPixelFrameConverter converter(theCablingMap.product(),fedid);
	uint32_t newDetId = detid;
	if(converter.hasDetUnit(newDetId)){
	  realfedID=fedid;
	  break;   
	}
      }
      if(fid == realfedID){
      //cout<<"FED ID is = "<<realfedID<<endl;
      if(realfedID==-1) continue; 
      DetId detId(detid);
      uint32_t detSubId = detId.subdetId();
      std::string outputname;
      bool HalfModule = false;
      if (detSubId == 2){   //FPIX
	PixelEndcapName nameworker(detid);
	outputname = nameworker.name();
      } else if(detSubId == 1){   //BPIX
	PixelBarrelName nameworker(detid);
	outputname = nameworker.name();
	HalfModule = nameworker.isHalfModule();

      } else{
	continue;
      }	
      std::map<int,int> myrocmap;
      myfedmap[detid]=realfedID;
      mynamemap[detid]=outputname;
      
      for(std::vector< std::pair< std::pair<int,int>, float> >::const_iterator pxl = noisyPixels.begin(); 
          pxl != noisyPixels.end(); pxl++){
        std::pair<int,int> offlineaddress = (*pxl).first;
	float Noise_frac = (*pxl).second;
	int offlineColumn = offlineaddress.first;
        int offlineRow = offlineaddress.second;
        counter++;
        //cout<<"noisy pixel counter: "<<counter<<endl;

        sipixelobjects::ElectronicIndex cabling; 
        SiPixelFrameConverter formatter(theCablingMap.product(),realfedID);
        sipixelobjects::DetectorIndex detector = {detid, offlineRow, offlineColumn};      
	formatter.toCabling(cabling,detector);
        // cabling should now contain cabling.roc and cabling.dcol  and cabling.pxid
        // however, the coordinates now need to be converted from dcl,pxid to the row,col coordinates used in the calibration info 
        sipixelobjects::LocalPixel::DcolPxid loc;
        loc.dcol = cabling.dcol;
        loc.pxid = cabling.pxid;
	
	
	// OLD version, not 31X compatible:
//        const sipixelobjects::PixelFEDCabling *theFed= theCablingMap.product()->fed(realfedID);
//	const sipixelobjects::PixelFEDLink * link = theFed->link(cabling.link);
//	const sipixelobjects::PixelROC *theRoc = link->roc(cabling.roc);
//	sipixelobjects::LocalPixel locpixel(loc);
	
	
	// FIX to adhere to new cabling map. To be replaced with CalibTracker/SiPixelTools detid - > hardware id classes ASAP.
	//        const sipixelobjects::PixelFEDCabling *theFed= theCablingMap.product()->fed(realfedID);
	//        const sipixelobjects::PixelFEDLink * link = theFed->link(cabling.link);
	//        const sipixelobjects::PixelROC *theRoc = link->roc(cabling.roc);
        sipixelobjects::LocalPixel locpixel(loc);
        assert(realfedID >= 0);
        assert(cabling.link >= 0);
        assert(cabling.roc >= 0);
	sipixelobjects::CablingPathToDetUnit path = {static_cast<unsigned int>(realfedID), 
                                                     static_cast<unsigned int>(cabling.link),
                                                     static_cast<unsigned int>(cabling.roc)};  
	const sipixelobjects::PixelROC *theRoc = theCablingMap->findItem(path);
	// END of FIX
	
        int onlineColumn = locpixel.rocCol();
        int onlineRow= locpixel.rocRow();
	myrocmap[(theRoc->idInDetUnit())]++;

	// ROC numbers in the barrel go from 8 to 15 instead of 0 to 7 in half modules.  This is a 
	// fix to get the roc number, and add 8 to it if:
	// it's a Barrel module AND on the minus side AND a Half module

	int rocnumber = -1;

	if((detSubId == 1) && (outputname.find("mO")!=string::npos || outputname.find("mI")!=string::npos) && (HalfModule)){
	  rocnumber = theRoc->idInDetUnit() + 8;
	}
	else{
	  rocnumber = theRoc->idInDetUnit();
	}

        //cout<<counter<<" : \t detid= "<<detid<<" , OFF col,row= "<<offlineColumn<<","<<offlineRow<<" , ON roc,col,row= "<<theRoc->idInDetUnit()<<","<<onlineColumn<<","<<onlineRow<<endl;
        myfile_ <<"NAME: "<<outputname<<" , DETID: "<<detid<<" , OFFLINE: col,row: "<<offlineColumn<<","<<offlineRow<<"  \t , ONLINE: roc,col,row: "<<rocnumber<<","<<onlineColumn<<","<<onlineRow<< "  \t , fed,dcol,pixid,link: "<<realfedID<<","<<loc.dcol<<","<<loc.pxid<<","<<cabling.link << ", Noise fraction: " << Noise_frac << std::endl;
      }
      for(std::map<int, int>::const_iterator nrc = myrocmap.begin(); nrc != myrocmap.end(); nrc++){
	if((*nrc).second > 0){
	  n_noisyrocs_all++;
	  if(detSubId == 2){
	    n_noisyrocs_endcap++;
	  } else if(detSubId == 1){
	    n_noisyrocs_barrel++;}
	}
	if((*nrc).second > 40){
	  n_verynoisyrocs_all++;
	  if(detSubId == 2){
	    n_verynoisyrocs_endcap++;
	  } else if(detSubId == 1){
	    n_verynoisyrocs_barrel++;}
	}
      }
      }

    }
    }
    myfile_ << "There are " << n_noisyrocs_all << " noisy ROCs (ROCs with at least 1 noisy pixel) in the entire detector. " << n_noisyrocs_endcap << " are in the FPIX and " << n_noisyrocs_barrel << " are in the BPIX. " << endl;
    myfile_ << "There are " << n_verynoisyrocs_all << " highly noisy ROCs (ROCs with at least 10% of all pixels passing the noise threshold) in the entire detector. " << n_verynoisyrocs_endcap << " are in the FPIX and " << n_verynoisyrocs_barrel << " are in the BPIX. " << endl;

  }
  myfile_.close();
//cout<<"...leaving SiPixelInformationExtractor::findNoisyPixels!"<<endl;
  return;
}


//
// -- Create Images 
//
void SiPixelInformationExtractor::createImages(DQMStore* bei){
  histoPlotter_->createPlots(bei);
}

//
// -- Set HTML Header in xgi output
//
/* removing xdaq deps
void SiPixelInformationExtractor::setHTMLHeader(xgi::Output * out) {
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/html");
  out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
  out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
  out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
}
*/
//
// -- Set XML Header in xgi output
//
/* removing xdaq deps
void SiPixelInformationExtractor::setXMLHeader(xgi::Output * out) {
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
  out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
  out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
  *out << "<?xml version=\"1.0\" ?>" << std::endl;

}
*/
//
// -- Set Plain Header in xgi output
//
/* removing xdaq deps
void SiPixelInformationExtractor::setPlainHeader(xgi::Output * out) {
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/plain");
  out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
  out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
  out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");

}
*/
