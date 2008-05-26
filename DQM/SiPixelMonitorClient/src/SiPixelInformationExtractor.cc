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
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/WebComponents/interface/CgiReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
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

#include <qstring.h>
#include <qregexp.h>

#include <iostream>
#include <math.h>

#include <cstdlib> // for free() - Root can allocate with malloc() - sigh...
 
using namespace std;
using namespace edm;

//------------------------------------------------------------------------------
/*! \brief Constructor of the SiPixelInformationExtractor class.
 *  
 */
SiPixelInformationExtractor::SiPixelInformationExtractor() {
  edm::LogInfo("SiPixelInformationExtractor") << 
    " Creating SiPixelInformationExtractor " << "\n" ;
  
  canvas_ = new TCanvas("PlotCanvas", "Plot Canvas"); 
  readReference_ = false;
  allMods_=0;
  errorMods_=0;
  qflag_=1.;
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
  
  if (canvas_) delete canvas_;
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
void SiPixelInformationExtractor::getSingleModuleHistos(DQMStore * bei, 
                                                        const multimap<string, string>& req_map, 
							xgi::Output * out){

  vector<string> hlist;
  getItemList(req_map,"histo", hlist);

  uint32_t detId = atoi(getItemValue(req_map,"ModId").c_str());
 
  int width  = atoi(getItemValue(req_map, "width").c_str());
  int height = atoi(getItemValue(req_map, "height").c_str());

  string opt =" ";
  
  SiPixelFolderOrganizer folder_organizer;
  string path;
  folder_organizer.getModuleFolder(detId,path);   

  if((bei->pwd()).find("Module_") == string::npos &&
     (bei->pwd()).find("FED_") == string::npos){
    cout<<"This is not a pixel module or FED!"<<endl;
    return;
  }
 
  vector<MonitorElement*> all_mes = bei->getContents(path);
  setHTMLHeader(out);
  *out << path << " ";

  QRegExp rx("(\\w+)_(siPixel|ctfWithMaterialTracks)") ;
  QString theME ;

  for (vector<string>::const_iterator ih = hlist.begin();
       ih != hlist.end(); ih++) {
    for (vector<MonitorElement *>::const_iterator it = all_mes.begin();
	 it!= all_mes.end(); it++) {
      MonitorElement * me = (*it);
      if (!me) continue;
      theME = me->getName();
      string temp_s ; 
      if( rx.search(theME) != -1 ) { temp_s = rx.cap(1).latin1() ; }
      if (temp_s == (*ih)) {
	string full_path = path + "/" + me->getName();
	histoPlotter_->setNewPlot(full_path, opt, width, height);
	*out << me->getName() << " " ;
      }
    }
  }
}

//
// -- Plot Tracker Map MEs
//
void SiPixelInformationExtractor::getTrackerMapHistos(DQMStore* bei, 
                                                      const std::multimap<std::string, std::string>& req_map, 
						      xgi::Output * out) {

  vector<string> hlist;
  string tkmap_name;
  SiPixelConfigParser config_parser;
  string localPath = string("DQM/SiPixelMonitorClient/test/sipixel_monitorelement_config.xml");
  config_parser.getDocument(edm::FileInPath(localPath).fullPath());
  if (!config_parser.getMENamesForTrackerMap(tkmap_name, hlist)) return;
  if (hlist.size() == 0) return;

  uint32_t detId = atoi(getItemValue(req_map,"ModId").c_str());
 
  int width  = atoi(getItemValue(req_map, "width").c_str());
  int height = atoi(getItemValue(req_map, "height").c_str());

  string opt =" ";
  
  SiPixelFolderOrganizer folder_organizer;
  string path;
  folder_organizer.getModuleFolder(detId,path);   

  if((bei->pwd()).find("Module_") == string::npos &&
     (bei->pwd()).find("FED_") == string::npos){
    cout<<"This is not a pixel module or FED!"<<endl;
    return;
  }
 
  vector<MonitorElement*> all_mes = bei->getContents(path);
  setHTMLHeader(out);
  *out << path << " ";

  QRegExp rx("(\\w+)_(siPixel|ctfWithMaterialTracks)") ;
  QString theME ;

  for (vector<string>::iterator ih = hlist.begin();
       ih != hlist.end(); ih++) {
    for (vector<MonitorElement *>::const_iterator it = all_mes.begin();
	 it!= all_mes.end(); it++) {
      MonitorElement * me = (*it);
      if (!me) continue;
      theME = me->getName();
      string temp_s ; 
      if( rx.search(theME) != -1 ) { temp_s = rx.cap(1).latin1() ; }
      if (temp_s == (*ih)) {
	string full_path = path + "/" + me->getName();
	histoPlotter_->setNewPlot(full_path, opt, width, height);
	*out << me->getName() << " " ;
      }      
    }
  }   
}

//============================================================================================================
// --  Return type of ME
//
std::string  SiPixelInformationExtractor::getMEType(MonitorElement * theMe)
{
  QString qtype = theMe->getRootObject()->IsA()->GetName() ;
  if(         qtype.contains("TH1") > 0 )
  {
    return "TH1" ;
  } else if ( qtype.contains("TH2") > 0  ) {
    return "TH2" ;
  } else if ( qtype.contains("TH3") > 0 ) {
    return "TH3" ;
  }
  return "TH1" ;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
void SiPixelInformationExtractor::readModuleAndHistoList(DQMStore* bei, 
                                                         xgi::Output * out) {
//cout<<"entering SiPixelInformationExtractor::readModuleAndHistoList"<<endl;
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
  if (currDir.find("Module_") != string::npos ||
      currDir.find("FED_") != string::npos)  {
    if (histos.size() == 0) {
      //cout<<"currDir="<<currDir<<endl;

      vector<string> contents = bei->getMEs();
          
      for (vector<string>::const_iterator it = contents.begin();
	   it != contents.end(); it++) {
	string hname          = (*it).substr(0, (*it).find("_siPixel"));
	if (hname==" ") hname = (*it).substr(0, (*it).find("_ctfWithMaterialTracks"));
        string fullpathname   = bei->pwd() + "/" + (*it); 

        MonitorElement * me   = bei->get(fullpathname);
	
        string htype          = "undefined" ;
        if (me) 
	{
	 htype = me->getRootObject()->IsA()->GetName() ;
	}
	//cout<<"hname="<<hname<<endl;
        histos[hname] = htype ;
        string mId=" ";
	if(hname.find("ndigis")                !=string::npos) mId = (*it).substr((*it).find("ndigis_siPixelDigis_")+20, 9);
	if(mId==" " && hname.find("nclusters") !=string::npos) mId = (*it).substr((*it).find("nclusters_siPixelClusters_")+26, 9);
        if(mId==" " && hname.find("residualX") !=string::npos) mId = (*it).substr((*it).find("residualX_ctfWithMaterialTracks_")+32, 9);
        if(mId==" " && hname.find("NErrors") !=string::npos) mId = (*it).substr((*it).find("NErrors_siPixelDigis_")+21, 9);
        if(mId==" " && hname.find("ClustX") !=string::npos) mId = (*it).substr((*it).find("ClustX_siPixelRecHit_")+21, 9);
        if(mId==" " && hname.find("pixelAlive") !=string::npos) mId = (*it).substr((*it).find("pixelAlive_siPixelCalibDigis_")+29, 9);
        if(mId==" " && hname.find("Gain1d") !=string::npos) mId = (*it).substr((*it).find("Gain1d_siPixelCalibDigis_")+25, 9);
        if(mId!=" ") modules.push_back(mId);
        //cout<<"mId="<<mId<<endl;
      }    
    }
  } else {  
    vector<string> subdirs = bei->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
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

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
void SiPixelInformationExtractor::printModuleHistoList(DQMStore * bei, 
                                                       ostringstream& str_val){
//cout<<"entering SiPixelInformationExtractor::printModuleHistoList"<<endl;
  static string indent_str = "";
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
      QString qit = (*it) ;
      QRegExp rx("(\\w+)_(siPixel|ctfWithMaterialTracks)") ;
      if( rx.search(qit) > -1 ) {qit = rx.cap(1);} 
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
//	      <<        qit << "\n"
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
  static string indent_str = "";
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
      QString qit = (*it) ;
      QRegExp rx("(\\w+)_(siPixel|ctfWithMaterialTracks)");
      if( rx.search(qit) > -1 ) {qit = rx.cap(1);} 
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
	      <<        qit << "\n"
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
  static string indent_str = "";
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
        QString qit = (*it) ;
        QRegExp rx("(\\w+)_(siPixel|ctfWithMaterialTracks)") ;
        if( rx.search(qit) > -1 ) {qit = rx.cap(1);} 
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
        	<<	  qit << "\n"
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
void SiPixelInformationExtractor::getIMGCImage(const multimap<string, string>& req_map, 
                                               xgi::Output * out){
//cout<<"Entering SiPixelInformationExtractor::getIMGCImage: "<<endl;
  string path = getItemValue(req_map,"Path");
  //string plot = getItemValue(req_map,"Plot");
  //string folder = getItemValue(req_map,"Folder");
  //string path = folder + "/" + plot;
  string image;
//  cout<<"... trying to getNamedImageBuffer for path "<<path<<endl;
  histoPlotter_->getNamedImageBuffer(path, image);

  out->getHTTPResponseHeader().addHeader("Content-Type", "image/png");
  out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
  out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
  out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
  *out << image;
//cout<<"... leaving SiPixelInformationExtractor::getIMGCImage!"<<endl;
}

void SiPixelInformationExtractor::getIMGCImage(multimap<string, string>& req_map, 
                                               xgi::Output * out){
  
  string path = getItemValue(req_map,"Path");
  //string plot = getItemValue(req_map,"Plot");
  //string folder = getItemValue(req_map,"Folder");
  //string path = folder + "/" + plot;
  string image;
//  cout<<"I am in getIMGCImage now and trying to getNamedImageBuffer for path "<<path<<endl;
  histoPlotter_->getNamedImageBuffer(path, image);

  out->getHTTPResponseHeader().addHeader("Content-Type", "image/png");
  out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
  out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
  out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
  *out << image;

}


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

  QString theMEType = getMEType(theME) ;

//   cout << ACRed << ACReverse
//        << "[SiPixelInformationExtractor::computeStatus()]"
//        << ACPlain
//        << " Computing average for "
//        << theME->getName()
//        << endl ;

  if( theMEType.contains("TH1") > 0 )
  {
   meanX = (double)theME->getMean();
   getNormalization(theME, normX, "TH1") ;
   normalizationX = fabs( normX.second - normX.first) ;
   if( normalizationX == 0 ) {normalizationX=1.E-20;}
   colorValue  = meanX / normalizationX ;
   norm.first  = normX.first ;
   norm.second = normX.second ;
  }
  
  if( theMEType.contains("TH2") > 0 )
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
						   QString              theMEType) 
{
  double normLow  = 0 ;
  double normHigh = 0 ;

  if( theMEType.contains("TH1") > 0 )
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
						     QString              theMEType) 
{
  double normLow  = 0 ;
  double normHigh = 0 ;

  if( theMEType.contains("TH2") > 0 )
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
  string currDir = bei->pwd();
   
  QRegExp rx("(\\w+)_(siPixel|ctfWithMaterialTracks)") ;
  QString theME ;
   
  // Get ME from Collector/FU0/Tracker/PixelEndcap/HalfCylinder_pX/Disk_X/Blade_XX/Panel_XX/Module_XX
  if (currDir.find("Module_") != string::npos ||
      currDir.find("FED_") != string::npos)  
  {
    vector<string> contents = bei->getMEs(); 
       
    for (vector<string>::const_iterator it = contents.begin(); it != contents.end(); it++) 
    {
      theME = *it ;
      if( rx.search(theME) == -1 ) {continue ;} // If the ME is not a siPixel or ctfWithMaterialTrack one, skip
      if (rx.cap(1).latin1() == theMEName)  
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
void SiPixelInformationExtractor::sendTkUpdatedStatus(DQMStore  * bei, 
                                                      xgi::Output            * out,
						      std::string            & theMEName,
						      std::string            & theTKType) 
{
  int rval, gval, bval;
  vector<string>          colorMap ;
  vector<MonitorElement*> me_list;
  pair<double,double>     norm ;
  double sts ;
    
  bei->cd();
  selectMEList(bei, theMEName, me_list) ;
  bei->cd();

  string detId = "undefined";

  QRegExp rx("\\w+_\\w+_(\\d+)");
  
//   cout << ACYellow << ACBold
//        << "[SiPixelInformationExtractor::sendTkUpdatedStatus()] "
//        << ACPlain
//        << "Preparing color map update for " 
//        << theMEName
//        << " type "
//        << theTKType
//        << " - List size: "
//        << me_list.size() 
//        << endl ;
  
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
    QString meName    = (*it)->getName() ;
    QString theMEType = getMEType(*it) ;
    if( rx.search(meName) != -1 ) 
    {
     detId = rx.cap(1).latin1() ;
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
  
//   cout << ACYellow << ACBold
//        << "[SiPixelInformationExtractor::sendTkUpdatedStatus()] "
//        << ACPlain
//        << "Color map consists of "
//        << colorMap.size()
//        << " snippets: start shipping back"
//        << endl ;

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

//   cout << ACYellow << ACBold
//        << "[SiPixelInformationExtractor::sendTkUpdatedStatus()] "
//        << ACPlain
//        << "Color map updated within range " 
//        << norm.first
//        << "-"
//        << norm.second
//        << endl ;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *  
 *  Given a pointer to ME returns the associated detId 
 */
int SiPixelInformationExtractor::getDetId(MonitorElement * mE) 
{
 QRegExp rx("(\\w+)_(\\w+)_(\\d+)") ;
 QString mEName = mE->getName() ;

 int detId = 0;
 
 if( rx.search(mEName) != -1 )
 {
  detId = rx.cap(3).toInt() ;
 } else {
  cout << ACYellow << ACBold
       << "[SiPixelInformationExtractor::getDetId()] "
       << ACPlain
       << "Could not extract detId from "
       << mEName
       << endl ;
 }
      
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
       
  QRegExp rx("(\\w+)_(siPixel|ctfWithMaterialTracks)") ;
  QString theME ;
   
  // Get ME from Collector/FU0/Tracker/PixelEndcap/HalfCylinder_pX/Disk_X/Blade_XX/Panel_XX/Module_XX
  if (currDir.find("Module_") != string::npos ||
      currDir.find("FED_") != string::npos)  
  {
    vector<string> contents = bei->getMEs(); 
       
    for (vector<string>::const_iterator it = contents.begin(); it != contents.end(); it++) 
    {
      theME = *it ;
//       cout << ACRed << ACReverse
//            << "[SiPixelInformationExtractor::getMEList()]"
//            << ACPlain
//            << " ME: " 
//            << theME
//            << endl ;
      if( rx.search(theME) == -1 ) 
      {
       cout << ACRed << ACBold
            << "[SiPixelInformationExtractor::getMEList()]"
	    << ACPlain
	    << " ----> Skipping " 
	    << theME
	    << endl ;
       continue ;
      } // If the ME is not a Pixel one, skip
      string full_path = currDir + "/" + (*it);
      string mEName = rx.cap(1).latin1() ;
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

void SiPixelInformationExtractor::bookGlobalQualityFlag(DQMStore * bei) {
//std::cout<<"BOOK GLOBAL QUALITY FLAG MEs!"<<std::endl;
  bei->cd();
  bei->setCurrentFolder("Pixel/EventInfo");
  SummaryReport = bei->bookFloat("reportSummary");
  SummaryReportMap = bei->book2D("reportSummaryMap","Pixel EtaPhi Summary Map",60,-3.,3.,64,-3.2,3.2);
  SummaryReportMap->setAxisTitle("Eta",1);
  SummaryReportMap->setAxisTitle("Phi",2);
  bei->setCurrentFolder("Pixel/EventInfo/reportSummaryContents");
  SummaryBarrel = bei->bookFloat("SummaryBarrel");
  SummaryShellmI = bei->bookFloat("SummaryShellmI");
  SummaryShellmO = bei->bookFloat("SummaryShellmO");
  SummaryShellpI = bei->bookFloat("SummaryShellpI");
  SummaryShellpO = bei->bookFloat("SummaryShellpO");
  SummaryEndcap = bei->bookFloat("SummaryEndcap");
  SummaryHCmI = bei->bookFloat("SummaryHCmI");
  SummaryHCmO = bei->bookFloat("SummaryHCmO");
  SummaryHCpI = bei->bookFloat("SummaryHCpI");
  SummaryHCpO = bei->bookFloat("SummaryHCpO");
}

void SiPixelInformationExtractor::computeGlobalQualityFlag(DQMStore * bei, 
                                                           bool init)
{
//cout<<"entering SiPixelInformationExtractor::ComputeGlobalQualityFlag"<<endl;
//   cout << ACRed << ACBold
//        << "[SiPixelInformationExtractor::ComputeGlobalQualityFlag]"
//        << ACPlain
//        << " Enter" 
//        << endl ;
  if(init){
    allMods_=0; errorMods_=0; qflag_=0.; 
    bpix_mods_=0; err_bpix_mods_=0; bpix_flag_=0.;
    shellmI_mods_=0; err_shellmI_mods_=0; shellmI_flag_=0.;
    shellmO_mods_=0; err_shellmO_mods_=0; shellmO_flag_=0.;
    shellpI_mods_=0; err_shellpI_mods_=0; shellpI_flag_=0.;
    shellpO_mods_=0; err_shellpO_mods_=0; shellpO_flag_=0.;
    fpix_mods_=0; err_fpix_mods_=0; fpix_flag_=0.;
    hcylmI_mods_=0; err_hcylmI_mods_=0; hcylmI_flag_=0.;
    hcylmO_mods_=0; err_hcylmO_mods_=0; hcylmO_flag_=0.;
    hcylpI_mods_=0; err_hcylpI_mods_=0; hcylpI_flag_=0.;
    hcylpO_mods_=0; err_hcylpO_mods_=0; hcylpO_flag_=0.;
  }
  
  string currDir = bei->pwd();
  string dname = currDir.substr(currDir.find_last_of("/")+1);
  
  QRegExp rx("Module_");
 
  if(rx.search(dname)!=-1){
    if(currDir.find("Pixel")!=string::npos) allMods_++;
    if(currDir.find("Barrel")!=string::npos) bpix_mods_++;
    if(currDir.find("Shell_mI")!=string::npos) shellmI_mods_++;
    if(currDir.find("Shell_mO")!=string::npos) shellmO_mods_++;
    if(currDir.find("Shell_pI")!=string::npos) shellpI_mods_++;
    if(currDir.find("Shell_pO")!=string::npos) shellpO_mods_++;
    if(currDir.find("Endcap")!=string::npos) fpix_mods_++;
    if(currDir.find("HalfCylinder_mI")!=string::npos) hcylmI_mods_++;
    if(currDir.find("HalfCylinder_mO")!=string::npos) hcylmO_mods_++;
    if(currDir.find("HalfCylinder_pI")!=string::npos) hcylpI_mods_++;
    if(currDir.find("HalfCylinder_pO")!=string::npos) hcylpO_mods_++;
      
    vector<string> meVec = bei->getMEs();
    for (vector<string>::const_iterator it = meVec.begin();
	 it != meVec.end(); it++) {
      string full_path = currDir + "/" + (*it);
      MonitorElement * me = bei->get(full_path);
    
      if (!me) continue;
      std::vector<QReport *> my_map = me->getQReports();
      if (my_map.size() > 0) {
        string image_name;
        selectImage(image_name,my_map);
        if(image_name!="images/LI_green.gif") {
          errorMods_++;
          if(currDir.find("Pixel")!=string::npos) errorMods_++;
          if(currDir.find("Barrel")!=string::npos) err_bpix_mods_++;
          if(currDir.find("Shell_mI")!=string::npos) err_shellmI_mods_++;
          if(currDir.find("Shell_mO")!=string::npos) err_shellmO_mods_++;
          if(currDir.find("Shell_pI")!=string::npos) err_shellpI_mods_++;
          if(currDir.find("Shell_pO")!=string::npos) err_shellpO_mods_++;
          if(currDir.find("Endcap")!=string::npos) err_fpix_mods_++;
          if(currDir.find("HalfCylinder_mI")!=string::npos) err_hcylmI_mods_++;
          if(currDir.find("HalfCylinder_mO")!=string::npos) err_hcylmO_mods_++;
          if(currDir.find("HalfCylinder_pI")!=string::npos) err_hcylpI_mods_++;
          if(currDir.find("HalfCylinder_pO")!=string::npos) err_hcylpO_mods_++;
        }	
      }
    }
  }
  if(allMods_>0) qflag_ = (float(allMods_)-float(errorMods_))/float(allMods_);
  if(bpix_mods_>0) bpix_flag_ = (float(bpix_mods_)-float(err_bpix_mods_))/float(bpix_mods_);
  if(shellmI_mods_>0) shellmI_flag_ = (float(shellmI_mods_)-float(err_shellmI_mods_))/float(shellmI_mods_);
  if(shellmO_mods_>0) shellmO_flag_ = (float(shellmO_mods_)-float(err_shellmO_mods_))/float(shellmO_mods_);
  if(shellpI_mods_>0) shellpI_flag_ = (float(shellpI_mods_)-float(err_shellpI_mods_))/float(shellpI_mods_);
  if(shellpO_mods_>0) shellpO_flag_ = (float(shellpO_mods_)-float(err_shellpO_mods_))/float(shellpO_mods_);
  if(fpix_mods_>0) fpix_flag_ = (float(fpix_mods_)-float(err_fpix_mods_))/float(fpix_mods_);
  if(hcylmI_mods_>0) hcylmI_flag_ = (float(hcylmI_mods_)-float(err_hcylmI_mods_))/float(hcylmI_mods_);
  if(hcylmO_mods_>0) hcylmO_flag_ = (float(hcylmO_mods_)-float(err_hcylmO_mods_))/float(hcylmO_mods_);
  if(hcylpI_mods_>0) hcylpI_flag_ = (float(hcylpI_mods_)-float(err_hcylpI_mods_))/float(hcylpI_mods_);
  if(hcylpO_mods_>0) hcylpO_flag_ = (float(hcylpO_mods_)-float(err_hcylpO_mods_))/float(hcylpO_mods_);
  
  vector<string> subDirVec = bei->getSubdirs();  
  for (vector<string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    bei->cd(*ic);
    init=false;
    computeGlobalQualityFlag(bei,init);
    bei->goUp();
  }
  bei->cd();
  bei->setCurrentFolder("Pixel/EventInfo");
  SummaryReport = bei->get("Pixel/EventInfo/reportSummary");
  if(SummaryReport) SummaryReport->Fill(qflag_);
  bei->setCurrentFolder("Pixel/EventInfo/reportSummaryContents"); 
  SummaryBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/SummaryBarrel");
  if(SummaryBarrel) SummaryBarrel->Fill(bpix_flag_);
  SummaryShellmI = bei->get("Pixel/EventInfo/reportSummaryContents/SummaryShellmI");
  if(SummaryShellmI) SummaryShellmI->Fill(shellmI_flag_);
  SummaryShellmO = bei->get("Pixel/EventInfo/reportSummaryContents/SummaryShellmO");
  if(SummaryShellmO)   SummaryShellmO->Fill(shellmO_flag_);
  SummaryShellpI = bei->get("Pixel/EventInfo/reportSummaryContents/SummaryShellpI");
  if(SummaryShellpI)   SummaryShellpI->Fill(shellpI_flag_);
  SummaryShellpO = bei->get("Pixel/EventInfo/reportSummaryContents/SummaryShellpO");
  if(SummaryShellpO)   SummaryShellpO->Fill(shellpO_flag_);
  SummaryEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/SummaryEndcap");
  if(SummaryEndcap)   SummaryEndcap->Fill(fpix_flag_);
  SummaryHCmI = bei->get("Pixel/EventInfo/reportSummaryContents/SummaryHCmI");
  if(SummaryHCmI)   SummaryHCmI->Fill(hcylmI_flag_);
  SummaryHCmO = bei->get("Pixel/EventInfo/reportSummaryContents/SummaryHCmO");
  if(SummaryHCmO)   SummaryHCmO->Fill(hcylmO_flag_);
  SummaryHCpI = bei->get("Pixel/EventInfo/reportSummaryContents/SummaryHCpI");
  if(SummaryHCpI)   SummaryHCpI->Fill(hcylpI_flag_);
  SummaryHCpO = bei->get("Pixel/EventInfo/reportSummaryContents/SummaryHCpO");
  if(SummaryHCpO)   SummaryHCpO->Fill(hcylpO_flag_);

}

void SiPixelInformationExtractor::fillGlobalQualityPlot(DQMStore * bei, bool init, edm::EventSetup const& eSetup)
{
  //calculate eta and phi of the modules and fill a 2D plot:
  
  if(init){
    allmodsEtaPhi = new TH2F("allmodsEtaPhi","allmodsEtaPhi",60,-3.,3.,64,-3.2,3.2);
    errmodsEtaPhi = new TH2F("errmodsEtaPhi","errmodsEtaPhi",60,-3.,3.,64,-3.2,3.2);
    goodmodsEtaPhi = new TH2F("goodmodsEtaPhi","goodmodsEtaPhi",60,-3.,3.,64,-3.2,3.2);
    count=0; errcount=0;
    init=false;
  }
  string currDir = bei->pwd();
  string dname = currDir.substr(currDir.find_last_of("/")+1);
  
  QRegExp rx("Module_");
 
  if(rx.search(dname)!=-1){
    vector<string> meVec = bei->getMEs();
    float detEta=-5.; float detPhi=-5.;
    bool first=true; bool once=true;
    for (vector<string>::const_iterator it = meVec.begin();
	 it != meVec.end(); it++) {
      if(!once) continue;
      string full_path = currDir + "/" + (*it);
      MonitorElement * me = bei->get(full_path);
      if (!me) continue;
      int id=0;
      if(first){ id = getDetId(me); first=false; }
      DetId detid = DetId(id);
      if(detid.det()!=1) continue;
      edm::ESHandle<TrackerGeometry> pDD;
      eSetup.get<TrackerDigiGeometryRecord>().get( pDD );
      for(TrackerGeometry::DetContainer::const_iterator it = 
	  pDD->dets().begin(); it != pDD->dets().end(); it++){
        if(dynamic_cast<PixelGeomDetUnit*>((*it))!=0){
          DetId detId = (*it)->geographicalId();
	  if(detId!=detid) continue;
	  //if(detId.subdetId()!=1) continue;
          const GeomDetUnit * geoUnit = pDD->idToDetUnit( detId );
          const PixelGeomDetUnit * pixDet = dynamic_cast<const PixelGeomDetUnit*>(geoUnit);
	  float detR = pixDet->surface().position().perp();
	  float detZ = pixDet->surface().position().z();
	  detPhi = pixDet->surface().position().phi();
	  detEta = -1.*log(tan(atan2(detR,detZ)/2.));
	  cout<<"Module: "<<currDir<<" , Eta= "<<detEta<<" , Phi= "<<detPhi<<endl;
	  once=false;
	}
      }
    }  
    //got module ID and eta and phi now! time to count:
    count++;
    allmodsEtaPhi->Fill(detEta,detPhi);
    bool anyerr=false;
    for (vector<string>::const_iterator it = meVec.begin();
	 it != meVec.end(); it++){
      if(anyerr) continue;
      string full_path = currDir + "/" + (*it);
      MonitorElement * me = bei->get(full_path);
      if (!me) continue;
      if(me->hasError()||me->hasWarning()||me->hasOtherReport()) anyerr=true;
    }
    if(anyerr){
      errcount++;
      errmodsEtaPhi->Fill(detEta,detPhi);
    }
  }

  vector<string> subDirVec = bei->getSubdirs();  
  for (vector<string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    bei->cd(*ic);
    init=false;
    fillGlobalQualityPlot(bei,init,eSetup);
    bei->goUp();
  }
  
  SummaryReportMap = bei->get("Pixel/EventInfo/reportSummaryMap");
  if(SummaryReportMap){ 
    float contents=0.;
    for(int i=1; i!=61; i++)for(int j=1; j!=65; j++){
      contents = (allmodsEtaPhi->GetBinContent(i,j))-(errmodsEtaPhi->GetBinContent(i,j));
      //cout<<"all: "<<allmodsEtaPhi->GetBinContent(i,j)<<" , error: "<<errmodsEtaPhi->GetBinContent(i,j)<<" , contents: "<<contents<<endl;
      goodmodsEtaPhi->SetBinContent(i,j,contents);
      if(allmodsEtaPhi->GetBinContent(i,j)>0){
        //contents = allmodsEtaPhi->GetBinContent(i,j);
	contents = (goodmodsEtaPhi->GetBinContent(i,j))/(allmodsEtaPhi->GetBinContent(i,j));
      }else{
        contents = -1.;
      }
      SummaryReportMap->setBinContent(i,j,contents);
    }
  }
  //cout<<"counters: "<<count<<" , "<<errcount<<endl;
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
void SiPixelInformationExtractor::setHTMLHeader(xgi::Output * out) {
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/html");
  out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
  out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
  out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
}
//
// -- Set XML Header in xgi output
//
void SiPixelInformationExtractor::setXMLHeader(xgi::Output * out) {
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
  out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
  out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
  *out << "<?xml version=\"1.0\" ?>" << std::endl;

}
//
// -- Set Plain Header in xgi output
//
void SiPixelInformationExtractor::setPlainHeader(xgi::Output * out) {
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/plain");
  out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
  out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
  out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");

}
