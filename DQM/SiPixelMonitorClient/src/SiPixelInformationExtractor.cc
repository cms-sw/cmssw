/*! \file SiPixelInformationExtractor.cc
 *
 *  \brief This class represents ...
 *  
 *  (Documentation under development)
 *  
 */
#include "DQM/SiPixelMonitorClient/interface/SiPixelInformationExtractor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelUtility.h"
#include "DQM/SiPixelMonitorClient/interface/ANSIColors.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/WebComponents/interface/CgiReader.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"

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

#include <qstring.h>
#include <qregexp.h>

#include <iostream>

#include <cstdlib> // for free() - Root can allocate with malloc() - sigh...
 
using namespace std;

//------------------------------------------------------------------------------
/*! \brief Constructor of the SiPixelInformationExtractor class.
 *  
 */
SiPixelInformationExtractor::SiPixelInformationExtractor() {
  edm::LogInfo("SiPixelInformationExtractor") << 
    " Creating SiPixelInformationExtractor " << "\n" ;
  
  canvas_ = new TCanvas("PlotCanvas", "Plot Canvas"); 
  paveOnCanvas = new TPaveText(0.7,0.75,0.99,0.94,"NDCtr");   

  // variables used for reading layout names
  // ---------------------------------------
  layoutParser_ = 0;
  readReference_ = false;
  layoutMap.clear();

  // variables used to store qtest information
  // for the slide show plot
  //  -----------------------------------------
  qtestsParser_ = 0;
  readQTestMap_  = false;
  readMeMap_     = false;
  qtestsMap.clear();
  meQTestsMap.clear();

  // function used for reading configuration files [layout and qtest]
  // ----------------------------------------------------------------
  readConfiguration();

}

//------------------------------------------------------------------------------
/*! \brief Destructor of the SiPixelInformationExtractor class.
 *  
 */
SiPixelInformationExtractor::~SiPixelInformationExtractor() {
  edm::LogInfo("SiPixelInformationExtractor") << 
    " Deleting SiPixelInformationExtractor " << "\n" ;
  
  if (canvas_) delete canvas_;
  if (paveOnCanvas)    delete paveOnCanvas;

  if (layoutParser_) delete layoutParser_;
  if (qtestsParser_) delete qtestsParser_;
}

//------------------------------------------------------------------------------
/*! \brief Read Configurationn File
 *
 */
void SiPixelInformationExtractor::readConfiguration() {
  //  cout << "entering in SiPixelInformationExtractor::readConfiguration" << endl;

  // read layout configuration file
  // ------------------------------
  string localPath = string("DQM/SiPixelMonitorClient/test/sipixel_plot_layout_config.xml");
  if (layoutParser_ == 0) {
    layoutParser_ = new SiPixelLayoutParser();
    layoutParser_->getDocument(edm::FileInPath(localPath).fullPath());
  }
  if (layoutParser_->getAllLayouts(layoutMap)) {
     edm::LogInfo("SiPixelInformationExtractor") << 
                  " Layouts correctly readout " << "\n" ;
     //     cout << "SiPixelInformationExtractor: correctly Layout readout " << endl;
  } else {
    edm::LogInfo("SiPixelInformationExtractor") << 
                 " Problem in reading Layout " << "\n" ;
    //    cout << "SiPixelInformationExtractor: Problem in reading Layout " << endl;
  }
  if (layoutParser_) delete layoutParser_;
  
  // create default .png files for the slide show [Plot not ready yet!!]
  // -------------------------------------------------------------------
  createDummiesFromLayout();


  // read quality test configuration file
  // ------------------------------------
  localPath = string("DQM/SiPixelMonitorClient/test/sipixel_qualitytest_config.xml");
  if (qtestsParser_ == 0) {
    qtestsParser_ = new SiPixelQTestsParser();
    qtestsParser_->getDocument(edm::FileInPath(localPath).fullPath());
  }
  if (qtestsParser_->getAllQTests(qtestsMap)){
     edm::LogInfo("SiPixelInformationExtractor") << 
                  " QTestsMap correctly readout " << "\n" ;
     //     cout << "SiPixelInformationExtractor: correctly QTestsMap readout " << endl;
     readQTestMap_ = true;
  } else {
    edm::LogInfo("SiPixelInformationExtractor") << 
                 " Problem in reading QTestsMap " << "\n" ;
    //    cout << "SiPixelInformationExtractor: Problem in reading QTestsMap " << endl;
  }
  if (qtestsParser_->monitorElementTestsMap(meQTestsMap)){
    edm::LogInfo("SiPixelInformationExtractor") << 
                 " ME-QTestsMap correctly readout " << "\n" ;
    //    cout << "SiPixelInformationExtractor: correctly ME-QTestsMap readout " << endl;
    readMeMap_ = true;
  } else {
    edm::LogInfo("SiPixelInformationExtractor") << 
                 " Problem in reading ME-QTestsMap " << "\n" ;
    //    cout << "SiPixelInformationExtractor: Problem in reading ME-QTestsMap " << endl;
  }
  if (qtestsParser_) delete qtestsParser_;
  
  cout << "..leaving SiPixelInformationExtractor::readConfiguration" << endl;
}
//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *  
 */
//void SiPixelInformationExtractor::createModuleTree(MonitorUserInterface* mui) {
void SiPixelInformationExtractor::createModuleTree(DaqMonitorBEInterface* bei) {
//cout<<"entering SiPixelInformationExtractor::createModuleTree..."<<endl;
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  string structure_name;
  vector<string> me_names;
  if (!configParser_->getMENamesForTree(structure_name, me_names)){
    cout << "SiPixelInformationExtractor::createModuleTree: Failed to read Tree configuration parameters!! ";
    return;
  }
  bei->cd();
  fillBarrelList(bei, structure_name, me_names);
  bei->cd();
  fillEndcapList(bei, structure_name, me_names);
  bei->cd();
  actionExecutor_->createLayout(bei);
  string fname = "test1.xml";
  configWriter_->write(fname);
  if (configWriter_) delete configWriter_;
  configWriter_ = 0;
//cout<<"leaving SiPixelInformationExtractor::createModuleTree..."<<endl;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *  
 */
//void SiPixelInformationExtractor::fillBarrelList(MonitorUserInterface* mui,
void SiPixelInformationExtractor::fillBarrelList(DaqMonitorBEInterface* bei,
                                                 string dir_name,
						 vector<string>& me_names) {
  //cout<<"entering SiPixelInformationExtractor::fillBarrelList..."<<endl;
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  string currDir = bei->pwd();
  if (currDir.find(dir_name) != string::npos)  {
    vector<MonitorElement*> mod_mes;

  //  vector<string> contents = mui->getMEs(); 
    vector<string> contents = bei->getMEs(); 
    
    for (vector<string>::const_iterator iv = me_names.begin();
	 iv != me_names.end(); iv++) {
      for (vector<string>::const_iterator im = contents.begin();
	   im != contents.end(); im++) {
        string sname = (*iv);
        string tname = sname.substr(8,(sname.find("_",8)-8)) + "_";
	if (((*im)).find(tname) == 0) {
	  string fullpathname = bei->pwd() + "/" + (*im); 
          getModuleME(bei, fullpathname);                       
	}
      }
    }
  } else {  
    vector<string> subdirs = bei->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      if((*it).find("PixelEndcap")!=string::npos) continue;
      bei->cd(*it);
      fillBarrelList(bei, dir_name, me_names);
      bei->goUp();
    }
  }
  //cout<<"...leaving SiPixelActionExecutor::fillBarrelList!"<<endl;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *  
 */
//void SiPixelInformationExtractor::fillEndcapList(MonitorUserInterface* mui,
void SiPixelInformationExtractor::fillEndcapList(DaqMonitorBEInterface* bei,
                                                 string dir_name,
						 vector<string>& me_names) {
  //cout<<"entering SiPixelInformationExtractor::fillEndcapList..."<<endl;
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  string currDir = bei->pwd();
  if (currDir.find(dir_name) != string::npos)  {
    vector<MonitorElement*> mod_mes;

  //  vector<string> contents = mui->getMEs(); 
    vector<string> contents = bei->getMEs(); 
    
    for (vector<string>::const_iterator iv = me_names.begin();
	 iv != me_names.end(); iv++) {
      for (vector<string>::const_iterator im = contents.begin();
	   im != contents.end(); im++) {
        string sname = (*iv);
        string tname = sname.substr(8,(sname.find("_",8)-8)) + "_";
	if (((*im)).find(tname) == 0) {
	  string fullpathname = bei->pwd() + "/" + (*im); 
          getModuleME(bei, fullpathname);                        
	}
      }
    }
  } else {  
    vector<string> subdirs = bei->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      if((bei->pwd()).find("PixelBarrel")!=string::npos) bei->goUp();
      bei->cd((*it));
      if((*it).find("PixelBarrel")!=string::npos) continue;
      fillEndcapList(bei, dir_name, me_names);
      bei->goUp();
    }
  }
  //cout<<"...leaving SiPixelActionExecutor::fillEndcapList!"<<endl;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *  
 *  Returns a pointer to a ME filtered by me_name from the list of ME in the current directory
 *  In doing so it clears its content (not sure why...)
 */
//MonitorElement* SiPixelInformationExtractor::getModuleME(MonitorUserInterface* mui,
MonitorElement* SiPixelInformationExtractor::getModuleME(DaqMonitorBEInterface* bei,
                                                         string me_name) {
//cout<<"Entering SiPixelInformationExtractor::getModuleME..."<<endl;
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  MonitorElement* me = 0;
  // If already booked

  // non-backward compatible MUI<->BEI change:
  //vector<string> contents = mui->getMEs();    
  vector<string> contents = bei->getMEs();   
   
  for (vector<string>::const_iterator it = contents.begin();
       it != contents.end(); it++) {
    if ((*it).find(me_name) == 0) {
      string fullpathname = bei->pwd() + "/" + (*it); 

  // non-backward compatible MUI<->BEI change:
  //    me = mui->get(fullpathname);
      me = bei->get(fullpathname);
      
      if (me) {
	MonitorElementT<TNamed>* obh1 = dynamic_cast<MonitorElementT<TNamed>*> (me);
	if (obh1) {
	  TH1F * root_obh1 = dynamic_cast<TH1F *> (obh1->operator->());
	  if (root_obh1) root_obh1->Reset();        
	}
	return me;
      }
    }
  }
  
  cout  << ACRed << ACBold << ACReverse
        << "[SiPixelInformationExtractor::getModuleME()]"
	<< ACPlain << ACYellow << ACBold 
	<< " Potential bug: "
	<< ACPlain
	<< "No module found for "
	<< me_name
	<<endl;
  return NULL;
  //cout<<"...leaving SiPixelInformationExtractor::getModuleME!"<<endl;
}

//------------------------------------------------------------------------------
/*! \brief Monitor elements extractor. 
 *
 *  This method returns a vector of pointers to MonitorElements (mes) satisfying an 
 *  input filter (names + mid).
 *  The 'mid'   selector is usually the DetId 
 *  The 'names' selector is the list of ME names obtained by parsing the appropriate
 *  xml configuration file (see sipixel_monitorelement_config.xml)
 *  The method is specialized to siPixel Monitor Elements only
 *  
 */
//void SiPixelInformationExtractor::selectSingleModuleHistos(MonitorUserInterface    * mui,  
void SiPixelInformationExtractor::selectSingleModuleHistos(DaqMonitorBEInterface   * bei,  
                                                           string                    mid,  
							   vector<string>          & names,
							   vector<MonitorElement*> & mes) 
{  
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  string currDir = bei->pwd();
  //QRegExp rx("(\\w+)_siPixel") ;
  //QRegExp rx2("(\\w+)_ctfWithMaterialTracks") ;
  QRegExp rx("(\\w+)_3") ;
  QString theME ;
  if (currDir.find("Module_") != string::npos ||
      currDir.find("FED_") != string::npos)  
  {
  //  vector<string> contents = mui->getMEs();    
    vector<string> contents = bei->getMEs(); 
       
    for (vector<string>::const_iterator it = contents.begin(); it != contents.end(); it++) 
    {
//       cout << ACRed << ACReverse
//            << "[SiPixelInformationExtractor::selectSingleModuleHistos()]"
// 	   << ACPlain
// 	   << " Got: "
// 	   <<  *it
// 	   << endl ;
      if((*it).find(mid) != string::npos)
      {
        for (vector<string>::const_iterator ih = names.begin(); ih != names.end(); ih++) 
	{
	  theME = *it ;
          string temp_s ; 
          //if( rx1.search(theME) != -1 ) { temp_s = rx1.cap(1).latin1() ; }
          //else if( rx2.search(theME) != -1 ) { temp_s = rx2.cap(1).latin1() ; }
          if( rx.search(theME) != -1 ) { temp_s = rx.cap(1).latin1() ; }
	  if (temp_s == (*ih)) 
	  {
	    string full_path = currDir + "/" + (*it);
	    //cout<<"full_path="<<full_path<<endl;

  //	    MonitorElement * me = mui->get(full_path.c_str());
	    MonitorElement * me = bei->get(full_path.c_str());
	    
	    if (me) 
	    {
	     mes.push_back(me);
	    }
	  }
        }
      }
    }
    if (mes.size() >0) 
    {
     return;
    }
  } else {  
    vector<string> subdirs = bei->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin(); it != subdirs.end(); it++) 
    {
      bei->cd(*it);
      selectSingleModuleHistos(bei, mid, names, mes);
      bei->goUp();
    }
  }
//cout<<"leaving SiPixelInformationExtractor::selectSingleModuleHistos"<<endl;
}
//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
//void SiPixelInformationExtractor::plotSingleModuleHistos(MonitorUserInterface* mui, 
void SiPixelInformationExtractor::plotSingleModuleHistos(DaqMonitorBEInterface* bei, 
                                                         multimap<string, string>& req_map) {
//cout<<"entering SiPixelInformationExtractor::plotSingleModuleHistos"<<endl;
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  vector<string> item_list;  

  string mod_id = getItemValue(req_map,"ModId");
  if (mod_id.size() < 9) {
    setCanvasMessage("Wrong Module Id!!");
    fillImageBuffer();
    canvas_->Clear(); 
    return;
  }
  item_list.clear();     
  getItemList(req_map,"histo", item_list); // item_list holds all histos to plot
  vector<MonitorElement*> me_list;

  bei->cd();
  selectSingleModuleHistos(bei, mod_id, item_list, me_list);
  bei->cd();

//  plotHistos(req_map,me_list);
  if (me_list.size() == 0) {
    setCanvasMessage("Wrong Module Id!!");  
  } else {
    plotHistos(req_map,me_list);
  }
  fillImageBuffer();
  canvas_->Clear();
//cout<<"leaving SiPixelInformationExtractor::plotSingleModuleHistos"<<endl;
}
//============================================================================================================
// --  Plot a Selected Monitor Element
// 
//void SiPixelInformationExtractor::plotTkMapHisto(MonitorUserInterface * mui, 
void SiPixelInformationExtractor::plotTkMapHisto(DaqMonitorBEInterface * bei, 
                                                 string                  theModId, 
						 string                  theMEName) 
{
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  vector<MonitorElement*> me_list;
  vector<string>	  theMENameList;
  theMENameList.push_back(theMEName) ;
    
  bei->cd();
  selectSingleModuleHistos(bei, theModId, theMENameList, me_list);
  bei->cd();

  if( me_list.size() < 1 )
  {
   cout << ACYellow << ACBold << ACReverse
	<< "[SiPixelInformationExtractor::plotTkMapHisto()]"
	<< ACCyan << ACBold 
	<< " Size of me_list is zero!"
	<< ACPlain 
	<< endl ;
  }

//  cout << ACYellow << ACBold 
//       << "[SiPixelInformationExtractor::plotTkMapHisto()] "
//       << ACPlain 
//       << "Number of MEs to plot for "
//       << theMEName
//       << " ("
//       << theModId
//       << "): "
//       << me_list.size() 
//       << endl ;
  for( vector<MonitorElement*>::iterator it=me_list.begin(); it!=me_list.end(); it++)
  {
//   cout << ACYellow << ACBold 
//	<< "[SiPixelInformationExtractor::plotTkMapHisto()] "
//	<< ACPlain 
//	<< "Going to plot "
//	<< theMEName 
//	<< " --> "
//	<< (*it)->getName() 
//	<< endl ;
   plotHisto(*it, theMEName,"800","800") ;
  }
    
}
//============================================================================================================
// --  Return type of ME
//
std::string  SiPixelInformationExtractor::getMEType(MonitorElement * theMe)
{
  MonitorElementT<TNamed>* histogramObj = dynamic_cast<MonitorElementT<TNamed>*>(theMe);
  if(histogramObj) 
  {
    QString qtype = histogramObj->operator->()->IsA()->GetName() ;
    if(         qtype.contains("TH1") > 0 )
    {
     return "TH1" ;
    } else if ( qtype.contains("TH2") > 0  ) {
     return "TH2" ;
    } else if ( qtype.contains("TH3") > 0 ) {
     return "TH3" ;
    }
    
  } else {
   cout << ACYellow << ACBold 
   	<< "[SiPixelInformationExtractor::getMEType()] "
   	<< ACRed << ACBold << ACReverse
   	<< "WARNING:"
   	<< ACPlain 
	<< " Could not dynamic_cast "
	<< ACCyan
   	<< theMe->getName()
	<< " to TNamed"
	<< ACPlain
   	<< endl ;
  }
  return "TH1" ;
}

//============================================================================================================
// --  Plot Selected Monitor Elements
// 
void SiPixelInformationExtractor::plotHisto(MonitorElement * theMe, 
                                            std::string      theMEName,
					    std::string      canvasW,
					    std::string      canvasH) 
{
  QString meName ;
//   cout << ACYellow << ACBold << ACReverse
//        << "[SiPixelInformationExtractor::plotHisto()]"
//        << ACCyan << ACBold 
//        << " Plotting "
//        << ACPlain 
//        << theMEName
//        << " res: "
//        << canvasW
//        << "x"
//        << canvasH
//        << endl ;
  QString cW = canvasW ;
  QString cH = canvasH ;
  TCanvas * theCanvas = new TCanvas("TrackerMapPlotsCanvas", 
                                    "TrackerMapPlotsCanvas",
				    cW.toInt(),
				    cH.toInt());
  gROOT->Reset(); 
  gStyle->SetPalette(1,0);

  MonitorElementT<TNamed>* histogramObj = dynamic_cast<MonitorElementT<TNamed>*>(theMe);
  if(histogramObj) 
  {
    string opt = "" ;
    QString type = histogramObj->operator->()->IsA()->GetName() ;
    if(         type.contains("TH1") > 0 )
    {
     opt = "" ;
    } else if ( type.contains("TH2") > 0  ) {
     opt = "COLZ" ;
    } else if ( type.contains("TH3") > 0 ) {
     opt = "" ;
    }
    histogramObj->operator->()->Draw(opt.c_str());
  } else {
   cout << ACYellow << ACBold 
   	<< "[SiPixelInformationExtractor::plotHisto()] "
   	<< ACRed << ACBold << ACReverse
   	<< "WARNING:"
   	<< ACPlain 
	<< " Could not dynamic_cast "
	<< ACCyan
   	<< theMEName
	<< " to TNamed"
	<< ACPlain
   	<< endl ;
  }
  theCanvas->Update();
  fillNamedImageBuffer(theCanvas,theMEName);
//   cout << ACYellow << ACBold << ACReverse
//        << "[SiPixelInformationExtractor::plotHisto()]"
//        << ACPlain 
//        << " Done"
//        << endl ;
   delete theCanvas ;
}
//============================================================================================================
// --  Plot Selected Monitor Elements
// 
//void SiPixelInformationExtractor::plotTkMapHistos(MonitorUserInterface     * mui, 
void SiPixelInformationExtractor::plotTkMapHistos(DaqMonitorBEInterface    * bei, 
                                                  multimap<string, string> & req_map, 
						  string                     sname) 
{
//cout<<"entering SiPixelInformationExtractor::plotSingleModuleHistos"<<endl;
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  string mod_id = getItemValue(req_map,"ModId");
//   cout << ACYellow << ACBold << ACReverse
//        << "[SiPixelInformationExtractor::plotTkMapHistos()]"
//        << ACPlain << " Registering call for "
//        << sname
//        << "(" << mod_id << ")"
//        << endl ;

  vector<string> item_list;  

  //cout<<"mod_id in plotSingleModuleHistos:"<<mod_id<<endl;
  if (mod_id.size() < 9) return;
  item_list.clear();     
  getItemList(req_map,"histo", item_list); // item_list holds all histos to plot
  vector<MonitorElement*> me_list;

  bei->cd();
  selectSingleModuleHistos(bei, mod_id, item_list, me_list);
  bei->cd();

  QRegExp rx(sname) ;
  QString meName ;

  bool histoFound = false ;
 
  for( vector<MonitorElement*>::iterator it=me_list.begin(); it!=me_list.end(); it++)
  {
   meName = (*it)->getName() ;
   if( rx.search(meName) == -1 ) {continue;}
//    cout << ACYellow << ACBold << ACReverse
//         << "[SiPixelInformationExtractor::plotTkMapHistos()]"
// 	<< ACCyan << ACBold 
// 	<< " Fetching "
// 	<< ACPlain 
// 	<< meName
// 	<< endl ;
   vector<MonitorElement*> one_me ;
   one_me.push_back(*it) ;
   plotHistos(req_map,one_me);
   histoFound = true ;
  }
  if( !histoFound ) 
  {
   cout << ACYellow << ACBold << ACReverse
        << "[SiPixelInformationExtractor::plotTkMapHistos()]"
	<< ACRed << ACBold << " Requested ME not found: "
	<< ACPlain << " "
	<< endl ;
  }
//cout<<"leaving SiPixelInformationExtractor::plotSingleModuleHistos"<<endl;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
//void SiPixelInformationExtractor::plotSingleHistogram(MonitorUserInterface * mui,
void SiPixelInformationExtractor::plotSingleHistogram(DaqMonitorBEInterface * bei,
		                                      std::multimap<std::string, 
						      std::string>& req_map){
//cout<<"entering SiPixelInformationExtractor::plotSingleHistogram"<<endl;
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  vector<string> item_list;  

  string path_name = getItemValue(req_map,"Path");
  if (path_name.size() == 0) return;
  
  //MonitorElement* me = mui->get(path_name);
  MonitorElement* me = bei->get(path_name);
  
  vector<MonitorElement*> me_list;
  if (me) {
    me_list.push_back(me);
    plotHistos(req_map,me_list);
  }
//cout<<"leaving SiPixelInformationExtractor::plotSingleHistogram"<<endl;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
void SiPixelInformationExtractor::plotHistos(multimap<string,string>& req_map, 
  			                     vector<MonitorElement*> me_list){
//cout<<"entering SiPixelInformationExtractor::plotHistos"<<endl;
  int nhist = me_list.size();
  if (nhist == 0) return;
  int width = 600;
  int height = 600;
  
/////  TCanvas canvas("TestCanvas", "Test Canvas");
/////  canvas.Clear();
  
  canvas_->Clear();
  gROOT->Reset(); gStyle->SetPalette(1);
  int ncol=1, nrow=1;
 
  float xlow = -1.0;
  float xhigh = -1.0;
  
  if (nhist == 1) {
    if (hasItem(req_map,"xmin")) xlow = atof(getItemValue(req_map,"xmin").c_str());
    if (hasItem(req_map,"xmax")) xhigh = atof(getItemValue(req_map,"xmax").c_str()); 
    ncol = 1;
    nrow = 1;
  } else {
    if (hasItem(req_map,"cols")) ncol = atoi(getItemValue(req_map, "cols").c_str());
    if (hasItem(req_map,"rows")) nrow = atoi(getItemValue(req_map, "rows").c_str());
    if (ncol*nrow < nhist) {
      if (nhist == 2) {
	ncol = 1;
	nrow = 2;
      } else if (nhist == 3) {
	ncol = 1;
	nrow = 3;
      } else if (nhist == 4) {
	ncol = 2;
	nrow = 3;
      } else if (nhist > 4 && nhist <= 10) {
        ncol = 2;
	nrow = nhist/ncol+1;
      } else if (nhist > 10 && nhist <= 20) {
        ncol = 3;
	nrow = nhist/ncol+1;
      } else if (nhist > 20 && nhist <= 40) {
         ncol = 4;
	 nrow = nhist/ncol+1;
      } 		

    }
  }

  if (hasItem(req_map,"width")) 
              width = atoi(getItemValue(req_map, "width").c_str());    
  if (hasItem(req_map,"height"))
              height = atoi(getItemValue(req_map, "height").c_str());

/////  canvas.SetWindowSize(width,height);
/////  canvas.Divide(ncol, nrow);

  canvas_->SetWindowSize(width,height);
  canvas_->Divide(ncol, nrow);
  int i=0;
  for (vector<MonitorElement*>::const_iterator it = me_list.begin();
       it != me_list.end(); it++) {
    i++;
    int istat =  SiPixelUtility::getStatus((*it));
    string tag;
    int icol;
    SiPixelUtility::getStatusColor(istat, icol, tag);
  
    MonitorElementT<TNamed>* ob = 
      dynamic_cast<MonitorElementT<TNamed>*>((*it));
    if (ob) {
/////      canvas.cd(i);
      canvas_->cd(i);
      //      TAxis* xa = ob->operator->()->GetXaxis();
      //      xa->SetRangeUser(xlow, xhigh);
      if(hasItem(req_map,"colpal")){
        //cout<<"IE::plotHistos found colpal!"<<endl;
        gROOT->Reset(); gStyle->SetPalette(1); gStyle->SetOptStat(0);
        ob->operator->()->Draw("colz");
      }else{
	string hname = ob->operator->()->GetName();
	//cout<<"histo name:"<<hname<<endl;
	if(hname.find("hitmap") != string::npos){
          gROOT->Reset(); gStyle->SetPalette(1); gStyle->SetOptStat(0);
          ob->operator->()->Draw("colz");
	}else{  
          gStyle->SetOptStat(1);
          ob->operator->()->Draw();
	}
      }
      if (icol != 1) {
	TText tt;
	tt.SetTextSize(0.12);
	tt.SetTextColor(icol);
	tt.DrawTextNDC(0.5, 0.5, tag.c_str());
      }
      if (hasItem(req_map,"logy")) {
	  gPad->SetLogy(1);
      }
    } else setCanvasMessage("Plot does not exist (yet)!!!");
  }
  gStyle->SetPalette(1);
  canvas_->Update();
  fillImageBuffer();
  canvas_->Modified();
//cout<<"leaving SiPixelInformationExtractor::plotHistos"<<endl;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
//void SiPixelInformationExtractor::readModuleAndHistoList(MonitorUserInterface* mui, 
void SiPixelInformationExtractor::readModuleAndHistoList(DaqMonitorBEInterface* bei, 
                                                         xgi::Output * out, 
							 bool coll_flag) {
//cout<<"entering SiPixelInformationExtractor::readModuleAndHistoList"<<endl;
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
   std::map<std::string,std::string> hnames;
   std::vector<std::string> mod_names;
   if (coll_flag)  bei->cd("Collector/Collated");
   fillModuleAndHistoList(bei, mod_names, hnames);
   //for (std::vector<std::string>::iterator im = mod_names.begin();
   //     im != mod_names.end(); im++) cout<<"mod_names="<<*im<<endl;
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
   if (coll_flag)  bei->cd();
//cout<<"leaving SiPixelInformationExtractor::readModuleAndHistoList"<<endl;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
//void SiPixelInformationExtractor::fillModuleAndHistoList(MonitorUserInterface * mui, 
void SiPixelInformationExtractor::fillModuleAndHistoList(DaqMonitorBEInterface * bei, 
                                                         vector<string>        & modules,
							 map<string,string>    & histos) {
//cout<<"entering SiPixelInformationExtractor::fillModuleAndHistoList"<<endl;
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  string currDir = bei->pwd();
  if (currDir.find("Module_") != string::npos ||
      currDir.find("FED_") != string::npos)  {
    if (histos.size() == 0) {
      //cout<<"currDir="<<currDir<<endl;

  //    vector<string> contents = mui->getMEs();    
      vector<string> contents = bei->getMEs();
          
      for (vector<string>::const_iterator it = contents.begin();
	   it != contents.end(); it++) {
	string hname          = (*it).substr(0, (*it).find("_siPixel"));
	if (hname==" ") hname = (*it).substr(0, (*it).find("_ctfWithMaterialTracks"));
        string fullpathname   = bei->pwd() + "/" + (*it); 

  //      MonitorElement * me   = mui->get(fullpathname);
        MonitorElement * me   = bei->get(fullpathname);
	
        string htype          = "undefined" ;
        if (me) 
	{
         MonitorElementT<TNamed>* histogramObj = dynamic_cast<MonitorElementT<TNamed>*> (me);
         if(histogramObj) 
         {
          htype = histogramObj->operator->()->IsA()->GetName() ;
	 }
	}
	//cout<<"hname="<<hname<<endl;
        histos[hname] = htype ;
        string mId=" ";
	if(hname.find("ndigis")                !=string::npos) mId = (*it).substr((*it).find("ndigis_siPixelDigis_")+20, 9);
	if(mId==" " && hname.find("nclusters") !=string::npos) mId = (*it).substr((*it).find("nclusters_siPixelClusters_")+26, 9);
        if(mId==" " && hname.find("residualX") !=string::npos) mId = (*it).substr((*it).find("residualX_ctfWithMaterialTracks_")+32, 9);
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
//void SiPixelInformationExtractor::readModuleHistoTree(MonitorUserInterface* mui, 
void SiPixelInformationExtractor::readModuleHistoTree(DaqMonitorBEInterface* bei, 
                                                      string& str_name, 
						      xgi::Output * out, 
						      bool coll_flag) {
//cout<<"entering  SiPixelInformationExtractor::readModuleHistoTree"<<endl;
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  ostringstream modtree;
  if (goToDir(bei, str_name, coll_flag)) {
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
//void SiPixelInformationExtractor::printModuleHistoList(MonitorUserInterface * mui, 
void SiPixelInformationExtractor::printModuleHistoList(DaqMonitorBEInterface * bei, 
                                                       ostringstream& str_val){
//cout<<"entering SiPixelInformationExtractor::printModuleHistoList"<<endl;
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  static string indent_str = "";
  string currDir = bei->pwd();
  string dname = currDir.substr(currDir.find_last_of("/")+1);
  str_val << " <li>\n"
	  << "  <a href=\"#\" id=\"" << currDir << "\">\n   " 
	  <<     dname << "\n"
	  << "  </a>\n"
	  << endl << endl;

  //vector<string> meVec     = mui->getMEs(); 
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
      QRegExp rx("(\\w+)_(siPixel|ctfWithMaterialTracks|_3)") ;
      if( rx.search(qit) > -1 ) {qit = rx.cap(1);} 
      str_val << "    <li class=\"dhtmlgoodies_sheet.gif\">\n"
	      << "     <input id      = \"selectedME\""
	      << "            folder  = \"" << currDir << "\""
	      << "            type    = \"checkbox\""
	      << "            name    = \"selected\""
	      << "            class   = \"smallCheckBox\""
	      << "            value   = \"" << (*it) << "\""
	      << "            onclick = \"javascript:IMGC.selectedIMGCItems()\" />\n"
	      << "     <a href=\"javascript:IMGC.updateIMGC('" << currDir << "')\">\n       " 
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
//void SiPixelInformationExtractor::readSummaryHistoTree(MonitorUserInterface* mui, 
void SiPixelInformationExtractor::readSummaryHistoTree(DaqMonitorBEInterface* bei, 
                                                       string& str_name, 
						       xgi::Output * out, 
						       bool coll_flag) {
//cout<<"entering  SiPixelInformationExtractor::readSummaryHistoTree"<<endl;
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  ostringstream sumtree;
  if (goToDir(bei, str_name, coll_flag)) {
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
//void SiPixelInformationExtractor::printSummaryHistoList(MonitorUserInterface * mui, 
void SiPixelInformationExtractor::printSummaryHistoList(DaqMonitorBEInterface * bei, 
                                                        ostringstream& str_val){
//cout<<"entering SiPixelInformationExtractor::printSummaryHistoList"<<endl;
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  static string indent_str = "";
  string currDir = bei->pwd();
  string dname = currDir.substr(currDir.find_last_of("/")+1);
  if (dname.find("Module_") ==0 || dname.find("FED_")==0) return;
  str_val << " <li>\n"
          << "  <a href=\"#\" id=\"" << currDir << "\">\n   " 
	  <<     dname 
	  << "  </a>" 
	  << endl;

  //vector<string> meVec     = mui->getMEs(); 
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
      //QRegExp rx1("(\\w+)_siPixel") ;
      //QRegExp rx2("(\\w+)_ctfWithMaterialTracks") ;
      QRegExp rx("(\\w+)_3") ;
      //if( rx1.search(qit) > -1 ) {qit = rx1.cap(1);} 
      //else if( rx2.search(qit) > -1 ) {qit = rx2.cap(1);} 
      if( rx.search(qit) > -1 ) {qit = rx.cap(1);} 
      str_val << "    <li class=\"dhtmlgoodies_sheet.gif\">\n"
	      << "     <input id      = \"selectedME\""
	      << "            folder  = \"" << currDir << "\""
	      << "            type    = \"checkbox\""
	      << "            name    = \"selected\""
	      << "            class   = \"smallCheckBox\""
	      << "            value   = \"" << (*it) << "\""
	      << "            onclick = \"javascript:IMGC.selectedIMGCItems()\" />\n"
              << "     <a href=\"javascript:IMGC.updateIMGC('" << currDir << "')\">\n       " 
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
//void SiPixelInformationExtractor::readAlarmTree(MonitorUserInterface* mui, 
void SiPixelInformationExtractor::readAlarmTree(DaqMonitorBEInterface* bei, 
                                                string& str_name, 
						xgi::Output * out, 
						bool coll_flag){
//cout<<"entering SiPixelInformationExtractor::readAlarmTree"<<endl;
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  ostringstream alarmtree;
  if (goToDir(bei, str_name, coll_flag)) {
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
//void SiPixelInformationExtractor::printAlarmList(MonitorUserInterface * mui, 
void SiPixelInformationExtractor::printAlarmList(DaqMonitorBEInterface * bei, 
                                                 ostringstream& str_val){
//cout<<"entering SiPixelInformationExtractor::printAlarmList"<<endl;
//   cout << ACRed << ACBold
//        << "[SiPixelInformationExtractor::printAlarmList()]"
//        << ACPlain
//        << " Enter" 
//        << endl ;
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
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

  //vector<string> meVec = mui->getMEs(); 
  vector<string> meVec = bei->getMEs();
   
  if (subDirVec.size() == 0 && meVec.size() == 0) {
    str_val <<  "</li> "<< endl;    
    return;
  }
  str_val << "<ul>" << endl;
  for (vector<string>::const_iterator it = meVec.begin();
	   it != meVec.end(); it++) {
    string full_path = currDir + "/" + (*it);

  //  MonitorElement * me = mui->get(full_path);
    MonitorElement * me = bei->get(full_path);
    
    if (!me) continue;
    dqm::qtests::QR_map my_map = me->getQReports();
    if (my_map.size() > 0) {
      string image_name1;
      selectImage(image_name1,my_map);
      if(image_name1!="images/LI_green.gif") {
        alarmCounter_++;
        QString qit = (*it) ;
        //QRegExp rx1("(\\w+)_siPixel") ;
        //QRegExp rx2("(\\w+)_ctfWithMaterialTracks") ;
        QRegExp rx("(\\w+)_3") ;
        //if( rx1.search(qit) > -1 ) {qit = rx1.cap(1);} 
        //else if( rx2.search(qit) > -1 ) {qit = rx2.cap(1);} 
        if( rx.search(qit) > -1 ) {qit = rx.cap(1);} 
//        str_val << "<li class=\"dhtmlgoodies_sheet.gif\"><a href=\"javascript:RequestPlot.ReadStatus('"
//		<< full_path<< "')\">" << (*it) << "</a><img src=\""
//		<< image_name1 << "\""<< "</li>" << endl;
        str_val << "	<li class=\"dhtmlgoodies_sheet.gif\">\n"
        	<< "	 <input id	= \"selectedME\""
        	<< "		folder  = \"" << currDir << "\""
        	<< "		type	= \"checkbox\""
        	<< "		name	= \"selected\""
        	<< "		class	= \"smallCheckBox\""
        	<< "		value	= \"" << (*it) << "\""
        	<< "		onclick = \"javascript:IMGC.selectedIMGCItems()\" />\n"
        	<< "	 <a href=\"javascript:IMGC.updateIMGC('" << currDir << "')\">\n       " 
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
void SiPixelInformationExtractor::getItemList(multimap<string, string>& req_map, 
                                              string item_name,
					      vector<string>& items) {
//cout<<"entering SiPixelInformationExtractor::getItemList"<<endl;
  items.clear();
  for (multimap<string, string>::const_iterator it = req_map.begin();
       it != req_map.end(); it++) {
    //cout<<"....item_name="<<item_name<<endl;
    //cout<<"....first="<<it->first<<" ....second="<<it->second<<endl;
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
string SiPixelInformationExtractor::getItemValue(multimap<string,string>& req_map,
						 std::string item_name){
//cout<<"entering SiPixelInformationExtractor::getItemValue"<<endl;
  multimap<string,string>::iterator pos = req_map.find(item_name);
  string value = " ";
  if (pos != req_map.end()) {
    value = pos->second;
  }
  return value;
//cout<<"leaving SiPixelInformationExtractor::getItemValue"<<endl;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
void SiPixelInformationExtractor::fillNamedImageBuffer(TCanvas * c1, 
                                                       std::string theName) 
{
  // Now extract the image
  // 114 - stands for "no write on Close"
//   cout << ACYellow << ACBold
//        << "[SiPixelInformationExtractor::fillNamedImageBuffer()] "
//        << ACPlain
//        << "A canvas: "
//        << c1->GetName() 
//        << endl ;
  TImageDump imgdump("tmp.png", 114);
  c1->Paint();

// get an internal image which will be automatically deleted
// in the imgdump destructor
  TImage *image = imgdump.GetImage();
  if( image == NULL )
  {
   cout << ACYellow << ACBold
   	<< "[SiPixelInformationExtractor::fillNamedImageBuffer()] "
   	<< ACRed << ACBold
   	<< "WARNING: " 
	<< ACPlain
	<< "No TImage found for "
	<< theName
   	<< endl ;
    return ;
  }
  char *buf;
  int sz = 0;
  image->GetImageBuffer(&buf, &sz);

  pictureBuffer_.str("");
  for (int i = 0; i < sz; i++) pictureBuffer_ << buf[i];
  
//  delete [] buf;
  ::free(buf); // buf is allocated via realloc() by a C language AfterStep library invoked by the
               // default (and so far only) TImage implementation in root, TASImage.
  
  namedPictureBuffer[theName] = pictureBuffer_.str() ;
//  cout << ACCyan << ACBold << ACReverse 
//       << "[SiPixelInformationExtractor::fillNamedImageBuffer()]"
//       << ACPlain
//       << " Storing away " << theName
//       << " size now is: " << namedPictureBuffer.size() 
//       << endl ;
//  for( map<std::string, std::string>::iterator buf =namedPictureBuffer.begin();
//  					       buf!=namedPictureBuffer.end(); buf++)
//  {
//   cout << ACCyan << ACBold << ACReverse 
//  	<< "[SiPixelInformationExtractor::fillNamedImageBuffer()]"
//  	<< ACPlain
//        << " ME: "
//        << buf->first
//        << endl ;
//  }
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
void SiPixelInformationExtractor::fillImageBuffer(TCanvas& c1) {
//cout<<"entering SiPixelInformationExtractor::fillImageBuffer"<<endl;
  c1.SetFixedAspectRatio(kTRUE);
  c1.SetCanvasSize(800, 600);
  gStyle->SetPalette(1);
  // Now extract the image
  // 114 - stands for "no write on Close"
  TImageDump imgdump("tmp.png", 114);
  c1.Paint();

 // get an internal image which will be automatically deleted
 // in the imgdump destructor
  TImage *image = imgdump.GetImage();

  char *buf;
  int sz;
  image->GetImageBuffer(&buf, &sz);         /* raw buffer */
  pictureBuffer_.str("");
  for (int i = 0; i < sz; i++)
    pictureBuffer_ << buf[i];
  
  delete [] buf;
//cout<<"leaving SiPixelInformationExtractor::fillImageBuffer"<<endl;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
const ostringstream&  SiPixelInformationExtractor::getImage() const {
//cout<<"entering SiPixelInformationExtractor::getImage"<<endl;
  return pictureBuffer_;
//cout<<"leaving SiPixelInformationExtractor::getImage"<<endl;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
//const ostringstream&  SiPixelInformationExtractor::getIMGCImage(MonitorUserInterface* mui, 
const ostringstream&  SiPixelInformationExtractor::getIMGCImage(DaqMonitorBEInterface* bei, 
                                                                std::string theFullPath,
								std::string canvasW,
								std::string canvasH) 
{
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  // MonitorElement * theME = mui->get(theFullPath) ;
   MonitorElement * theME = bei->get(theFullPath) ;
   
   if( !theME ) 
   {
     cout << ACRed << ACBold
          << "[SiPixelInformationExtractor::getIMGCImage()] " 
	  << ACPlain
	  << "FATAL: no ME found for full path "
	  << theFullPath
	  << endl ;
   }
   plotHisto(theME, theFullPath, canvasW, canvasH) ;
   return getNamedImage(theFullPath) ;   
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
const ostringstream&  SiPixelInformationExtractor::getNamedImage(std::string theName) 
{
//   cout << ACCyan << ACBold << ACReverse 
//        << "[SiPixelInformationExtractor::getNamedImage()]"
//        << ACPlain
//        << " Requested " << theName
//        << endl ;
  pictureBuffer_.str("") ;
  map<std::string, std::string>::iterator thisBuffer = namedPictureBuffer.find(theName) ; 
  if( thisBuffer == namedPictureBuffer.end() )
  {
    cout << ACCyan << ACBold << ACReverse 
    	 << "[SiPixelInformationExtractor::getNamedImage()]"
    	 << ACRed << ACBold << ACReverse
    	 << " WARNING: " 
	 << ACPlain
	 << "ME image buffer for "
	 << theName
	 << " not found among "
	 << namedPictureBuffer.size()
	 << " MEs"
    	 << endl ;
    for( map<std::string, std::string>::iterator buf =namedPictureBuffer.begin();
                                                 buf!=namedPictureBuffer.end(); buf++)
    {
     cout << ACCyan << ACBold << ACReverse 
    	  << "[SiPixelInformationExtractor::getNamedImage()]"
          << ACPlain
	  << " ME: "
	  << buf->first
	  << endl ;
    }
    cout << ACCyan << ACBold << ACReverse 
         << "[SiPixelInformationExtractor::getNamedImage()] "
    	 << ACPlain << endl ;
    return pictureBuffer_;
  } else {
//    cout << ACCyan << ACBold << ACReverse 
//         << "[SiPixelInformationExtractor::getNamedImage()] "
//    	 << ACPlain 
//	 << "Buffer containing picture found for "
//	 << theName
//	 << endl ;
  }
  pictureBuffer_ << thisBuffer->second ;
//   cout << ACCyan << ACBold << ACReverse 
//        << "[SiPixelInformationExtractor::getNamedImage()]"
//        << ACPlain
//        << " Returning " << theName
//        << endl ;
  return pictureBuffer_;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
//bool SiPixelInformationExtractor::goToDir(MonitorUserInterface* mui, 
bool SiPixelInformationExtractor::goToDir(DaqMonitorBEInterface* bei, 
                                          string& sname, 
					  bool flg){ 
//cout<<"entering SiPixelInformationExtractor::goToDir"<<endl;
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  bei->cd();
  bei->cd("Collector");
  //cout << mui->pwd() << endl;
  vector<string> subdirs;
  subdirs = bei->getSubdirs();
  if (subdirs.size() == 0) return false;
  
  if (flg) bei->cd("Collated");
  else bei->cd(subdirs[0]);
  //cout << bei->pwd() << endl;
  subdirs.clear();
  subdirs = bei->getSubdirs();
  if (subdirs.size() == 0) return false;
  //cout<<"sname="<<sname<<endl;
  bei->cd(sname);
  string dirName = bei->pwd();
  if (dirName.find(sname) != string::npos) return true;
  else return false;  
//cout<<"leaving SiPixelInformationExtractor::goToDir"<<endl;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
void SiPixelInformationExtractor::selectImage(string& name, 
                                              int status){
//cout<<"entering SiPixelInformationExtractor::selectImage"<<endl;
  switch(status){
  case dqm::qstatus::STATUS_OK:
    name="images/LI_green.gif";
    break;
  case dqm::qstatus::WARNING:
    name="images/LI_yellow.gif";
    break;
  case dqm::qstatus::ERROR:
    name="images/LI_red.gif";
    break;
  case dqm::qstatus::INVALID:
    break;
  case dqm::qstatus::INSUF_STAT:
    name="images/LI_blue.gif";
    break;
  default:
    name="images/LI_orange.gif";
    break;
  }
//cout<<"leaving SiPixelInformationExtractor::selectImage"<<endl;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
void SiPixelInformationExtractor::selectImage(string& name, 
                                              dqm::qtests::QR_map& test_map){
//cout<<"entering SiPixelInformationExtractor::selectImage"<<endl;
  int istat = 999;
  int status = 0;
  for (dqm::qtests::QR_map::const_iterator it = test_map.begin(); it != test_map.end();
       it++) {
    status = it->second->getStatus();
    if (status > istat) istat = status;
  }
  selectImage(name, status);
//cout<<"leaving SiPixelInformationExtractor::selectImage"<<endl;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
//void SiPixelInformationExtractor::readStatusMessage(MonitorUserInterface* mui, 
void SiPixelInformationExtractor::readStatusMessage(DaqMonitorBEInterface* bei, 
                                                    string& path,
						    xgi::Output * out) {
//cout<<"entering SiPixelInformationExtractor::readStatusMessage"<<endl;
  //DaqMonitorBEInterface * bei = mui->getBEInterface();

  //MonitorElement* me = mui->get(path);
  MonitorElement* me = bei->get(path);
  
  string hpath;
  ostringstream test_status; //this is the output stream displayed in browser!
  if (!me) {
    test_status << " ME Does not exist ! ";
    hpath = "NOME";
  } else {
    hpath = path.substr(0,path.find("."));
    string me_name=me->getName();
    float me_entries=me->getEntries();
    float me_mean=me->getMean(1);
    float me_meanError=me->getMeanError(1);
    float me_RMS=me->getRMS(1);
    float me_RMSError=me->getRMSError(1);
    dqm::qtests::QR_map test_map = me->getQReports();
    for (dqm::qtests::QR_map::const_iterator it = test_map.begin(); it != test_map.end();
	 it++) {
      string qt_name = it->first;
      test_status << " QTest Name: "<<qt_name<<" --->";;
      int qt_status = it->second->getStatus();
      test_status<<" Status:";
      switch(qt_status){
      case dqm::qstatus::WARNING:
        test_status<<" WARNING "<<endl;
	break;
      case dqm::qstatus::ERROR:
        test_status<<" ERROR "<<endl;
	break;
      case dqm::qstatus::DISABLED:
        test_status<<" DISABLED "<<endl;
	break;
      case dqm::qstatus::INVALID:
        test_status<<" INVALID "<<endl;
	break;
      case dqm::qstatus::INSUF_STAT:
        test_status<<" NOT ENOUGH STATISTICS "<<endl;
	break;
      default:
        test_status<<" Unknown (status="<<qt_status<<")"<<endl;
      }
      //if (status == dqm::qstatus::WARNING) test_status << " Warning : " << endl;
      //else if (status == dqm::qstatus::ERROR) test_status << " Error : " << endl;
      //else if (status == dqm::qstatus::STATUS_OK) test_status << " Ok : " << endl;
      //else if (status == dqm::qstatus::OTHER) test_status << " Other(" << status << ") : " << endl;
      test_status << "&lt;br/&gt;";
/*      string mess_str = it->second->getMessage();
      mess_str = mess_str.substr(mess_str.find(" Test")+5);
      //test_status << " QTest Name  : " << mess_str.substr(0, mess_str.find(")")+1) << endl;
      //test_status << "&lt;br/&gt;";
      test_status <<  " QTest Detail  : " << mess_str.substr(mess_str.find(")")+2) << endl;
      //if(mess_str.substr(0, mess_str.find(")")+1) == "Mean within allowed range?") 
      //  test_status << "Mean= "<<
      std::vector<dqm::me_util::Channel> badchs=it->second->getBadChannels();
      //cout<<"STATUS: "<<status<<" ***** MESSAGE: "<<mess_str<<" ***** BAD CHANNELS: "<<endl;
      //cout<<"STATUS: "<<status<<" ***** MESSAGE: "<<mess_str<<" ***** BAD CHANNELS: "<<it->second->getBadChannels()<<endl;
*/
      if(qt_status!=dqm::qstatus::STATUS_OK){
        string test_mess=it->second->getMessage();
	string mess_str=test_mess.substr(test_mess.find("("));
        test_status<<"&lt;br/&gt;";
        test_status <<  " QTest Detail  : " << mess_str.substr(mess_str.find(")")+2);
        test_status << "&lt;br/&gt;";
        test_status <<  " ME : mean = " << me_mean << " =/- " << me_meanError
                    << ", RMS= " << me_RMS << " =/- " << me_RMSError;
        test_status << "&lt;br/&gt;";
        if(qt_status == dqm::qstatus::INSUF_STAT) test_status <<  " entries = " << me_entries;
        test_status << "&lt;br/&gt;";
        /*      std::vector<dqm::me_util::Channel> badChannels=it->second->getBadChannels();																		
              if(!badChannels.empty())  																								
              test_status << " Channels that failed test " << ":\n";																					
              vector<dqm::me_util::Channel>::iterator badchsit = badChannels.begin();																			
              while(badchsit != badChannels.end())																							
              { 																											
        	test_status << " Channel ("																								
        	     << badchsit->getBinX() << ","																							
        	     << badchsit->getBinY() << ","																							
        	     << badchsit->getBinZ()																								
        	     << ") Contents: " << badchsit->getContents() << " +- "																				
        	     << badchsit->getRMS() << endl;																							
        																												
        	++badchsit;																										
              } 																											
        test_status << "&lt;br/&gt;";*/
      }
    }      
  }
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  *out << "<?xml version=\"1.0\" ?>" 			 << endl;
  *out << "<StatusAndPath>" 	     			 << endl;
  *out << "<StatusList>"    	     			 << endl;
  *out << "<Status>" << test_status.str() << "</Status>" << endl;      
  *out << "</StatusList>" 		   		 << endl;
  *out << "<PathList>"    		   		 << endl;
  *out << "<HPath>"  << hpath             << "</HPath>"  << endl;  
  *out << "</PathList>"      		   		 << endl;
  *out << "</StatusAndPath>" 		   		 << endl;
//cout<<"leaving SiPixelInformationExtractor::readStatusMessage"<<endl;
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
//void SiPixelInformationExtractor::selectMEList(MonitorUserInterface    * mui,  
void SiPixelInformationExtractor::selectMEList(DaqMonitorBEInterface   * bei,  
					       string	               & theMEName,
					       vector<MonitorElement*> & mes) 
{  
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  string currDir = bei->pwd();
   
  //QRegExp rx("(\\w+)_siPixel") ;
  //QRegExp rx2("(\\w+)_ctfWithMaterialTracks") ;
  QRegExp rx("(\\w+)_3") ;
  QString theME ;
   
  // Get ME from Collector/FU0/Tracker/PixelEndcap/HalfCylinder_pX/Disk_X/Blade_XX/Panel_XX/Module_XX
  if (currDir.find("Module_") != string::npos ||
      currDir.find("FED_") != string::npos)  
  {
  //  vector<string> contents = mui->getMEs();    
    vector<string> contents = bei->getMEs(); 
       
    for (vector<string>::const_iterator it = contents.begin(); it != contents.end(); it++) 
    {
      theME = *it ;
      //if( rx1.search(theME) == -1 && rx2.search(theME) == -1 ) {continue ;} // If the ME is not a siPixel or ctfWithMaterialTrack one, skip
      if( rx.search(theME) == -1 ) {continue ;} // If the ME is not a siPixel or ctfWithMaterialTrack one, skip
      //if (rx1.cap(1).latin1() == theMEName || 
      //    rx2.cap(1).latin1() == theMEName)      // Found the ME we were looking for
      if (rx.cap(1).latin1() == theMEName)  
      {
        string full_path = currDir + "/" + (*it);

  //      MonitorElement * me = mui->get(full_path.c_str());
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
//void SiPixelInformationExtractor::sendTkUpdatedStatus(MonitorUserInterface  * mui, 
void SiPixelInformationExtractor::sendTkUpdatedStatus(DaqMonitorBEInterface  * bei, 
                                                      xgi::Output            * out,
						      std::string            & theMEName,
						      std::string            & theTKType) 
{
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  int rval, gval, bval;
  vector<string>          colorMap ;
  vector<MonitorElement*> me_list;
  pair<double,double>     norm ;
  double sts ;
    
  bei->cd();
  selectMEList(bei, theMEName, me_list) ;
  bei->cd();

  string detId = "undefined";

  //QRegExp rx1("\\w+_siPixel\\w+_(\\d+)") ;
  //QRegExp rx2("\\w+_ctfWithMaterialTracks\\w+_(\\d+)") ;
  QRegExp rx("\\w+_\\w+_(\\d+)") ;

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
    //if( rx1.search(meName) != -1 || 
    //    rx2.search(meName) != -1 )
    if( rx.search(meName) != -1 ) 
    {
     //detId = rx1.cap(1).latin1() ;
     //if (detId=="undefined") detId = rx2.cap(1).latin1() ;
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
 //QRegExp rx1("siPixel(\\w+)_(\\d+)") ;
 //QRegExp rx2("ctfWithMaterialTracks(\\w+)_(\\d+)") ;
 QRegExp rx("(\\w+)_(\\w+)_(\\d+)") ;
 QString mEName = mE->getName() ;

 int detId = 0;
 
 //if( rx1.search(mEName) != -1 ||
 //    rx2.search(mEName) != -1 )
 if( rx.search(mEName) != -1 )
 {
  //detId = rx1.cap(2).toInt() ;
  //if (detId==0) detId = rx2.cap(2).toInt() ;
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
//void SiPixelInformationExtractor::getMEList(MonitorUserInterface     * mui,  
void SiPixelInformationExtractor::getMEList(DaqMonitorBEInterface    * bei,  
					    map<string, int>         & mEHash)
{
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  string currDir = bei->pwd();
   
//   cout << ACRed << ACBold
//        << "[SiPixelInformationExtractor::getMEList()]"
//        << ACPlain
//        << " Requesting ME list in " 
//        << currDir
//        << endl ;
       
//  QRegExp rx("(\\w+)_siPixel") ;
  //QRegExp rx2("(\\w+)_ctfWithMaterialTracks") ;
  QRegExp rx("(\\w+)_3") ;
  QString theME ;
   
  // Get ME from Collector/FU0/Tracker/PixelEndcap/HalfCylinder_pX/Disk_X/Blade_XX/Panel_XX/Module_XX
  if (currDir.find("Module_") != string::npos ||
      currDir.find("FED_") != string::npos)  
  {
  //  vector<string> contents = mui->getMEs();    
    vector<string> contents = bei->getMEs(); 
       
    for (vector<string>::const_iterator it = contents.begin(); it != contents.end(); it++) 
    {
      theME = *it ;
      //if( rx1.search(theME) == -1 && rx2.search(theME) == -1 ) {continue ;} // If the ME is not a siPixel or ctfWithMaterialTracks one, skip
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
      } // If the ME is not a siPixel one, skip
      string full_path = currDir + "/" + (*it);
      //string mEName = rx1.cap(1).latin1() ;
      //if(mEName==" ") mEName = rx2.cap(1).latin1() ;
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

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
void SiPixelInformationExtractor::fillImageBuffer() {
//cout<<"entering SiPixelInformationExtractor::fillImageBuffer"<<endl;
  canvas_->SetFixedAspectRatio(kTRUE);
  gStyle->SetPalette(1);
  // Now extract the image
  // 114 - stands for "no write on Close"
  TImageDump imgdump("tmp.png", 114);
  canvas_->Paint();

 // get an internal image which will be automatically deleted
 // in the imgdump destructor
  TImage *image = imgdump.GetImage();

  char *buf;
  int sz;
  image->GetImageBuffer(&buf, &sz);         /* raw buffer */
  pictureBuffer_.str("");
  for (int i = 0; i < sz; i++)
    pictureBuffer_ << buf[i];
  
  delete [] buf;
//cout<<"leaving SiPixelInformationExtractor::fillImageBuffer"<<endl;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
void SiPixelInformationExtractor::setCanvasMessage(const string& error_string) {
  TText tLabel;
  tLabel.SetTextSize(0.16);
  tLabel.SetTextColor(4);
  tLabel.DrawTextNDC(0.1, 0.5, error_string.c_str());
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
//void SiPixelInformationExtractor::plotHistosFromPath(MonitorUserInterface * mui, 
void SiPixelInformationExtractor::plotHistosFromPath(DaqMonitorBEInterface * bei, 
                                                     std::multimap<std::string, std::string>& req_map){

  cout << ACYellow << ACBold
       << "[SiPixelInformationExtractor::plotHistosFromPath()] "
       << ACPlain 
       << " Enter" 
       << endl ;
  //DaqMonitorBEInterface * bei = mui->getBEInterface();
  vector<string> item_list;  
  getItemList(req_map,"Path", item_list);
  
  if (item_list.size() == 0) 
  {
   cout << ACYellow << ACBold					  
   	<< "[SiPixelInformationExtractor::plotHistosFromPath()] " 
	<< ACBold << ACRed
	<< "Nothing to plot!"
   	<< ACPlain << endl ;					  
   return;
  }
  vector<MonitorElement*> me_list;
  string htype  = getItemValue(req_map,"histotype");
  if (htype.size() == 0) htype="individual";

  for (vector<string>::iterator it = item_list.begin(); it != item_list.end(); it++) {  

    string path_name = (*it);
    if (path_name.size() == 0) continue;
    
  //  MonitorElement* me = mui->get(path_name);
    MonitorElement* me = bei->get(path_name);

    if (me) me_list.push_back(me);
  }
  if (me_list.size() == 0) 
  {
   cout << ACYellow << ACBold					  
   	<< "[SiPixelInformationExtractor::plotHistosFromPath()] " 
	<< ACBold << ACRed
	<< "Nothing to plot!"
   	<< ACPlain << endl ;					  
   return; 
  }
  //if (htype == "summary") plotHistos(req_map, me_list, true);
  //else plotHistos(req_map, me_list, false); 
  plotHistos(req_map, me_list); 

  gROOT->Reset(); gStyle->SetPalette(1);
  fillImageBuffer();
  canvas_->Clear();
}
//------------------------------------------------------------------------------
//
// -- Read Layout Group names
//
void SiPixelInformationExtractor::readLayoutNames(xgi::Output * out){
  //  cout << "entering in SiPixelInformationExtractor::readLayoutNames" << endl;

  if (layoutMap.size() > 0) {
    out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
    *out << "<?xml version=\"1.0\" ?>" << std::endl;
    *out << "<LayoutList>" << endl;
    cout << "<?xml version=\"1.0\" ?>" << endl;
    cout << "<LayoutList>" << endl;

   for (map<string, vector< string > >::iterator it =  layoutMap.begin();
	                                         it != layoutMap.end(); 
                                                 it++) {
     *out << "<LName>" << it->first << "</LName>" << endl;  
     cout << "<LName>" << it->first << "</LName>" << endl;  
     for(vector<string>::iterator ivec =  it->second.begin();
	                          ivec != it->second.end(); 
                                  ivec++){
       *out << "<LMEName>" << *ivec << "</LMEName>" << endl;  
       cout << "<LMEName>" << *ivec << "</LMEName>" << endl;  
     }
   }
   *out << "</LayoutList>" << endl;
   cout << "</LayoutList>" << endl;
  }  
  //  cout << "leaving SiPixelInformationExtractor::readLayoutNames" << endl;
}

//------------------------------------------------------------------------------
//
// -- Plot Dummy Histograms from Layout
//
void SiPixelInformationExtractor::createDummiesFromLayout(){
  //  cout << "entering in SiPixelInformationExtractor::createDummiesFromLayout" << endl;

  if (layoutMap.size() == 0) return;

  canvas_->SetWindowSize(600,600);
  canvas_->Clear();

  for (map<std::string, std::vector< std::string > >::iterator it =  layoutMap.begin(); 
                                                               it != layoutMap.end(); 
                                                               it++) {
    for(vector<string>::iterator ivec =  it->second.begin();
                               	 ivec != it->second.end(); 
	                         ivec++){
      string fname  = "images/" + *ivec + ".png";
      setCanvasMessage("Plot not ready yet!!");
      canvas_->Print(fname.c_str(),"png");
      canvas_->Clear();

    }
  }
  //  cout << "leaving in SiPixelInformationExtractor::createDummiesFromLayout" << endl;
}
//------------------------------------------------------------------------------
//
// plot Histograms from Layout for Slide Show
//
void SiPixelInformationExtractor::plotHistosFromLayoutForSlideShow(DaqMonitorBEInterface * bei){
  //  cout << "entering in SiPixelInformationExtractor::plotHistosFromLayoutForSlideShow" << endl;

  //DaqMonitorBEInterface* bei = mui->getBEInterface();

  if (layoutMap.size() == 0) return;

  gStyle->SetOptStat("e");
  gStyle->SetStatFontSize(0.05);
  canvas_->SetWindowSize(600,600);
  canvas_->SetBorderMode(0);
  canvas_->SetFillColor(0);
  canvas_->Clear();

  paveOnCanvas->SetFillColor(0);

  TLine* l_ymin = new TLine();
  l_ymin->SetLineWidth(4);
  TLine* l_ymax = new TLine();
  l_ymax->SetLineWidth(4);

  for (map<std::string, std::vector< std::string > >::iterator it  = layoutMap.begin() ; 
                                                               it != layoutMap.end(); 
                                                             ++it) {
    gStyle->SetOptTitle(0);
    string subDet  = it->first;
    for (vector<string>::iterator im =  it->second.begin(); 
	                          im != it->second.end(); 
                                ++im) { 
      string meName = (*im);
      string path_name = "Collector/Collated/Tracker/" + subDet + "/" + meName;
      cout << "path_name: " << path_name << endl;

      if (path_name.size() == 0) continue;

      MonitorElement* me = bei->get(path_name.c_str());
      canvas_->Clear();
      if (!me){
	setCanvasMessage("ME doesn't exist!!");
      } else {
	TH1* histoME = ExtractTObject<TH1>().extract(me);
	if (histoME) {
	  TText tTitle;
	  tTitle.SetTextFont(64);
	  tTitle.SetTextSizePixels(20);
          setDrawingOption(histoME);
	  histoME->DrawCopy();
          string hname = histoME->GetTitle();
          tTitle.DrawTextNDC(0.1, 0.92, hname.c_str());
	  setSubDetAxisDrawing(histoME, hname);

	  double ymin = -1.;
	  double ymax = -1.;
	  string messageString = "";
	  int    messageColor  = 1;
	  string expThreshold;

	  dqm::qtests::QR_map qtestReportMap = me->getQReports();
	  
	  for(map<string, vector<string> >::iterator meQTestsMapItr =  meQTestsMap.begin();
	                                             meQTestsMapItr != meQTestsMap.end(); 
                                                   ++meQTestsMapItr){
	    if((meQTestsMapItr->first).find(meName.c_str()) == string::npos) continue;

	    vector<string> meQTestNameList = meQTestsMapItr->second;
	    if(meQTestNameList.size() > 0){

	      for(vector<string>::iterator meQTestNameListItr = meQTestNameList.begin();
		                           meQTestNameListItr != meQTestNameList.end(); 
                                         ++meQTestNameListItr){
		
		map<string,string> qtestParamsMap = qtestsMap[*meQTestNameListItr];
		
		string qtestType = "ContentsYRange";
		string test = qtestParamsMap["type"];
		if(test == qtestType){
		  ymin = atof(qtestParamsMap["ymin"].c_str());
		  ymax = atof(qtestParamsMap["ymax"].c_str());
		  
		  if(!qtestReportMap.empty()){
		    int qtestReportStatus = qtestReportMap[*meQTestNameListItr]->getStatus();
		    switch(qtestReportStatus){
		    case dqm::qstatus::STATUS_OK:
		      messageString = " OK ";
		      messageColor  = kGreen;
		      break;
		    case dqm::qstatus::WARNING:
		      messageString = " WARNING ";
		      messageColor  = kRed;
		      expThreshold  = qtestParamsMap["warning"];
		      break;
		    case dqm::qstatus::ERROR:
		      messageString = " ERROR ";
		      messageColor  = kMagenta;
		      expThreshold  = qtestParamsMap["error"];
		      break;
		    case dqm::qstatus::INSUF_STAT:
		      messageString = " NOT ENOUGH STATISTICS ";
		      messageColor  = kBlue;
		      break;
		    } // end of switch
		  } // end of if(!qtestReportMap.empty())
		  
		} // end of if (test == qtestType) 
	      } // end loop over meQTestNameListItr
	     
	      double xmin = histoME->GetXaxis()->GetXmin();
	      double xmax = histoME->GetXaxis()->GetXmax(); 
	      
	      l_ymin->SetLineColor(messageColor);
	      l_ymin->SetX1(xmin);
	      l_ymin->SetX2(xmax);
	      l_ymin->SetY1(ymin);
	      l_ymin->SetY2(ymin);
	      l_ymin->Draw("same");
	      l_ymax->SetLineColor(messageColor);
	      l_ymax->SetX1(xmin);
	      l_ymax->SetX2(xmax);
	      l_ymax->SetY1(ymax);
	      l_ymax->SetY2(ymax);
	      l_ymax->Draw("same");
	      
	      paveOnCanvas->Clear();
	      
	      TText* statusOnCanvas = paveOnCanvas->AddText(messageString.c_str());
	      statusOnCanvas->SetTextSize(0.06);
	      statusOnCanvas->SetTextFont(112);
	      statusOnCanvas->SetNDC(kTRUE);
	      statusOnCanvas->SetTextColor(messageColor);
	      
	      if(messageString.find("OK") ==  string::npos){
		string textMessage = "input threshold: " + expThreshold + "%";
		TText* messageOnCanvas = paveOnCanvas->AddText(textMessage.c_str());
		messageOnCanvas->SetTextSize(0.04);
		//	  messageOnCanvas->SetTextFont(112);
		messageOnCanvas->SetNDC(kTRUE);
	      }
	      paveOnCanvas->Draw("same");
	      
	    }
	    
	  } // end loop over meQTestsMapItr
	
	} else setCanvasMessage("Plot not ready yet!!"); 
      }
      string fileName = "images/" + meName + ".png";
      string command  = "rm -f " + fileName;
      gSystem->Exec(command.c_str());
      canvas_->Print(fileName.c_str(),"png");
      canvas_->Clear();
    }
  }
  delete l_ymin;
  delete l_ymax;

  //  cout << "leaving SiPixelInformationExtractor::plotHistosFromLayoutForSlideShow" << endl;
}

////
//// Error encoding overview creation
////
void SiPixelInformationExtractor::plotErrorOverviewHistos(DaqMonitorBEInterface * bei
					      //string                 module,
					     // string               & source,
					     // MonitorElement       * me,
					     // string               & qtest,
					     // int                    qtestStatus
					                                        ){
//  
//  cout << "entering in SiPixelInformationExtractor::encodeError" << endl;
//
//  DaqMonitorBEInterface* bei = mui->getBEInterface();
//
//  int nbin;
//  float xmin = 0;
//  float xmax = float(nbin);
//  TH1F* errorCode = new TH1F("errorCode","error code",nbin,xmin,xmax);
//
//
//  string module_path_name = "Collector/FU0/Tracker/PixelBarrel"; 
//  int qtestStatus = bei->getStatus(module_path_name);
//
//
////  if(source.find("RAW") ) sourceCode = 1;
////  if(source.find("DIGI")) sourceCode = 10;
////  if(source.find("CLU") ) sourceCode = 100;
////  if(source.find("TRK") ) sourceCode = 1000;
////  if(source.find("REC") ) sourceCode = 10000;
//
//  int statusCode = -1;
//  switch(qtestStatus){
//  case dqm::qstatus::STATUS_OK:
//    statusCode = 0;
//    cout << " qtestStatus: OK " << endl;
//    break;
//  case dqm::qstatus::INSUF_STAT:
//    statusCode = 1;
//    cout << " qtestStatus: INSUF_STAT " << endl;
//    break;
//  case dqm::qstatus::WARNING:
//    statusCode = 2;
//    cout << " qtestStatus: WARNING " << endl;
//    break;
//  case dqm::qstatus::ERROR:
//    statusCode = 3;
//    cout << " qtestStatus: ERROR " << endl;
//    break;
//  } // end of switch
//
}

//------------------------------------------------------------------------------
//
// -- Set Drawing Option
//
void SiPixelInformationExtractor::setDrawingOption(TH1   * hist, 
						   float   xlow, 
						   float   xhigh) {
  if (!hist) return;

  TAxis* xa = hist->GetXaxis();
  TAxis* ya = hist->GetYaxis();


  xa->SetTitleOffset(0.7);
  xa->SetTitleSize(0.06);
  xa->SetLabelSize(0.04);
  xa->SetLabelColor(0);
  xa->SetAxisColor(0);

  ya->SetTitleOffset(0.7);
  ya->SetTitleSize(0.06);


  if (xlow != -1 &&  xhigh != -1.0) {
    xa->SetRangeUser(xlow, xhigh);
  }
}

//------------------------------------------------------------------------------
//
// -- Set Axis Drawing Option for slide show plots
//
void SiPixelInformationExtractor::setSubDetAxisDrawing(TH1    * hist, 
						       string & detector) {
  if (!hist) return;

  string ownName;
  int    ownNOModules = -1;
  if(detector.find("Barrel") != string::npos){
    ownName      = "Shell";
    ownNOModules = 192;
  }
  if(detector.find("Endcap") != string::npos){
    ownName = "HalfCylinder";
    ownNOModules = 168;
  }

  TF1* subDet = new TF1("subDet","x",0,ownNOModules);
  TGaxis* subDet_mI = new TGaxis(  0,0,ownNOModules,0,"subDet",1,"U");
  subDet_mI->SetTitle((ownName + "_mI").c_str());
  subDet_mI->SetTitleOffset(0.4);
  subDet_mI->CenterTitle(kTRUE);
  TGaxis* subDet_mO = new TGaxis(ownNOModules+1,0,2*ownNOModules,0,"subDet",1,"U");
  subDet_mO->SetTitle((ownName + "_mO").c_str());
  subDet_mO->SetTitleOffset(0.4);
  subDet_mO->CenterTitle(kTRUE);
  TGaxis* subDet_pI = new TGaxis(2*ownNOModules+1,0,3*ownNOModules,0,"subDet",1,"U");
  subDet_pI->SetTitle((ownName + "_pI").c_str());
  subDet_pI->SetTitleOffset(0.4);
  subDet_pI->CenterTitle(kTRUE);
  TGaxis* subDet_pO = new TGaxis(3*ownNOModules+1,0,4*ownNOModules,0,"subDet",1,"U");
  subDet_pO->SetTitle((ownName + "_pO").c_str());
  subDet_pO->SetTitleOffset(0.4);
  subDet_pO->CenterTitle(kTRUE);

  subDet_mI->Draw("same");
  subDet_mO->Draw("same");
  subDet_pI->Draw("same");
  subDet_pO->Draw("same");

  delete subDet;
//  delete subDet_mI;
//  delete subDet_mO;
//  delete subDet_pI;
//  delete subDet_pO;

}
