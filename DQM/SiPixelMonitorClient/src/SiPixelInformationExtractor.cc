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
#include <math.h>

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
}

//------------------------------------------------------------------------------
/*! \brief Destructor of the SiPixelInformationExtractor class.
 *  
 */
SiPixelInformationExtractor::~SiPixelInformationExtractor() {
  edm::LogInfo("SiPixelInformationExtractor") << 
    " Deleting SiPixelInformationExtractor " << "\n" ;
  
  if (canvas_) delete canvas_;
}

//------------------------------------------------------------------------------
/*! \brief Read Configuration File
 *
 */
void SiPixelInformationExtractor::readConfiguration() {
  //  cout << "entering in SiPixelInformationExtractor::readConfiguration" << endl;

  // read layout configuration file
  // ------------------------------
/*  string localPath = string("DQM/SiPixelMonitorClient/test/sipixel_plot_layout_config.xml");
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
*/
}
//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *  
 */
void SiPixelInformationExtractor::createModuleTree(DaqMonitorBEInterface* bei) {
//cout<<"entering SiPixelInformationExtractor::createModuleTree..."<<endl;
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
void SiPixelInformationExtractor::fillBarrelList(DaqMonitorBEInterface* bei,
                                                 string dir_name,
						 vector<string>& me_names) {
  //cout<<"entering SiPixelInformationExtractor::fillBarrelList..."<<endl;
  string currDir = bei->pwd();
  if (currDir.find(dir_name) != string::npos)  {
    vector<MonitorElement*> mod_mes;

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
      if((*it).find("Endcap")!=string::npos) continue;
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
void SiPixelInformationExtractor::fillEndcapList(DaqMonitorBEInterface* bei,
                                                 string dir_name,
						 vector<string>& me_names) {
  //cout<<"entering SiPixelInformationExtractor::fillEndcapList..."<<endl;
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
      if((bei->pwd()).find("Barrel")!=string::npos) bei->goUp();
      bei->cd((*it));
      if((*it).find("Barrel")!=string::npos) continue;
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
MonitorElement* SiPixelInformationExtractor::getModuleME(DaqMonitorBEInterface* bei,
                                                         string me_name) {
//cout<<"Entering SiPixelInformationExtractor::getModuleME..."<<endl;
  MonitorElement* me = 0;
  // If already booked

  vector<string> contents = bei->getMEs();   
   
  for (vector<string>::const_iterator it = contents.begin();
       it != contents.end(); it++) {
    if ((*it).find(me_name) == 0) {
      string fullpathname = bei->pwd() + "/" + (*it); 

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
void SiPixelInformationExtractor::selectSingleModuleHistos(DaqMonitorBEInterface   * bei,  
                                                           string                    mid,  
							   vector<string>          & names,
							   vector<MonitorElement*> & mes) 
{  
  string currDir = bei->pwd();
  QRegExp rx("(\\w+)_(siPixel|ctfWithMaterialTracks)") ;
  QString theME ;
  if (currDir.find("Module_") != string::npos ||
      currDir.find("FED_") != string::npos)  
  {
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
          if( rx.search(theME) != -1 ) { temp_s = rx.cap(1).latin1() ; }
	  if (temp_s == (*ih)) 
	  {
	    string full_path = currDir + "/" + (*it);
	    //cout<<"full_path="<<full_path<<endl;

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
void SiPixelInformationExtractor::plotSingleModuleHistos(DaqMonitorBEInterface* bei, 
                                                         multimap<string, string>& req_map) {
//cout<<"entering SiPixelInformationExtractor::plotSingleModuleHistos"<<endl;
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
void SiPixelInformationExtractor::plotTkMapHisto(DaqMonitorBEInterface * bei, 
                                                 string                  theModId, 
						 string                  theMEName) 
{
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
  int icW = cW.toInt() ;
  int icH = cH.toInt() ;
  TCanvas * theCanvas = new TCanvas("TrackerMapPlotsCanvas", 
                                    "TrackerMapPlotsCanvas",
				    icW,
				    icH);
  gROOT->Reset(); 
  gStyle->SetPalette(1,0);
  theCanvas->SetBorderMode(0);
  theCanvas->SetFillColor(0);
  
  TLine* l_min = new TLine();
  TLine* l_max = new TLine();
  l_min->SetLineWidth(4);
  l_max->SetLineWidth(4);
  
  TPaveText * paveOnCanvas = new TPaveText(0.57,0.79,0.77,0.99,"NDCtr");   
  paveOnCanvas->SetFillColor(0);
  

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

    int istat =  SiPixelUtility::getStatus(theMe);
	
    TH1F* histoME = ExtractTObject<TH1F>().extract(theMe);
    //cout << "theMEName: " << theMEName << endl;
    string var = theMEName.substr(theMEName.find_last_of("/")+1);
    //cout << "var: " << var << endl;
    
    //cout<<"istat="<<istat<<endl;
    if(istat!=0){
      string tag;
      int icol;
      SiPixelUtility::getStatusColor(istat, icol, tag);
      
      TText* statusOnCanvas = paveOnCanvas->AddText(tag.c_str());
      statusOnCanvas->SetTextSize(0.08);
      statusOnCanvas->SetTextFont(112);
      statusOnCanvas->SetNDC(kTRUE);
      statusOnCanvas->SetTextColor(icol);
      
      double ymax = -1.;  
      double ymin = -1.;
      double xmax = -1.;  
      double xmin = -1.;
      double warning = -1.;
      double error = -1.;
      double channelFraction = -1.;
      //if(var.find("SUM") != string::npos){
	//cout << "ME name: " << var << endl;
	setLines(theMe,var,ymin,ymax,warning,error,channelFraction);
	//cout << "ymin: " << ymin << " ymax: " << ymax << " warning: " << warning << " error: " << error << " channelFraction: " << channelFraction << endl;
	
	if(istat!=dqm::qstatus::STATUS_OK){
	  string textMessage = "fraction of channels failing:";
	  TText* messageOnCanvas = paveOnCanvas->AddText(textMessage.c_str());
	  messageOnCanvas->SetTextSize(0.03);
	  messageOnCanvas->SetNDC(kTRUE);
	  char text[10];
	  sprintf(text,"%.2f %%",channelFraction);
	  messageOnCanvas = paveOnCanvas->AddText(text);
	  messageOnCanvas->SetTextSize(0.035);
	  messageOnCanvas->SetNDC(kTRUE);
	  /*char newtext1[25];
	  sprintf(newtext1,"(warning level: %.0f %%)",warning);
	  messageOnCanvas = paveOnCanvas->AddText(newtext1);
	  messageOnCanvas->SetTextSize(0.025);
	  messageOnCanvas->SetNDC(kTRUE);
	  char newtext2[25];
	  sprintf(newtext2,"(error level: %.0f %%)",error);
	  messageOnCanvas = paveOnCanvas->AddText(newtext2);
	  messageOnCanvas->SetTextSize(0.025);
	  messageOnCanvas->SetNDC(kTRUE);
	  */
	}
	if(ymin!= -1. && ymax!=-1.){
	  SiPixelUtility::setDrawingOption(histoME);
	  l_min->SetLineColor(icol);
	  l_max->SetLineColor(icol);
          if(var.find("SUM") != string::npos){	  
	    xmin = histoME->GetXaxis()->GetXmin();
	    xmax = histoME->GetXaxis()->GetXmax(); 
	    //cout<<"xmin="<<xmin<<" , xmax="<<xmax<<" , ymin="<<ymin<<" , ymax="<<ymax<<endl;
	    l_min->SetX1(xmin);
	    l_min->SetX2(xmax);
	    l_min->SetY1(ymin);
	    l_min->SetY2(ymin);
	    l_min->Draw("same");
	    l_max->SetX1(xmin);
	    l_max->SetX2(xmax);
	    l_max->SetY1(ymax);
	    l_max->SetY2(ymax);
	    l_max->Draw("same");
	  }else{
	    xmin = ymin;
	    xmax = ymax;
	    ymin = histoME->GetYaxis()->GetBinLowEdge(1);
	    ymax = histoME->GetMaximum();
	    //cout<<"xmin="<<xmin<<" , xmax="<<xmax<<" , ymin="<<ymin<<" , ymax="<<ymax<<endl;
	    l_min->SetX1(xmin);
	    l_min->SetX2(xmin);
	    l_min->SetY1(ymin);
	    l_min->SetY2(ymax);
	    l_min->Draw("same");
	    l_max->SetX1(xmax);
	    l_max->SetX2(xmax);
	    l_max->SetY1(ymin);
	    l_max->SetY2(ymax);
	    l_max->Draw("same");
	  }
	}
	//setSubDetAxisDrawing(theMEName,histoME);
      //}
      paveOnCanvas->Draw("same");
    }
    if(((var.find("Barrel") != string::npos) && (var.find("SUM") != string::npos)) ||
       ((var.find("Endcap") != string::npos) && (var.find("SUM") != string::npos))) 
      setSubDetAxisDrawing(theMEName,histoME);
    
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
  delete paveOnCanvas;
  delete l_min;
  delete l_max;
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
void SiPixelInformationExtractor::plotSingleHistogram(DaqMonitorBEInterface * bei,
		                                      std::multimap<std::string, 
						      std::string>& req_map){
//cout<<"entering SiPixelInformationExtractor::plotSingleHistogram"<<endl;
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
void SiPixelInformationExtractor::readModuleAndHistoList(DaqMonitorBEInterface* bei, 
                                                         xgi::Output * out) {
//cout<<"entering SiPixelInformationExtractor::readModuleAndHistoList"<<endl;
   std::map<std::string,std::string> hnames;
   std::vector<std::string> mod_names;
   //if (coll_flag)  bei->cd("Collector/Collated");
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
   //if (coll_flag)  bei->cd();
//cout<<"leaving SiPixelInformationExtractor::readModuleAndHistoList"<<endl;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
void SiPixelInformationExtractor::fillModuleAndHistoList(DaqMonitorBEInterface * bei, 
                                                         vector<string>        & modules,
							 map<string,string>    & histos) {
//cout<<"entering SiPixelInformationExtractor::fillModuleAndHistoList"<<endl;
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
void SiPixelInformationExtractor::readModuleHistoTree(DaqMonitorBEInterface* bei, 
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
void SiPixelInformationExtractor::printModuleHistoList(DaqMonitorBEInterface * bei, 
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
void SiPixelInformationExtractor::readSummaryHistoTree(DaqMonitorBEInterface* bei, 
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
void SiPixelInformationExtractor::printSummaryHistoList(DaqMonitorBEInterface * bei, 
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
void SiPixelInformationExtractor::readAlarmTree(DaqMonitorBEInterface* bei, 
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
void SiPixelInformationExtractor::printAlarmList(DaqMonitorBEInterface * bei, 
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
    dqm::qtests::QR_map my_map = me->getQReports();
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
const ostringstream&  SiPixelInformationExtractor::getIMGCImage(DaqMonitorBEInterface* bei, 
                                                                std::string theFullPath,
								std::string canvasW,
								std::string canvasH) 
{
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
//	<< "[SiPixelInformationExtractor::getNamedImage()]"
//	<< ACPlain
//	<< " Requested " << theName
//	<< endl ;
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
//	 << "[SiPixelInformationExtractor::getNamedImage()] "
//       << ACPlain 
//       << "Buffer containing picture found for "
//       << theName
//       << endl ;
  }
  pictureBuffer_ << thisBuffer->second ;
//   cout << ACCyan << ACBold << ACReverse 
//	<< "[SiPixelInformationExtractor::getNamedImage()]"
//	<< ACPlain
//	<< " Returning " << theName
//	<< endl ;
  return pictureBuffer_;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
bool SiPixelInformationExtractor::goToDir(DaqMonitorBEInterface* bei, 
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
void SiPixelInformationExtractor::readStatusMessage(DaqMonitorBEInterface* bei, 
                                                    string& path,
						    xgi::Output * out) {
//cout<<"entering SiPixelInformationExtractor::readStatusMessage"<<endl;
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
void SiPixelInformationExtractor::selectMEList(DaqMonitorBEInterface   * bei,  
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
void SiPixelInformationExtractor::sendTkUpdatedStatus(DaqMonitorBEInterface  * bei, 
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
void SiPixelInformationExtractor::getMEList(DaqMonitorBEInterface    * bei,  
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
/*void SiPixelInformationExtractor::readLayoutNames(xgi::Output * out){
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
*/
//------------------------------------------------------------------------------
//
// -- Plot Dummy Histograms from Layout
//
/*void SiPixelInformationExtractor::createDummiesFromLayout(){
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
}*/
//------------------------------------------------------------------------------
//
// -- Set Axis Drawing Option for slide show plots
//
void SiPixelInformationExtractor::setSubDetAxisDrawing(string detector, TH1F * histo) {

  histo->GetXaxis()->SetLabelColor(0);

  string ownName = "";
  int    ownNOModules = 0;
  if(detector.find("Barrel") != string::npos || detector.find("Endcap") != string::npos){ 
    if(detector.find("Barrel") != string::npos){
      ownName      = "Shell";
      ownNOModules = 192;
    }
    if(detector.find("Endcap") != string::npos){
      ownName = "HalfCylinder";
      ownNOModules = 168;
    }
    
    //cout << "ownName: " << ownName << " ownNOModules: " << ownNOModules << endl;
    
    TText tt;
    tt.SetTextSize(0.04);
    string mI = ownName + "_mI"; tt.DrawTextNDC(0.12, 0.04, mI.c_str());
    string mO = ownName + "_mO"; tt.DrawTextNDC(0.32, 0.04, mO.c_str());
    string pI = ownName + "_pI"; tt.DrawTextNDC(0.52, 0.04, pI.c_str());
    string pO = ownName + "_pO"; tt.DrawTextNDC(0.72, 0.04, pO.c_str());
  }
  if(detector.find("Shell") != string::npos){
    ownName = "Layer";
    TText tt;
    tt.SetTextSize(0.04);
    string l_one   = ownName + "_1"; tt.DrawTextNDC(0.12, 0.03, l_one.c_str());
    string l_two   = ownName + "_2"; tt.DrawTextNDC(0.42, 0.03, l_two.c_str());
    string l_three = ownName + "_3"; tt.DrawTextNDC(0.72, 0.03, l_three.c_str());
  }
  if(detector.find("HalfCylinder") != string::npos){
    ownName = "Disk";
    TText tt;
    tt.SetTextSize(0.04);
    string d_one   = ownName + "_1"; tt.DrawTextNDC(0.12, 0.03, d_one.c_str());
    string d_two   = ownName + "_2"; tt.DrawTextNDC(0.62, 0.03, d_two.c_str());
  }
}

void SiPixelInformationExtractor::coloredHotModules(TH1F        * histo,
						    vector<int> & binList,
						    int           range,
						    int           color){
  if(!histo) return;

  for(vector<int>::iterator bin =  binList.begin();
                            bin != binList.end(); 
                            bin++){
    histo->SetFillColor(color);
    histo->GetXaxis()->SetRange(*bin,*bin+range);
    histo->DrawCopy("same");
  }

}


void SiPixelInformationExtractor::setLines(MonitorElement * me,
					   string & meName, 
					   double & ymin,
					   double & ymax, 
					   double & warning, 
					   double & error, 
					   double & channelFraction) {
//cout<<"Entering SiPixelInformationExtractor::setLines for "<<meName<<endl;
   std::vector<QReport *> report;
   std::string colour;

   if (me->hasError()){
     colour="red";
     report= me->getQErrors();
   } else if( me->hasWarning()){ 
     colour="orange";
     report= me->getQWarnings();
   } else if(me->hasOtherReport()){
     colour="black";
     report= me->getQOthers();
   } else {
     colour="green";
   }
   for(std::vector<QReport *>::iterator itr=report.begin(); itr!=report.end();++itr ){
     std::string text= (*itr)->getMessage();
     //std::cout<<"ME: "<<meName<<" QTmessage:"<<text<<std::endl;
     int num1 = text.find_first_of(":")+1;
     int num2 = text.find_first_of("-",num1)+1;
     int num3 = text.find_first_of(" ",num2);
     std::stringstream dummy(text.substr(num1,num2-num1-1));
     dummy >> ymin; 
     std::stringstream dummy1(text.substr(num2,num3-num2));
     dummy1 >> ymax; 
     //std::cout<<",ymin="<<ymin<<",ymax="<<ymax<<std::endl;
     int num4 = text.find_last_of("=")+2;
     std::stringstream dummy2(text.substr(num4));
     dummy2 >> channelFraction;
     channelFraction = (1.- channelFraction)*100.;
     error = 25.;
     warning = 10.;
     //std::cout<<",warning="<<warning<<",error="<<error<<std::endl;
   }
	  
}
