#include <TStyle.h>
#include <string>
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include "TTree.h"
#include "TString.h"
#include "TAxis.h"
#include "TProfile.h"
#include "TH2F.h"
#include "TROOT.h"
#include "TDirectory.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TDirectoryFile.h"
#include "TLegend.h"
#include "THStack.h"
#include <exception>
#include "TKey.h"
#include "TPad.h"
#include "TPaveText.h"
#include "TPaveStats.h"
// This line works only if we have a CMSSW environment...
#include "Alignment/OfflineValidation/interface/TkOffTreeVariables.h"



class PlotAlignmentValidation {
public:
  //PlotAlignmentValidation(TString *tmp);
  PlotAlignmentValidation(const char *fileList,std::string fileName="");
  ~PlotAlignmentValidation();
  void loadFileList(const char *fileList,std::string fileName="");
  void plotOverview(const char *outputFileName="TrackerValidationOutput.ps",Int_t minHits = 50,bool plotBoxHisto = false,bool localXCoorinatesOn = false);
  void plotOverviewFittedValues(const char *outputFileNamee="TrackerValidationOutput.ps",Int_t minHits = 50,bool plotBoxHisto = false,bool localXCoorinatesOn = false);
  void plotOutlierModules(const char *outputFileName="OutlierModules.ps",std::string plotVariable = "chi2PerDofX" ,float chi2_cut = 10,Int_t minHits = 50);
  void plotSummedHistos(bool plotNormHisto=false);

private : 
  TList getTreeList();
 
  void plotBoxOverview(TCanvas &c1, TList &treeList,std::string plot_Var1a,std::string plot_Var1b, std::string plot_Var2, Int_t filenumber,Int_t minHits);
  void plot1DDetailsSubDet(TCanvas &c1, TList &treeList, std::string plot_Var1a,std::string plot_Var1b, std::string plot_Var2, Int_t minHits);
  void plot1DDetailsBarrelLayer(TCanvas &c1, TList &treeList, std::string plot_Var1a,std::string plot_Var1b, Int_t minHits);
  void plot1DDetailsDiskWheel(TCanvas &c1, TList &treelist, std::string plot_Var1a,std::string plot_Var1b, Int_t minHits);
  void setHistStyle( TH1& hist,const char* titleX, const char* titleY, int color);
  void setTitleStyle( TNamed& h,const char* titleX, const char* titleY, int subDetId);
  void setNiceStyle();
  void setCanvasStyle( TCanvas& canv );
  void setLegendStyle( TLegend& leg );

  TString outputFile;
  TList *sourcelist;
  bool moreThanOneSource;
  std::string fileNames[10];
  int fileCounter;	
};

PlotAlignmentValidation::PlotAlignmentValidation(const char *inputFile,std::string legendName)
{
  fileCounter=0;
  sourcelist=new TList();
  sourcelist->Add( TFile::Open( inputFile ));
  moreThanOneSource=false;
  fileNames[fileCounter]=legendName;
  ++fileCounter;	
}

PlotAlignmentValidation::~PlotAlignmentValidation()
{
}

void PlotAlignmentValidation::loadFileList(const char *inputFile, std::string legendName)
{
 
  sourcelist->Add(TFile::Open(inputFile) );
  moreThanOneSource=true;
  TFile *first_source = (TFile*)sourcelist->First();
  TFile *nextsource = (TFile*)sourcelist->After( first_source );
  while ( nextsource ) {
    nextsource = (TFile*)sourcelist->After( nextsource );
  }
  fileNames[fileCounter]=legendName;
  fileCounter++;	
}

void PlotAlignmentValidation::plotSummedHistos(bool plotNormHisto)
{
  
  TFile *first_source = (TFile*)sourcelist->First();
  int file_Counter=1;
  setNiceStyle();
 
  gStyle->SetOptStat(11111);
  gStyle->SetOptFit(0111);

  TCanvas *c = new TCanvas("c", "c", 600,600);
  c->SetTopMargin(0.15);
  int subDet=0;
  std::cout<<"Which sub-detector do you want to plot?"<<std::endl;
  std::cout<<"Type in the number for the subdetector corresponding to the following list ?"<<std::endl;
  std::cout<<"1.TPB, 2.TBE+, 3.TBE-, 4.TIB, 5.TID+, 6.TID-, 7.TOB, 8.TEC+ or 9.TEC-"<<std::endl;
  std::cin>>subDet;
  TString histoName= "";
  if (plotNormHisto) {histoName= "h_NormXprime";}
  else histoName= "h_Xprime_";
  switch (subDet){
  case 1 : histoName+="TPBBarrel_0";break;
  case 2 : histoName+="TPEendcap_1";break;
  case 3 : histoName+="TPEendcap_2";break;
  case 4 : histoName+="TIBBarrel_0";break;
  case 5 : histoName+="TIDEndcap_1";break;
  case 6 : histoName+="TIDEndcap_2";break;
  case 7 : histoName+="TOBBarrel_3";break;
  case 8 : histoName+="TECEndcap_4";break;
  case 9 : histoName+="TECEndcap_5";break;
  }
 
  TH1 *sumHisto =(TH1*) first_source->FindKeyAny(histoName)->ReadObj();//FindObjectAny(histoName.Data());
  sumHisto->Draw();
  
  //get statistic box coordinate to plot all boxes one below the other
  //gStyle->SetStatY(0.9);
  //gStyle->SetStatW(0.15);
  //gStyle->SetStatBorderSize(1);
  //gStyle->SetStatH(0.05);

  TPaveStats *s = (TPaveStats*)sumHisto->FindObject("stats");
  std::cout<<"stat box name: "<<s<<std::endl;

  TFile *nextsource = (TFile*)sourcelist->After( first_source );
  while ( nextsource ) {
    file_Counter++;
    sumHisto = (TH1*) nextsource->FindObjectAny(histoName);
    sumHisto->SetLineColor(file_Counter);
    sumHisto->SetLineStyle(file_Counter);
    //hstack->Add(sumHisto);
    sumHisto->Draw("sames");
    gStyle->SetStatY(0.91-file_Counter*0.05);
    gStyle->SetStatW(0.15);
    gStyle->SetStatBorderSize(1);
    gStyle->SetStatH(0.05);
    nextsource = (TFile*)sourcelist->After( nextsource );
    
  }
  //hstack->Draw("nostack");
  char PlotName[100];
  sprintf( PlotName, "%s.eps",histoName.Data() );
  
  c->Print(PlotName);

  
}



void PlotAlignmentValidation::plotOverview(const char *outputFileName,Int_t minHits,bool plotBoxHisto,bool localXCoorinatesOn)
{
 
  
  setNiceStyle(); 
  //gStyle->SetOptStat(0);
 
  TList treeList=getTreeList();
   TCanvas *c = new TCanvas("c", "c", 1200,400);
  setCanvasStyle( *c );
  c->Divide(3,1);
  //ps->NewPage();

  //-------------------------------------------------
  //plot Hit map
  //-------------------------------------------------
  std::string histName_="Entriesprofile";
  c->cd(1);
  TTree *first_tree=(TTree*)treeList.First();
  first_tree->Draw("entries:posR:posZ","","COLZ2Prof");
  c->cd(2);
  first_tree->Draw("entries:posY:posX","","COLZ2Prof");
  c->cd(3);
  first_tree->Draw("entries:posR:posPhi","","COLZ2Prof");
    
  char PlotName[100];
  sprintf( PlotName, "%s.eps",histName_.c_str() );
  
  c->Print(PlotName);
  //   //c->Update();
  c->Close();  
  //----------------------------------------------------

  TCanvas *c1 = new TCanvas("canv", "canv", 800, 500);
  setCanvasStyle( *c1 );
  //c1->Divide(3,2);
  outputFile = outputFileName;   
  c1->Print( (outputFile+'[').Data() ); 
 
  //---------------------------------------------------------------
  //2D box histograms for a single alignmentobject
  //---------------------------------------------------------------
  int i=1;
  if(plotBoxHisto){
    std::cout<<"Which file do you want to choose for the box plots?Type in 1,2,or 3?"<<std::endl;
    std::cin>>i; 
    plotBoxOverview(*c1,treeList,"meanX","rmsX","posZ",i,minHits);
    plotBoxOverview(*c1,treeList,"meanX","rmsX","posR",i,minHits);
    plotBoxOverview(*c1,treeList,"meanX","rmsX","posPhi",i,minHits);
    plotBoxOverview(*c1,treeList,"meanNormX","rmsNormX","posZ",i,minHits);
    plotBoxOverview(*c1,treeList,"meanNormX","rmsNormX","posR",i,minHits);
    plotBoxOverview(*c1,treeList,"meanNormX","rmsNormX","posPhi",i,minHits);
  }
  //--------------------------------------------------------------------
  //1d histograms with (normalized) residuals per subdetector
  //--------------------------------------------------------------------
  plot1DDetailsSubDet(*c1,treeList,"meanX","rmsX","posZ",minHits);
  plot1DDetailsSubDet(*c1,treeList,"meanNormX","rmsNormX","posZ",minHits);
  //1d histograms with (normalized) residuals in local coordinates per subdetector
  if (localXCoorinatesOn){
    plot1DDetailsSubDet(*c1,treeList,"meanLocalX","rmsLocalX","posZ",minHits);
    plot1DDetailsSubDet(*c1,treeList,"meanNormLocalX","rmsNormLocalX","posZ",minHits);
  }
  //---------------------------------------------------------------
  //1d histograms with (normalized) residuals per layer in TIB and TOB
  //---------------------------------------------------------------
  plot1DDetailsBarrelLayer(*c1,treeList,"meanX","rmsX",minHits);
  plot1DDetailsBarrelLayer(*c1,treeList,"meanNormX","rmsNormX",minHits);
  //1d histograms with (normalized) residuals in local coordinates per layer in TIB and TOB
  if (localXCoorinatesOn){
    plot1DDetailsBarrelLayer(*c1,treeList,"meanLocalX","rmsLocalX",minHits);
    plot1DDetailsBarrelLayer(*c1,treeList,"meanNormLocalX","rmsNormLocalX",minHits);
  }  
 //---------------------------------------------------------------
  //1d histograms with (normalized) residuals per ring in TID and TEC
  //---------------------------------------------------------------
  plot1DDetailsDiskWheel(*c1,treeList,"meanX","rmsX",minHits);
  plot1DDetailsDiskWheel(*c1,treeList,"meanNormX","rmsNormX",minHits);
  //1d histograms with (normalized) residuals in local coordinates per layer in TIB and TOB
  if (localXCoorinatesOn){
    plot1DDetailsDiskWheel(*c1,treeList,"meanLocalX","rmsLocalX",minHits);
    plot1DDetailsDiskWheel(*c1,treeList,"meanNormLocalX","rmsNormLocalX",minHits);
  }  



  c1->Print( (outputFile+"]").Data() );  
  //c1->Close();
 
}


void PlotAlignmentValidation::plotOutlierModules(const char *outputFileName,string plotVariable, float plotVariable_cut ,int minHits)
{
 
  Int_t counter=0;
  setNiceStyle();
  
  gStyle->SetOptStat(111111);
  gStyle->SetStatY(0.9);
  TList treelist=getTreeList();
   
  TCanvas *c1 = new TCanvas("canv", "canv", 800, 500);
  //setCanvasStyle( *c1 );
  outputFile = outputFileName;   
  c1->Print( (outputFile+'[').Data() ); 
  
  
  c1->Divide(2,1);
 
  TTree *tree= (TTree*)treelist.First();
  TkOffTreeVariables *treeMem = 0; // ROOT will initilise
  tree->SetBranchAddress("TkOffTreeVariables", &treeMem);
  
  
  Long64_t nentries =  tree->GetEntriesFast();
   
  for (Long64_t i = 0; i < nentries; i++){
    
    tree->GetEntry(i);
    float var = 0;
    if (plotVariable == "chi2PerDofX") var =treeMem->chi2PerDofX;
    else if(plotVariable == "chi2PerDofY") var =treeMem->chi2PerDofY;
    else if(plotVariable == "fitMeanX") var =treeMem->fitMeanX;
    else if(plotVariable == "fitMeanY") var =treeMem->fitMeanY;
    else if(plotVariable == "fitSigmaX") var =treeMem->fitSigmaX;
    else if(plotVariable == "fitSigmaY") var =treeMem->fitSigmaY;
    else {
      cout<<"There is no variable "<<plotVariable<<" included in the tree."<<endl;
      break;
    }
    if (var > plotVariable_cut && treeMem->entries > minHits)
      { 
	TFile *f=(TFile*)sourcelist->First();
	
	TH1 *h = (TH1*) f->FindKeyAny(treeMem->histNameX.c_str())->ReadObj();//f->FindObjectAny(treeMem->histNameX.c_str());
	gStyle->SetOptFit(0111);
	cout<<"hist name "<<h->GetName()<<endl;

	TString path =(char*)strstr( gDirectory->GetPath(), "TrackerOfflineValidation" );
	//cout<<"hist path "<<path<<endl;
	//cout<<"wrote text "<<endl;
	if(h) cout<<h->GetEntries()<<endl;

	//modules' location as title
	c1->cd(0);
	TPaveText * text=new TPaveText(0,0.95,0.99,0.99);
	text->AddText(path);
	text->SetFillColor(0);
	text->SetShadowColor(0);
	text->SetBorderSize( 0 );
	text->Draw();
	
	//residual histogram
	c1->cd(1);
	TPad *subpad = (TPad*)c1->GetPad(1);
	subpad->SetPad(0,0,0.5,0.94);
	h->Draw();
	
	//norm. residual histogram
	h = (TH1*) f->FindObjectAny(treeMem->histNameNormX.c_str());
	if(h) cout<<h->GetEntries()<<endl;
	c1->cd(2);
	TPad *subpad2 = (TPad*)c1->GetPad(2);
	subpad2->SetPad(0.5,0,0.99,0.94);
	h->Draw();
	
	c1->Print(outputFile);
	counter++;
      }
    
  }
  c1->Print( (outputFile+"]").Data() );
  if (counter == 0) cout<<"no bad modules found"<<endl;
 
  
  //read the number of entries in the t3
  //TTree* tree=0;
  //tree=(TTree*)treeList->At(0);
  
  
  //c1->Close();
 
}



TList PlotAlignmentValidation::getTreeList()
{
  TList treeList = new TList();
  TFile *first_source = (TFile*)sourcelist->First();
  std::cout<<first_source->GetName()<<std::endl;
  TDirectoryFile *d=(TDirectoryFile*)first_source->Get("TrackerOfflineValidation"); 
  treeList.Add( (TTree*)(*d).Get("TkOffVal") );
  if( moreThanOneSource ==true ){
    TFile *nextsource = (TFile*)sourcelist->After( first_source );
    while ( nextsource ) {
      std::cout<<nextsource->GetName()<<std::endl;
      d=(TDirectoryFile*)nextsource->Get("TrackerOfflineValidation"); 
      
      treeList.Add((TTree*)(*d).Get("TkOffVal"));
      
      nextsource = (TFile*)sourcelist->After( nextsource );
    }
  }return treeList;
}

void PlotAlignmentValidation::plotOverviewFittedValues(const char *outputFileName,Int_t minHits,bool plotBoxHisto,bool localXCoorinatesOn)
{
 
  
  setNiceStyle(); 
  gStyle->SetOptStat(0);
 
   TList treeList=getTreeList();
   TCanvas *c1 = new TCanvas("canv", "canv", 800, 500);
   setCanvasStyle( *c1 );
   //c1->Divide(3,2);
   outputFile = outputFileName;   
   c1->Print( (outputFile+'[').Data() ); 
 
  //---------------------------------------------------------------
  //2D box histograms for a single alignmentobject
  //---------------------------------------------------------------
  int i=1;
  if(plotBoxHisto){
    std::cout<<"Which file do you want to choose for the box plots?Type in 1,2,or 3?"<<std::endl;
    std::cin>>i; 
    plotBoxOverview(*c1,treeList,"fitMeanX","fitSigmaX","posZ",i,minHits);
    plotBoxOverview(*c1,treeList,"fitMeanX","fitSigmaX","posR",i,minHits);
    plotBoxOverview(*c1,treeList,"fitMeanX","fitSigmaX","posPhi",i,minHits);
    plotBoxOverview(*c1,treeList,"fitMeanNormX","fitSigmaNormX","posZ",i,minHits);
    plotBoxOverview(*c1,treeList,"fitMeanNormX","fitSigmaNormX","posR",i,minHits);
    plotBoxOverview(*c1,treeList,"fitMeanNormX","fitSigmaNormX","posPhi",i,minHits);
  }
  //--------------------------------------------------------------------
  //1d histograms with (normalized) residuals per subdetector
  //--------------------------------------------------------------------
  plot1DDetailsSubDet(*c1,treeList,"fitMeanX","fitSigmaX","posZ",minHits);
  plot1DDetailsSubDet(*c1,treeList,"fitMeanNormX","fitSigmaNormX","posZ",minHits);
  //1d histograms with (normalized) residuals in local coordinates per subdetector
  if (localXCoorinatesOn){
    plot1DDetailsSubDet(*c1,treeList,"meanLocalX","rmsLocalX","posZ",minHits);
    plot1DDetailsSubDet(*c1,treeList,"meanNormLocalX","rmsNormLocalX","posZ",minHits);
  }
  //---------------------------------------------------------------
  //1d histograms with (normalized) residuals per layer in TIB and TOB
  //---------------------------------------------------------------
  plot1DDetailsBarrelLayer(*c1,treeList,"fitMeanX","fitSigmaX",minHits);
  plot1DDetailsBarrelLayer(*c1,treeList,"fitMeanNormX","fitSigmaNormX",minHits);
  //1d histograms with (normalized) residuals in local coordinates per layer in TIB and TOB
  if (localXCoorinatesOn){
    plot1DDetailsBarrelLayer(*c1,treeList,"meanLocalX","rmsLocalX",minHits);
    plot1DDetailsBarrelLayer(*c1,treeList,"meanNormLocalX","rmsNormLocalX",minHits);
  }  
 //---------------------------------------------------------------
  //1d histograms with (normalized) residuals per ring in TID and TEC
  //---------------------------------------------------------------
  plot1DDetailsDiskWheel(*c1,treeList,"fitMeanX","fitSigmaX",minHits);
  plot1DDetailsDiskWheel(*c1,treeList,"fitMeanNormX","fitSigmaNormX",minHits);
  //1d histograms with (normalized) residuals in local coordinates per layer in TIB and TOB
  if (localXCoorinatesOn){
    plot1DDetailsDiskWheel(*c1,treeList,"meanLocalX","rmsLocalX",minHits);
    plot1DDetailsDiskWheel(*c1,treeList,"meanNormLocalX","rmsNormLocalX",minHits);
  }  



  c1->Print( (outputFile+"]").Data() );  
  //c1->Close();
 
}


void  PlotAlignmentValidation::plotBoxOverview(TCanvas &c1, TList &treelist, std::string plot_Var1a,std::string plot_Var1b, std::string plot_Var2, Int_t fileNumber,Int_t minHits)
{c1.Clear();
 c1.Divide(3,2);
  TTree* tree=0;
  if (fileNumber<=treelist.LastIndex()+1)  tree=(TTree*)treelist.At(fileNumber-1);
  else {
    std::cout<<"WARNING:The choosen value is out of range and set to 1 to avoid segmentation violation"<<std::endl;
    tree=(TTree*)treelist.At(0);
  }

  TString plotVar= plot_Var1a+':'+plot_Var2;
  Int_t counter=1;
  for (int i=1;i<7;++i){
    //c1.cd(i);
    TString subdet = "entries>=";
    subdet+=minHits; 
    subdet+=" && subDetId==";
    subdet+=i;
   
     
    TH2F *histo = 0;
    tree->Draw(plotVar+">>myHisto",subdet,"goff box");
    if (gDirectory) gDirectory->GetObject("myHisto", histo);
    if (histo) histo->SetDirectory(0);
    if (histo)setHistStyle(*histo,plot_Var2.c_str() ,plot_Var1a.c_str(),2);
    if (histo)setTitleStyle(*histo,plot_Var2.c_str() ,plot_Var1a.c_str(),i);
    if (histo) histo->SetLineColor(2);
 
    TProfile *hProf = 0;
    tree->Draw(plotVar+">>myProf",subdet,"goff prof");
    if (gDirectory) gDirectory->GetObject("myProf", hProf);
    if (hProf) hProf->SetDirectory(0);
    if (hProf) hProf->SetMarkerStyle(20);
    if (hProf) hProf->SetMarkerSize(0.5);
    
    //legend settings 
    TLegend *leg_hist = new TLegend(0.42,0.7,0.705,0.89);
    setLegendStyle(*leg_hist);
    std::string fileName=(std::string)(tree->GetCurrentFile()->GetName());
    Int_t start =0;
    if (fileName.find('_') )start =fileName.find_first_of('_')+1;
    Int_t stop = fileName.find_first_of('.');
    stop=stop-start;
    std::string legEntry = fileName.substr(start,stop);
    leg_hist->AddEntry(histo,legEntry.c_str(),"l");
   
    //Draw hProfisto
    if (hProf->GetEntries()>0)c1.cd(counter);
    if (histo->GetEntries()>0)histo->Draw("box");
    if (histo->GetEntries()>0)hProf->Draw("same");
    if (hProf->GetEntries()>0)leg_hist->Draw();
    if (hProf->GetEntries()>0)counter++;   
  }
  //if more than half of the sub-canvaces are filled 
  //print Canvas and reset counter
  if (counter>4){
    c1.Print(outputFile);
    counter =1;
  }else counter =4; 

  //plot corresponding rms in same canvas
  plotVar= plot_Var1b+':'+plot_Var2;
 
  for (int i=1;i<7;++i){
    TString subdet = "entries>=";
    subdet+=minHits; 
    subdet+=" && subDetId==";
    subdet+=i;
   
         
    TH2F *histo = 0;
    tree->Draw(plotVar+">>myHisto",subdet,"goff box");
    if (gDirectory) gDirectory->GetObject("myHisto", histo);
    if (histo) histo->SetDirectory(0);
    if (histo)setHistStyle(*histo,plot_Var2.c_str() ,plot_Var1b.c_str(),2);
    if (histo)setTitleStyle(*histo,plot_Var2.c_str() ,plot_Var1b.c_str(),i);
    if (histo) histo->SetLineColor(2);
 

    TProfile *hProf = 0;
    tree->Draw(plotVar+">>myProf",subdet,"goff prof");
    if (gDirectory) gDirectory->GetObject("myProf", hProf);
    if (hProf) hProf->SetDirectory(0);
    if (hProf) hProf->SetMarkerStyle(20);
    if (hProf) hProf->SetMarkerSize(0.5);

    TLegend *leg_hist = new TLegend(0.42,0.7,0.705,0.89);
    setLegendStyle(*leg_hist);
    std::string fileName;
    std::string legEntry;	
	if (fileNames[fileNumber-1]==""){
	fileName=(std::string)(tree->GetCurrentFile()->GetName());
    Int_t start =0;
    if (fileName.find('_') )start =fileName.find_first_of('_')+1;
    Int_t stop = fileName.find_last_of('.'); 
    stop=stop-start;
    legEntry = fileName.substr(start,stop);
	}else{
	fileName=fileNames[fileNumber];
	legEntry = fileName;
	}
	
    leg_hist->AddEntry(histo,legEntry.c_str(),"l");
       
     
    //Draw hProfisto
    if (hProf->GetEntries()>0)c1.cd(counter);
    if (hProf->GetEntries()>0)histo->Draw("box");
    if (histo->GetEntries()>0)hProf->Draw("same");
    if (hProf->GetEntries()>0)leg_hist->Draw();
    if (hProf->GetEntries()>0)counter++;   

  }
   
  c1.Print(outputFile);  

}  

void  PlotAlignmentValidation::plot1DDetailsSubDet(TCanvas &c1, TList &treelist, std::string plot_Var1a,std::string 
plot_Var1b, std::string plot_Var2, Int_t minHits)
{ c1.Clear();
  c1.Divide(3,2);
  //plot mean values for each sub-detector
  TString plotVar= plot_Var1a;
  Int_t canvas_Counter=0;
  
  //loop over sub-detectors 
  for (int i=1;i<7;++i){
    Int_t histo_Counter=1;
    Int_t file_Counter =1;
    TLegend *leg_hist = new TLegend(0.55,0.7,0.85,0.89);
    setLegendStyle(*leg_hist);
    //loop over file list
    TTree *tree= (TTree*)treelist.First();
    //binning
    int nbinsX=100;
    double xmin=0;
    double xmax=0;
    float maxY=0;
    bool isHisto = false;
    THStack *hstack=new THStack("hstack","hstack");
    while ( tree ){
     
      TString subdet = "entries>=";
      subdet+=minHits; 
      subdet+=" && subDetId==";
      subdet+=i;
      
      char binning [50];
      sprintf (binning, ">>myhisto(%d,  %f , %f", nbinsX, xmin, xmax);
     
      TH1F *h = 0;
      if (histo_Counter==1&&plot_Var1a=="meanX")tree->Draw(plotVar+">>myhisto(100,-0.01,0.01)",subdet,"goff");
      else if (histo_Counter==1&&plot_Var1a=="meanNormX")tree->Draw(plotVar+">>myhisto(100,-2,2)",subdet,"goff");
      else tree->Draw(plotVar+binning,subdet,"goff");
      if (gDirectory) gDirectory->GetObject("myhisto", h);
     
      if (h->GetEntries()>0) {
	isHisto = true;
	h->SetDirectory(0);
      //general draw options
	h->SetLineWidth(2);
      //first histo only, setting optStat...
      if (histo_Counter==1)setHistStyle(*h,plot_Var1a.c_str() ,"#modules",file_Counter);
      //settings for overlay histograms
      if (histo_Counter!=1){
	h->SetLineColor(file_Counter);
	h->SetLineStyle(file_Counter);
	h->SetMarkerStyle(20+file_Counter);
      }

      //     
      
      //draw options
      if (maxY<h->GetMaximum()){
	maxY=h->GetMaximum();
      }
      
      if (histo_Counter==1){
	canvas_Counter++;
	hstack->Add(h);
	nbinsX=h->GetXaxis()->GetNbins();
	xmin=h->GetXaxis()->GetXmin();
	xmax=h->GetXaxis()->GetXmax();
	//if (plot_Var1a=="meanX")	xmin=-0.02,xmax=0.02;
	//else if (plot_Var1a=="meanNormX")	xmin=-2,xmax=2;
	

      }else if (histo_Counter!=1 &&  h->GetEntries()>0)hstack->Add(h);
     
      std::string fileName;
      std::string legEntry; 
      if (fileNames[file_Counter-1]==""){	 
      fileName=(std::string)(tree->GetCurrentFile()->GetName());
      Int_t start =0;
      if (fileName.find('_') )start =fileName.find_first_of('_')+1;
      Int_t stop = fileName.find_last_of('.');
      legEntry = fileName.substr(start,stop-start);
       }else{
         fileName = fileNames[file_Counter-1];
	 legEntry = fileName;
	}
      if(h)leg_hist->AddEntry(h,legEntry.c_str(),"l");
      }
      tree= (TTree*)treelist.After( tree );
      file_Counter++;
      histo_Counter++;
     
    
    }
    if(canvas_Counter>0){
      c1.cd(canvas_Counter);
      if (isHisto){
	hstack->Draw("nostack");
	hstack->GetYaxis()->SetRangeUser(0,maxY*1.1);
	setTitleStyle(*hstack,plot_Var1a.c_str() ,"#modules",i);
	setHistStyle(*hstack->GetHistogram(),plot_Var1a.c_str() ,"#modules",file_Counter);
	leg_hist->Draw(); 
      }
    }
   
  }
  
  //if more than half of the sub-canvaces are filled 
  //print Canvas and reset counter 
  if (canvas_Counter>=4){
    c1.Print(outputFile);
    canvas_Counter =0;
  }else canvas_Counter =3;  
  
  //plot corresponding rms in same canvas
  plotVar= plot_Var1b;
  plot_Var2="entries";
  //loop over sub-detectors 
  for (int i=1;i<7;++i){
    Int_t histo_Counter=1;
    Int_t file_Counter =1;
    TLegend *leg_hist = new TLegend(0.55,0.7,0.85,0.89);
    setLegendStyle(*leg_hist);
    //loop over file list
    TTree *tree= (TTree*)treelist.First();
    //binning
    int nbinsX=100;
    double xmin=0;
    double xmax=0;
    double maxY=0;
    bool isHisto = false;
    THStack *hstack=new THStack("hstack","hstack");
    while ( tree ){
     
      TString subdet = "entries>=";
      subdet+=minHits; 
      subdet+=" && subDetId==";
      subdet+=i;
      
      char binning [50];
      sprintf (binning, ">>myhisto(%d,  %f , %f", nbinsX, xmin, xmax);
      
      TH1F *h = 0;
      
      if (histo_Counter==1)tree->Draw(plotVar+">>myhisto(100,,)",subdet,"goff");
      else tree->Draw(plotVar+binning,subdet,"goff");

      if (gDirectory) gDirectory->GetObject("myhisto", h);
      if(h) h->SetDirectory(0);
      if ( h->GetEntries()>0 ){ 
	isHisto = true;
	h->SetDirectory(0);
	//general draw options
	if (h) h->SetLineWidth(2);
	//first histo only, setting optStat...
	if (histo_Counter==1)setHistStyle(*h,plot_Var1b.c_str() ,"#modules",file_Counter);
	//settings for overlay histograms
	if (histo_Counter!=1){
	  h->SetLineColor(file_Counter);
	  h->SetLineStyle(file_Counter);
	  h->SetMarkerStyle(20+file_Counter);
	}

    
	//draw options
	if (histo_Counter==1 ){
	  canvas_Counter++;
	  hstack->Add(h);
	  nbinsX=h->GetXaxis()->GetNbins();
	  xmin=h->GetXaxis()->GetXmin();
	  xmax=h->GetXaxis()->GetXmax();
	}else if (histo_Counter!=1 )hstack->Add(h);

          
	std::string fileName;
	std::string legEntry;
	if(fileNames[file_Counter-1]==""){
	  fileName=(std::string)(tree->GetCurrentFile()->GetName());
	  Int_t start =0;
	  if (fileName.find('_') )start =fileName.find_first_of('_')+1;
	  Int_t stop = fileName.find_last_of('.');
	  legEntry = fileName.substr(start,stop-start);
	}else{
	  fileName=fileNames[file_Counter-1];
	  legEntry = fileName;
	}
	
	leg_hist->AddEntry(h,legEntry.c_str(),"l");
      }
      tree= (TTree*)treelist.After( tree );
      file_Counter++;
      histo_Counter++;
      
    }
    
    if(canvas_Counter>0){
      c1.cd(canvas_Counter);
      if (isHisto) {
	hstack->Draw("nostack");
	hstack->GetYaxis()->SetRangeUser(0,maxY*1.1);
	setTitleStyle(*hstack,plot_Var1b.c_str() ,"#modules",i);
	setHistStyle(*hstack->GetHistogram(),plot_Var1b.c_str() ,"#modules",file_Counter);
	leg_hist->Draw();
      } 
    }
  }
  c1.Print(outputFile); 
               
}
void  PlotAlignmentValidation::plot1DDetailsBarrelLayer(TCanvas &c1, TList &treelist, std::string plot_Var1a,std::string plot_Var1b, Int_t minHits)
{ 
  c1.Clear();
  c1.Divide(3,2);

  for (Int_t k=1;k<3;++k){//loop over Subdet (only TIB and TOB)
    Int_t subDetId=0;
    Int_t nLayer=0;
    if (k==1)subDetId=3,nLayer=4; //TIB
    if (k==2)subDetId=5,nLayer=6; //TOB
    c1.cd(5)->Clear();
    c1.cd(6)->Clear();
    
    //plot mean Value and rms for all Layers in the TIB/TOB
    Int_t canvas_Counter=0;
   

    for (Int_t j=1;j<3;++j){
      TString plotVar=0;
      if (j==1)  plotVar = plot_Var1a;//mean value
      if (j==2)  plotVar = plot_Var1b;//rms
      
      
      //loop over 4 layers 
      for (int i=1;i<=nLayer;++i){
	//c1.cd(i);
	TString cut_conditions = "entries>";
	cut_conditions+=minHits ;
	cut_conditions+="&& subDetId==";
	cut_conditions+=subDetId;
	cut_conditions+=" && layer==";
	cut_conditions+=i;
 

	Int_t histo_Counter=1;
	Int_t file_Counter =1;
	TLegend *leg_hist = new TLegend(0.55,0.7,0.85,0.89);
	setLegendStyle(*leg_hist);
	//loop over file list
	TTree *tree= (TTree*)treelist.First();
	//binning
	int nbinsX=100;
	double xmin=0;
	double xmax=0;
	bool isHisto = false;
	cout<<"tree "<<tree->GetName()<<endl;
	 THStack *hstack=new THStack("hstack","hstack");
	while ( tree ){//loop over all files
	  
	 
	  char binning [100];
	  sprintf (binning, ">>myhisto(%d,  %f , %f", nbinsX, xmin, xmax);
	  TH1F *h = 0;
	  
	  //completly nonsence but necessary to prevent root exeption (cause unkonwn)
	  //tree->Draw(plotVar+">>myhisto(100,,)",cut_conditions,"goff");
	 
	  if ( histo_Counter==1 &&plot_Var1a=="meanX"&&j==1) tree->Draw(plotVar+">>myhisto(100,-0.005,0.005)",cut_conditions,"goff");
	  else if ( histo_Counter==1 &&plot_Var1a=="meanNormX"&&j==1) tree->Draw(plotVar+">>myhisto(100,-1,1)",cut_conditions,"goff");
	  else if (j!=1)tree->Draw(plotVar+">>myhisto(100,,)",cut_conditions,"goff");
	  else tree->Draw(plotVar+binning,cut_conditions,"goff");
	  if (gDirectory) gDirectory->GetObject("myhisto", h);
	  if(h) h->SetDirectory(0);
	  if ( h && h->GetEntries()>0 ){ 
	    isHisto = true;
	    if ( (j==1)&& histo_Counter==1)setHistStyle(*h,plot_Var1a.c_str() ,cut_conditions.Data(),file_Counter );
	    if ( (j==2)&& histo_Counter==1)setHistStyle(*h,plot_Var1b.c_str() ,cut_conditions.Data(),file_Counter );
	    h->SetLineWidth(2);
	    //settings for overlay histograms
	    if (histo_Counter!=1){
	      h->SetLineColor(file_Counter);
	      h->SetLineStyle(file_Counter);
	      h->SetMarkerStyle(20+file_Counter);
	    }
	  

	    //draw options
	    if (histo_Counter==1 ){
	      canvas_Counter++;
	      hstack->Add(h);
	      nbinsX=h->GetXaxis()->GetNbins();
	      xmin=h->GetXaxis()->GetXmin();
	      xmax=h->GetXaxis()->GetXmax();
	    }else if (histo_Counter!=1)hstack->Add(h);
	  
	 
	    std::string fileName;
	    std::string legEntry;
	    if (fileNames[file_Counter-1]==""){
	      fileName=(std::string)(tree->GetCurrentFile()->GetName());
	      Int_t start =0;
	      if (fileName.find('_') )start =fileName.find_first_of('_')+1;
	      Int_t stop = fileName.find_last_of('.');
	      legEntry = fileName.substr(start,stop-start);
	    }else{
	      legEntry=fileNames[file_Counter-1];
	    }
	    leg_hist->AddEntry(h,legEntry.c_str(),"l");
	  }
	  tree= (TTree*)treelist.After( tree );
	  file_Counter++;
	  histo_Counter++;   
	 
	 
	}//end of while loop
	if(canvas_Counter>0){
	  c1.cd(canvas_Counter);
	  if (isHisto) {
	    hstack->Draw("nostack");
	    if (j==1){
	    
	      setTitleStyle(*hstack,plot_Var1a.c_str() ,cut_conditions.Data(),i);
	      setHistStyle(*hstack->GetHistogram(),plot_Var1a.c_str() ,cut_conditions.Data(),file_Counter);
	    }
	    if (j==2){
	      setTitleStyle(*hstack,plot_Var1b.c_str() ,cut_conditions.Data(),i);
	      setHistStyle(*hstack->GetHistogram(),plot_Var1b.c_str() ,cut_conditions.Data(),file_Counter);
	    }
	    leg_hist->Draw();
	    
	  }
	}
      }//end of for loop layer
      if (canvas_Counter>=4 && j==1){
	c1.Print(outputFile);
	canvas_Counter =0;
      }
    }//end of for loop over variables (mean or rms)
    
    c1.Print(outputFile); 
  }               
  

}

void  PlotAlignmentValidation::plot1DDetailsDiskWheel(TCanvas &c1, TList &treelist, std::string plot_Var1a,std::string plot_Var1b, Int_t minHits)
{
 
  for (Int_t k=1;k<3;++k){//loop over Subdet (only TIB and TOB)
    Int_t subDetId=0;
    Int_t nWheels=0;
    if (k==1)subDetId=4,nWheels=3,c1.Clear(),c1.Divide(3,2) ;//TID
    if (k==2)subDetId=6,nWheels=9,c1.Clear(),c1.Divide(3,3);//TEC
    c1.cd(5)->Clear();
    c1.cd(6)->Clear();
    c1.cd(7)->Clear();
    //plot mean Value and rms for all Layers in the TIB/TOB
    Int_t canvas_Counter=0;
    
    for (Int_t j=1;j<3;++j){
      TString plotVar=0;
      if (j==1)  plotVar= plot_Var1a;//mean value
      if (j==2)  plotVar= plot_Var1b;//rms
      
      
      //loop over 4 layers 
      for (int i=1;i<=nWheels;++i){
	//c1.cd(i);
	TString cut_conditions = "entries>";
	cut_conditions+=minHits ;
	cut_conditions+="&& subDetId==";
	cut_conditions+=subDetId;
	cut_conditions+=" && layer==";
	cut_conditions+=i;
 

	Int_t histo_Counter=1;
	Int_t file_Counter =1;
	TLegend *leg_hist = new TLegend(0.55,0.7,0.89,0.89);
	setLegendStyle(*leg_hist);
	//loop over file list
	TTree *tree= (TTree*)treelist.First();
	//binning
	int nbinsX=100;
	double xmin=0;
	double xmax=0;
	bool isHisto = false;
	THStack *hstack=new THStack("hstack","hstack");
	while ( tree ){//loop over all files
	  
	 
	  char binning [50];
	  sprintf (binning, ">>myhisto(%d,  %f , %f", nbinsX, xmin, xmax);
	  TH1F *h = 0;	  
	  if (histo_Counter==1)tree->Draw(plotVar+">>myhisto(100,,)",cut_conditions,"goff");
	  else tree->Draw(plotVar+binning,cut_conditions,"goff");
	  if (gDirectory) gDirectory->GetObject("myhisto", h);
	  if (h)  h->SetDirectory(0);
	  if ( h && h->GetEntries() > 0 ){
	     isHisto = true;
	    h->SetDirectory(0);
	    if ( (j==1) && histo_Counter==1 ) setHistStyle( *h,plot_Var1a.c_str(),cut_conditions.Data(),file_Counter );
	    if ( (j==2) && histo_Counter==1 ) setHistStyle( *h,plot_Var1b.c_str(),cut_conditions.Data(),file_Counter );
	    h->SetLineWidth(2);
	  
	  //settings for overlay histograms
	  if (histo_Counter!=1){
	    h->SetLineColor(file_Counter);
	    h->SetLineStyle(file_Counter);
	    h->SetMarkerStyle(20+file_Counter);
	  }
	  
	
	  //draw options
	  if (histo_Counter==1){
	    canvas_Counter++;
	    hstack->Add(h);
	    nbinsX=h->GetXaxis()->GetNbins();
	    xmin=h->GetXaxis()->GetXmin();
	    xmax=h->GetXaxis()->GetXmax();
	  }else if (histo_Counter!=1 )hstack->Add(h);
	  
	  std::string fileName;
	  std::string legEntry;
	  if (fileNames[file_Counter-1]==""){	  
	    std::string fileName=(std::string)(tree->GetCurrentFile()->GetName());
	    Int_t start =0;
	    if (fileName.find('_') )start =fileName.find_first_of('_')+1;
	    Int_t stop = fileName.find_last_of('.');
	    legEntry = fileName.substr(start,stop-start);
	  }
	  else{
	    fileName=fileNames[file_Counter-1],
	      legEntry = fileName;
	  }
	  if(h)leg_hist->AddEntry(h,legEntry.c_str(),"l");
	  }
	  tree= (TTree*)treelist.After( tree );
	  ++file_Counter;
	  ++histo_Counter;   
	 
	 
	}//end of while loop
	if(canvas_Counter>0){
	  c1.cd(canvas_Counter);
	  if (isHisto) {
	    if (hstack) {
	    hstack->Draw("nostack");
	    if (j==1){
	    
	      setTitleStyle(*hstack,plot_Var1a.c_str() ,cut_conditions.Data(),i);
	      setHistStyle(*hstack->GetHistogram(),plot_Var1a.c_str() ,cut_conditions.Data(),file_Counter);
	    }
	    if (j==2){
	      setTitleStyle(*hstack,plot_Var1b.c_str() ,cut_conditions.Data(),i);
	      setHistStyle(*hstack->GetHistogram(),plot_Var1b.c_str() ,cut_conditions.Data(),file_Counter);
	    }
	    if (leg_hist)leg_hist->Draw();
	  }
	  }
	}
      }//end of for loop layer
      if (canvas_Counter>=4 && j==1){
	c1.Print(outputFile);
	canvas_Counter =0;
      }
    }//end of for loop over variables (mean or rms)
    
    c1.Print(outputFile); 
  }               
  
  
}

void  PlotAlignmentValidation::setCanvasStyle( TCanvas& canv )
{
  canv.SetFillStyle   ( 4000 );
  canv.SetLeftMargin  ( 0.45 );
  canv.SetRightMargin ( 0.05 );
  canv.SetBottomMargin( 0.15 );
  canv.SetTopMargin   ( 0.05 );
}
void  PlotAlignmentValidation::setLegendStyle( TLegend& leg )
{
  leg.SetFillStyle ( 0 );
  leg.SetFillColor ( 0 );
  leg.SetBorderSize( 0 ); 
}

void  PlotAlignmentValidation::setNiceStyle() {
  TStyle *MyStyle = new TStyle ("MyStyle", "My style for nicer plots");
  
  Float_t xoff = MyStyle->GetLabelOffset("X"),
    yoff = MyStyle->GetLabelOffset("Y"),
    zoff = MyStyle->GetLabelOffset("Z");

  MyStyle->SetCanvasBorderMode ( 0 );
  MyStyle->SetPadBorderMode    ( 0 );
  MyStyle->SetPadColor         ( 0 );
  MyStyle->SetCanvasColor      ( 0 );
  MyStyle->SetTitleColor       ( 0 );
  MyStyle->SetStatColor        ( 0 );
  MyStyle->SetTitleBorderSize  ( 0 );
  MyStyle->SetTitleFillColor   ( 0 );
  MyStyle->SetTitleH        ( 0.07 );
  MyStyle->SetTitleW        ( 1.00 );
  MyStyle->SetTitleFont     (  132 );

  MyStyle->SetLabelOffset (1.5*xoff, "X");
  MyStyle->SetLabelOffset (1.5*yoff, "Y");
  MyStyle->SetLabelOffset (1.5*zoff, "Z");

  MyStyle->SetTitleOffset (0.9,      "X");
  MyStyle->SetTitleOffset (1.2,      "Y");
  MyStyle->SetTitleOffset (0.9,      "Z");

  MyStyle->SetTitleSize   (0.045,    "X");
  MyStyle->SetTitleSize   (0.045,    "Y");
  MyStyle->SetTitleSize   (0.045,    "Z");

  MyStyle->SetLabelFont   (132,      "X");
  MyStyle->SetLabelFont   (132,      "Y");
  MyStyle->SetLabelFont   (132,      "Z");

  MyStyle->SetPalette(1);

  MyStyle->cd();
}


void  PlotAlignmentValidation::setTitleStyle( TNamed &hist,const char* titleX, const char* titleY,int subDetId)
{
  std::stringstream titel_Xaxis;
  std::stringstream titel_Yaxis;
  TString titelXAxis=titleX;
  TString titelYAxis=titleY;
  cout<<"plot "<<titelXAxis<<" vs "<<titelYAxis<<endl;
  if (titelYAxis.Contains("layer")&& titelYAxis.Contains("subDetId==3") ){
    switch (subDetId) {
    case 1: hist.SetTitle("TIB Layer 1");break;
    case 2: hist.SetTitle("TIB Layer 2");break;
    case 3: hist.SetTitle("TIB Layer 3");break;
    case 4: hist.SetTitle("TIB Layer 4");break;
   
      //default:hist.SetTitle();
    }
  } else if ( titelYAxis.Contains("layer") && titelYAxis.Contains("subDetId==5") ){
    switch (subDetId) {
    case 1: hist.SetTitle("TOB Layer 1");break;
    case 2: hist.SetTitle("TOB Layer 2");break;
    case 3: hist.SetTitle("TOB Layer 3");break;
    case 4: hist.SetTitle("TOB Layer 4");break;
    case 5: hist.SetTitle("TOB Layer 5");break;
    case 6: hist.SetTitle("TOB Layer 6");break;
      //default:hist.SetTitle();
    }
  }  else if (titelYAxis.Contains("layer")&& titelYAxis.Contains("subDetId==4") ){
    switch (subDetId) {
    case 1: hist.SetTitle("TID Disk 1");break;
    case 2: hist.SetTitle("TID Disk 2");break;
    case 3: hist.SetTitle("TID Disk 3");break;
    case 4: hist.SetTitle("TID Disk 4");break;
   
      //default:hist.SetTitle();
    }
  } else if ( titelYAxis.Contains("layer") && titelYAxis.Contains("subDetId==6") ){
    switch (subDetId) {
    case 1: hist.SetTitle("TEC Disk 1");break;
    case 2: hist.SetTitle("TEC Disk 2");break;
    case 3: hist.SetTitle("TEC Disk 3");break;
    case 4: hist.SetTitle("TEC Disk 4");break;
    case 5: hist.SetTitle("TEC Disk 5");break;
    case 6: hist.SetTitle("TEC Disk 6");break;
    case 7: hist.SetTitle("TEC Disk 7");break;
    case 8: hist.SetTitle("TEC Disk 8");break;
    case 9: hist.SetTitle("TEC Disk 9");break;
      //default:hist.SetTitle();
    }
  }else{
    switch (subDetId){
    case 1: hist.SetTitle("Pixel Barrel");break;
    case 2: hist.SetTitle("Pixel Endcap");break;
    case 3: hist.SetTitle("Tracker Inner Barrel");break;
    case 4: hist.SetTitle("Tracker Inner Disk");break;
    case 5: hist.SetTitle("Tracker Outer Barrel");break;
    case 6: hist.SetTitle("Tracker End Cap");break;
      //default:hist.SetTitle();
    }    
  }
 
}


void  PlotAlignmentValidation::setHistStyle( TH1& hist,const char* titleX, const char* titleY, int color)
{
  std::stringstream titel_Xaxis;
  std::stringstream titel_Yaxis;
  TString titelXAxis=titleX;
  TString titelYAxis=titleY;
  
  if ( titelXAxis.Contains("Phi") )titel_Xaxis<<titleX<<"[rad]";
  else if( titelXAxis.Contains("meanX") )titel_Xaxis<<"#LTx'_{pred}-x'_{hit}#GT[cm]";
  else if( titelXAxis.Contains("rmsX") )titel_Xaxis<<"RMS(x'_{pred}-x'_{hit})[cm]";
  else if( titelXAxis.Contains("meanNormX") )titel_Xaxis<<"#LTx'_{pred}-x'_{hit}/#sigma#GT";
  else if( titelXAxis.Contains("rmsNormX") )titel_Xaxis<<"RMS(x'_{pred}-x'_{hit}/#sigma)";
  else if( titelXAxis.Contains("meanLocalX") )titel_Xaxis<<"#LTx_{pred}-x_{hit}#GT[cm]";
  else if( titelXAxis.Contains("rmsLocalX") )titel_Xaxis<<"RMS(x_{pred}-x_{hit})[cm]";
  else if( titelXAxis.Contains("meanNormLocalX") )titel_Xaxis<<"#LTx_{pred}-x_{hit}/#sigma#GT[cm]";
  else if( titelXAxis.Contains("rmsNormLocalX") )titel_Xaxis<<"RMS(x_{pred}-x_{hit}/#sigma)[cm]";
  else titel_Xaxis<<titleX<<"[cm]";
  
  if (hist.IsA()->InheritsFrom( TH1F::Class() ) )hist.SetLineColor(color);
  if (hist.IsA()->InheritsFrom( TProfile::Class() ) ) {
    hist.SetMarkerStyle(20);
    hist.SetMarkerSize(0.8);
    hist.SetMarkerColor(color);
  }
 

  hist.GetXaxis()->SetTitle( (titel_Xaxis.str()).c_str() );
  hist.GetXaxis()->SetTitleSize  ( 0.05 );
  hist.GetXaxis()->SetTitleColor (    1 );
  hist.GetXaxis()->SetTitleOffset(  1   );
  hist.GetXaxis()->SetTitleFont  (   62 );
  hist.GetXaxis()->SetLabelSize  ( 0.05 );
  hist.GetXaxis()->SetLabelFont  (   62 );
  //hist.GetXaxis()->CenterTitle   (      );
  hist.GetXaxis()->SetNdivisions (  505 );
 
  

  if ( titelYAxis.Contains("meanX") )titel_Yaxis<<"#LTx'_{pred}-x'_{hit}#GT[cm]";
  else if ( titelYAxis.Contains("rmsX") )titel_Yaxis<<"RMS(x'_{pred}-x'_{hit})[cm]";
  else if( titelYAxis.Contains("meanNormX") )titel_Yaxis<<"#LTx'_{pred}-x'_{hit}/#sigma#GT";
  else if( titelYAxis.Contains("rmsNormX") )titel_Yaxis<<"RMS(x_'{pred}-x'_{hit}/#sigma)";
  else if( titelYAxis.Contains("meanLocalX") )titel_Yaxis<<"#LTx_{pred}-x_{hit}#GT[cm]";
  else if( titelYAxis.Contains("rmsLocalX") )titel_Yaxis<<"RMS(x_{pred}-x_{hit})[cm]";
  else if ( titelYAxis.Contains("layer")&& titelYAxis.Contains("subDetId")||titelYAxis.Contains("#modules") )titel_Yaxis<<"#modules";
  else if ( titelYAxis.Contains("ring")&& titelYAxis.Contains("subDetId")||titelYAxis.Contains("#modules") )titel_Yaxis<<"#modules";
  else titel_Yaxis<<titleY<<"[cm]";

  

  hist.GetYaxis()->SetTitle( (titel_Yaxis.str()).c_str()  );
  //hist.SetMinimum(1);
  hist.GetYaxis()->SetTitleSize  ( 0.05 );
  hist.GetYaxis()->SetTitleColor (    1 );
  if ( hist.IsA()->InheritsFrom( TH2::Class() ) ) hist.GetYaxis()->SetTitleOffset( 0.95 );
  else hist.GetYaxis()->SetTitleOffset( 1.0 );
  hist.GetYaxis()->SetTitleFont  (   62 );
  hist.GetYaxis()->SetLabelSize  ( 0.03 );
  hist.GetYaxis()->SetLabelFont  (   62 );

 
}
