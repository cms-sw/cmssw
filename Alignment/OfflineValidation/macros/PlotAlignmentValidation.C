#include <TStyle.h>
#include <TSystem.h>
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
#include "TF1.h"

// This line works only if we have a CMSSW environment...
#include "Alignment/OfflineValidation/interface/TkOffTreeVariables.h"

class TkOfflineVariables {
public:
  TkOfflineVariables(std::string fileName, std::string baseDir, std::string legName="", int color=1, int style=1);
  int getLineColor(){ return lineColor; };
  int getLineStyle(){ return lineStyle; };
  std::string getName(){ return legendName; }
  TTree* getTree(){ return tree; };
  TFile* getFile(){ return file; };
private:
  TFile* file;
  TTree* tree;
  int lineColor;
  int lineStyle;
  std::string legendName;
};

bool useFit_ =false;


TkOfflineVariables::TkOfflineVariables(std::string fileName, std::string baseDir, std::string legName, int lColor, int lStyle)
{
  lineColor = lColor;
  lineStyle = lStyle;
  if (legName=="") {
    int start = 0;
    if (fileName.find('/') ) start =fileName.find_last_of('/')+1;
    int stop = fileName.find_last_of('.');
    legendName = fileName.substr(start,stop-start);
  } else { 
    legendName = legName;
  }

  //fill the tree pointer
  file = TFile::Open( fileName.c_str() );
  TDirectoryFile *d = 0;
  if (file->Get( baseDir.c_str() ) )  {
    d = (TDirectoryFile*)file->Get( baseDir.c_str() );
    if ((*d).Get("TkOffVal")) {
      tree = (TTree*)(*d).Get("TkOffVal");
    } else {
      std::cout<<"no tree named TkOffVal"<<std::endl;
    }
  } else {
    std::cout<<"no directory named "<<baseDir.c_str()<<std::endl;
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

class PlotAlignmentValidation {
public:
  //PlotAlignmentValidation(TString *tmp);
  PlotAlignmentValidation(const char *inputFile,std::string fileName="", int lineColor=1, int lineStyle=1);
  ~PlotAlignmentValidation();
  void loadFileList(const char *inputFile, std::string fileName="", int lineColor=2, int lineStyle=1);
  void useFitForDMRplots(bool usefit = false);
  void plotOutlierModules(const char *outputFileName="OutlierModules.ps",std::string plotVariable = "chi2PerDofX" ,float chi2_cut = 10,unsigned int minHits = 50);//method dumps selected modules into ps file
  void plotSubDetResiduals(bool plotNormHisto=false, unsigned int subDetId=7);//subDetector number :1.TPB, 2.TBE+, 3.TBE-, 4.TIB, 5.TID+, 6.TID-, 7.TOB, 8.TEC+ or 9.TEC-
  void plotDMR(const std::string& plotVar="medianX",Int_t minHits = 50, const std::string& options = "plain");
  void plotHitMaps();
  void setOutputDir( std::string dir );
  void setTreeBaseDir( std::string dir = "TrackerOfflineValidationStandalone");
  
  TH1* addHists(const char *selection, const TString &residType = "xPrime", bool printModuleIds = false);//add hists fulfilling 'selection' on TTree; residType: xPrime,yPrime,xPrimeNorm,yPrimeNorm,x,y,xNorm; if (printModuleIds): cout DetIds
  
private : 
  TList getTreeList();
  std::string treeBaseDir;
  
  std::pair<float,float> fitGauss(TH1 *hist,int color);
  //void plotBoxOverview(TCanvas &c1, TList &treeList,std::string plot_Var1a,std::string plot_Var1b, std::string plot_Var2, Int_t filenumber,Int_t minHits);
  //void plot1DDetailsSubDet(TCanvas &c1, TList &treeList, std::string plot_Var1a,std::string plot_Var1b, std::string plot_Var2, Int_t minHits);
  //void plot1DDetailsBarrelLayer(TCanvas &c1, TList &treeList, std::string plot_Var1a,std::string plot_Var1b, Int_t minHits);
  //void plot1DDetailsDiskWheel(TCanvas &c1, TList &treelist, std::string plot_Var1a,std::string plot_Var1b, Int_t minHits);
  void setHistStyle( TH1& hist,const char* titleX, const char* titleY, int color);
  void setTitleStyle( TNamed& h,const char* titleX, const char* titleY, int subDetId);
  void setNiceStyle();
  void setCanvasStyle( TCanvas& canv );
  void setLegendStyle( TLegend& leg );

  TString outputFile;
  std::string outputDir;
  TList *sourcelist;
  std::vector<TkOfflineVariables*> sourceList;
  bool moreThanOneSource;
  std::string fileNames[10];
  int fileCounter;	

  // These are helpers for DMR plotting

  struct DMRPlotInfo {
    std::string variable;
    int nbins;
    double min, max;
    int minHits;
    bool plotPlain, plotSplits;
    int subDetId;
    THStack* hstack;
    TLegend* legend;
    TkOfflineVariables* vars;
    float maxY;
    TH1F* h;
    TH1F* h1;
    TH1F* h2;
    bool firsthisto;
  };

  std::string getSelectionForDMRPlot(int minHits, int subDetId, int direction);
  std::string getVariableForDMRPlot(const std::string& histoname, const std::string& variable,
				    int nbins, double min, double max);
  void setDMRHistStyleAndLegend(TH1F* h, DMRPlotInfo& plotinfo, int direction);
  void plotDMRHistogram(DMRPlotInfo& plotinfo, int direction);

};

//------------------------------------------------------------------------------
PlotAlignmentValidation::PlotAlignmentValidation(const char *inputFile,std::string legendName, int lineColor, int lineStyle)
{
  setOutputDir("$TMPDIR");
  setTreeBaseDir();
  sourcelist = NULL;
  
  loadFileList( inputFile, legendName, lineColor, lineStyle);
  moreThanOneSource=false;
}

//------------------------------------------------------------------------------
PlotAlignmentValidation::~PlotAlignmentValidation()
{
  delete sourcelist;

  for(std::vector<TkOfflineVariables*>::iterator it = sourceList.begin();
      it != sourceList.end(); ++it){
    delete (*it);
  }

}

//------------------------------------------------------------------------------
void PlotAlignmentValidation::loadFileList(const char *inputFile, std::string legendName, int lineColor, int lineStyle)
{

  sourceList.push_back( new TkOfflineVariables( inputFile, treeBaseDir, legendName, lineColor, lineStyle ) );
  
}

//------------------------------------------------------------------------------
void PlotAlignmentValidation::useFitForDMRplots(bool usefit)
{

  useFit_ = usefit;
  
}

//------------------------------------------------------------------------------
void PlotAlignmentValidation::setOutputDir( std::string dir )
{
  // we should check if this dir exsits...
  std::cout <<"'"<< outputDir <<"' = "<< dir << std::endl;
  outputDir = dir;
}

//------------------------------------------------------------------------------
void PlotAlignmentValidation::plotSubDetResiduals(bool plotNormHisto,unsigned int subDetId)
{
  setNiceStyle();
 
  gStyle->SetOptStat(11111);
  gStyle->SetOptFit(0000);

  TCanvas *c = new TCanvas("c", "c", 600,600);
  c->SetTopMargin(0.15);
  TString histoName= "";
  if (plotNormHisto) {histoName= "h_NormXprime";}
  else histoName= "h_Xprime_";
  switch (subDetId){
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
  int tmpcounter = 0;
  TH1 *sumHisto = 0;
  for(std::vector<TkOfflineVariables*>::iterator it = sourceList.begin();
      it != sourceList.end(); ++it) {
    if (tmpcounter == 0 ) {
      TFile *f= (*it)->getFile();
      sumHisto =(TH1*) f->FindKeyAny(histoName)->ReadObj();//FindObjectAny(histoName.Data());
      sumHisto->SetLineColor(tmpcounter+1);
      sumHisto->SetLineStyle(tmpcounter+1);
      sumHisto->GetFunction("tmp")->SetBit(TF1::kNotDraw);
      sumHisto->Draw();
      
      //get statistic box coordinate to plot all boxes one below the other
      //gStyle->SetStatY(0.91);
      //gStyle->SetStatW(0.15);
      //gStyle->SetStatBorderSize(1);
      //gStyle->SetStatH(0.10);
      
      
      tmpcounter++;
    } else {
      sumHisto = (TH1*) (*it)->getFile()->FindObjectAny(histoName);
      sumHisto->SetLineColor(tmpcounter+1);
      sumHisto->SetLineStyle(tmpcounter+1);
      sumHisto->GetFunction("tmp")->SetBit(TF1::kNotDraw);
      //hstack->Add(sumHisto);
      
      c->Update();
      tmpcounter++;  
    }
    TObject *statObj = sumHisto->GetListOfFunctions()->FindObject("stats");
    if (statObj && statObj->InheritsFrom(TPaveStats::Class())) {
      TPaveStats *stats = static_cast<TPaveStats*>(statObj);
      stats->SetLineColor(tmpcounter+1);
      stats->SetTextColor(tmpcounter+1);
      stats->SetFillColor(10);
      stats->SetX1NDC(0.91-tmpcounter*0.1);
      stats->SetX2NDC(0.15);
      stats->SetY1NDC(1);
      stats->SetY2NDC(0.10);
      sumHisto->Draw("sames");
    }
  }
  //hstack->Draw("nostack");
  char PlotName[1000];
  sprintf( PlotName, "%s/%s.eps", outputDir.c_str(), histoName.Data() );
  
  c->Print(PlotName);
  //delete c;
  //c=0;
    
}

//------------------------------------------------------------------------------
void PlotAlignmentValidation::plotHitMaps()
{
  
  setNiceStyle(); 
  //gStyle->SetOptStat(0);
  
  TCanvas *c = new TCanvas("c", "c", 1200,400);
  setCanvasStyle( *c );
  c->Divide(3,1);
  //ps->NewPage();

  //-------------------------------------------------
  //plot Hit map
  //-------------------------------------------------
  std::string histName_="Entriesprofile";
  c->cd(1);
  TTree *tree= (*sourceList.begin())->getTree();
  tree->Draw("entries:posR:posZ","","COLZ2Prof");
  c->cd(2);
  tree->Draw("entries:posY:posX","","COLZ2Prof");
  c->cd(3);
  tree->Draw("entries:posR:posPhi","","COLZ2Prof");
  
  char PlotName[1000];
  sprintf( PlotName, "%s/%s.eps", outputDir.c_str(), histName_.c_str() );
  
  c->Print(PlotName);
  //   //c->Update();
  c->Close();  
  //----------------------------------------------------
  
}

//------------------------------------------------------------------------------
void PlotAlignmentValidation::plotOutlierModules(const char *outputFileName, std::string plotVariable,
						 float plotVariable_cut ,int unsigned minHits)
{
 
  Int_t counter=0;
  setNiceStyle();
  
  gStyle->SetOptStat(111111);
  gStyle->SetStatY(0.9);
  //TList treelist=getTreeList();
  
  TCanvas *c1 = new TCanvas("canv", "canv", 800, 500);
  //setCanvasStyle( *c1 );
  outputFile = outputDir +'/'+ outputFileName;   
  c1->Print( (outputFile+'[').Data() ); 
  
  
  c1->Divide(2,1);
  
  TTree *tree= (*sourceList.begin())->getTree();
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
//   cout<<"treeMem->entries  "<<treeMem->entries<<endl;  
//  cout<<"var                  "<<var<<endl;
//  cout<<"plotVariable_cut     "<<plotVariable_cut<<endl;
    
    if (var > plotVariable_cut && treeMem->entries > minHits)
      {
	
	TFile *f=(*sourceList.begin())->getFile();//(TFile*)sourcelist->First();
	
	if(f->FindKeyAny(treeMem->histNameX.c_str())!=0){
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
	else{
	  cout<<"There are no residual histograms on module level stored!"<<endl;
	  cout<<"Please make sure that moduleLevelHistsTransient = cms.bool(False) in the validation job!"<<endl;
	  break;
	}
      }
    
  }
  c1->Print( (outputFile+"]").Data() );
  if (counter == 0) cout<<"no bad modules found"<<endl;
  
  
  //read the number of entries in the t3
  //TTree* tree=0;
  //tree=(TTree*)treeList->At(0);
  
  
  //c1->Close();
  
}

//------------------------------------------------------------------------------
TList PlotAlignmentValidation::getTreeList()
{
  TList treeList = new TList();
  TFile *first_source = (TFile*)sourcelist->First();
  std::cout<<first_source->GetName()<<std::endl;
  TDirectoryFile *d=(TDirectoryFile*)first_source->Get( treeBaseDir.c_str() ); 
  treeList.Add( (TTree*)(*d).Get("TkOffVal") );
  
  if( moreThanOneSource ==true ){
    TFile *nextsource = (TFile*)sourcelist->After( first_source );
    while ( nextsource ) {
      std::cout<<nextsource->GetName()<<std::endl;
      d=(TDirectoryFile*)nextsource->Get("TrackerOfflineValidation"); 
      
      treeList.Add((TTree*)(*d).Get("TkOffVal"));
      
      nextsource = (TFile*)sourcelist->After( nextsource );
    }
  }
  return treeList;
}

//------------------------------------------------------------------------------
void PlotAlignmentValidation::setTreeBaseDir( std::string dir )
{
  treeBaseDir = dir;  
}

//------------------------------------------------------------------------------
void  PlotAlignmentValidation::plotDMR(const std::string& variable, Int_t minHits, const std::string& options)
{

  // Variable name should end with X or Y. If it doesn't, recursively calls plotDMR twice with
  // X and Y added, respectively
  if (variable == "mean" || variable == "median" || variable == "meanNorm" ||
      variable == "rms" || variable == "rmsNorm") {
    plotDMR(variable+"X", minHits, options);
    plotDMR(variable+"Y", minHits, options);
    return;
  }

  bool plotPlain = false, plotSplits = false;
  if (options.find("plain") != std::string::npos) { plotPlain = true; }
  if (options.find("split") != std::string::npos) { plotSplits = true; }
  // Defaults to plotting only plain plot if empty (or invalid)
  // option string is given
  if (!plotPlain && !plotSplits) { plotPlain = true; }

  // This boolean array tells for which detector modules to plot split DMR plots
  // They are plotted for BPIX, FPIX, TIB and TOB
  static bool plotSplitsFor[6] = { true, true, true, false, true, false };

  DMRPlotInfo plotinfo;

  setNiceStyle(); 
  gStyle->SetOptStat(0);
  
  TCanvas c("canv", "canv", 600, 600);
  setCanvasStyle( c );

  plotinfo.variable = variable;
  plotinfo.minHits = minHits;
  plotinfo.plotPlain = plotPlain;

  if (variable == "meanX") {          plotinfo.nbins = 50;  plotinfo.min = -0.001; plotinfo.max = 0.001; }
  else if (variable == "meanY") {     plotinfo.nbins = 50;  plotinfo.min = -0.005; plotinfo.max = 0.005; }
  else if (variable == "medianX") {   plotinfo.nbins = 50;  plotinfo.min = -0.005; plotinfo.max = 0.005; }
  else if (variable == "medianY") {   plotinfo.nbins = 50;  plotinfo.min = -0.005; plotinfo.max = 0.005; }
  else if (variable == "meanNormX") { plotinfo.nbins = 100; plotinfo.min = -2.0;   plotinfo.max = 2.0; }
  else if (variable == "meanNormY") { plotinfo.nbins = 100; plotinfo.min = -2.0;   plotinfo.max = 2.0; }
  else if (variable == "rmsX") {      plotinfo.nbins = 100; plotinfo.min = 0.0;    plotinfo.max = 0.1; }
  else if (variable == "rmsY") {      plotinfo.nbins = 100; plotinfo.min = 0.0;    plotinfo.max = 0.1; }
  else if (variable == "rmsNormX") {      plotinfo.nbins = 100; plotinfo.min = 0.3;    plotinfo.max = 1.8; }
  else if (variable == "rmsNormY") {      plotinfo.nbins = 100; plotinfo.min = 0.3;    plotinfo.max = 1.8; }
  else {
    std::cerr << "Unknown variable " << variable << std::endl;
    plotinfo.nbins = 100; plotinfo.min = -0.1; plotinfo.max = 0.1;
  }

  for (int i=1; i<=6; ++i) {

    // Skip strip detectors if plotting any "Y" variable
    if (i != 1 && i != 2 && variable.length() > 0 && variable[variable.length()-1] == 'Y') {
      continue;
    }
 
    plotinfo.plotSplits = plotSplits && plotSplitsFor[i-1];
    if (!plotinfo.plotPlain && !plotinfo.plotSplits) {
      continue;
    }

    THStack hstack("hstack", "hstack");
    plotinfo.maxY = 0;
    plotinfo.subDetId = i;
    plotinfo.legend = new TLegend(0.17, 0.8, 0.85, 0.88);
    setLegendStyle(*plotinfo.legend);
    plotinfo.hstack = &hstack;
    plotinfo.h = plotinfo.h1 = plotinfo.h2 = 0;
    plotinfo.firsthisto = true;
    
    for(std::vector<TkOfflineVariables*>::iterator it = sourceList.begin();
	it != sourceList.end(); ++it){

      plotinfo.vars = *it;

      if (plotinfo.plotPlain) {
	plotDMRHistogram(plotinfo, 0);
      }

      if (plotinfo.plotSplits) {
	plotDMRHistogram(plotinfo, -1);
	plotDMRHistogram(plotinfo, 1);
      }

      if (plotinfo.plotPlain) {
	if (plotinfo.h) { setDMRHistStyleAndLegend(plotinfo.h, plotinfo, 0); }
      }

      if (plotinfo.plotSplits) {
	// Add delta mu to the histogram
	if (plotinfo.h1 != 0 && plotinfo.h2 != 0 && !plotinfo.plotPlain) {
	  std::ostringstream legend;
	  std::string unit = " #mum";
	  legend.precision(2);
	  float factor = 10000.0f;
	  if (plotinfo.variable == "meanNormX" || plotinfo.variable == "meanNormY" ||
	      plotinfo.variable == "rmsNormX" || plotinfo.variable == "rmsNormY") {
	    factor = 1.0f;
	    unit = "";
	  }
	  float deltamu = factor*(plotinfo.h2->GetMean(1) - plotinfo.h1->GetMean(1));
	  legend << plotinfo.vars->getName() << ": #Delta#mu = " << deltamu << unit;
	  plotinfo.legend->AddEntry(static_cast<TObject*>(0), legend.str().c_str(), ""); 
	}
	if (plotinfo.h1) { setDMRHistStyleAndLegend(plotinfo.h1, plotinfo, -1); }
	if (plotinfo.h2) { setDMRHistStyleAndLegend(plotinfo.h2, plotinfo, 1); }
      }
      
    }
    
    if (plotinfo.h != 0 || plotinfo.h1 != 0 || plotinfo.h2 != 0) {

      hstack.Draw("nostack");
      hstack.SetMaximum(plotinfo.maxY*1.3);
      setTitleStyle(hstack, variable.c_str(), "#modules", plotinfo.subDetId);
      setHistStyle(*hstack.GetHistogram(), variable.c_str(), "#modules", 1);

      plotinfo.legend->Draw(); 
 
      std::ostringstream plotName;
      plotName << outputDir << "/D";
     
      if (variable=="medianX") plotName << "medianR_";
      else if (variable=="medianY") plotName << "medianYR_";
      else if (variable=="meanX") plotName << "meanR_";
      else if (variable=="meanY") plotName << "meanYR_";
      else if (variable=="meanNormX") plotName << "meanNR_";
      else if (variable=="meanNormY") plotName << "meanNYR_";
      else if (variable=="rmsX") plotName << "rmsR_";
      else if (variable=="rmsY") plotName << "rmsYR_";
      else if (variable=="rmsNormX") plotName << "rmsNR_";
      else if (variable=="rmsNormY") plotName << "rmsNYR_";

      switch (i) {
      case 1: plotName << "TPB"; break;
      case 2: plotName << "TPE"; break;
      case 3: plotName << "TIB"; break;
      case 4: plotName << "TID"; break;
      case 5: plotName << "TOB"; break;
      case 6: plotName << "TEC"; break;
      }

      if (plotPlain && !plotSplits) { plotName << "_plain"; }
      else if (!plotPlain && plotSplits) { plotName << "_split"; }
 
      plotName << ".eps";

      c.Update(); 
      c.Print(plotName.str().c_str());
      
    }
    
  }

}

//------------------------------------------------------------------------------
TH1* PlotAlignmentValidation::addHists(const char *selection, const TString &residType,
				       bool printModuleIds)
{
  enum ResidType {
    xPrimeRes, yPrimeRes, xPrimeNormRes, yPrimeNormRes, xRes, yRes, xNormRes, /*yResNorm*/
    ResXvsXProfile,  ResXvsYProfile, ResYvsXProfile, ResYvsYProfile
  };
  ResidType rType = xPrimeRes;
  if (residType == "xPrime") rType = xPrimeRes;
  else if (residType == "yPrime") rType = yPrimeRes;
  else if (residType == "xPrimeNorm") rType = xPrimeNormRes;
  else if (residType == "yPrimeNorm") rType = yPrimeNormRes;
  else if (residType == "x") rType = xRes;
  else if (residType == "y") rType = yRes;
  else if (residType == "xNorm") rType = xNormRes;
  // else if (residType == "yNorm") rType = yResNorm;
  else if (residType == "ResXvsXProfile") rType = ResXvsXProfile;
  else if (residType == "ResYvsXProfile") rType = ResYvsXProfile;
  else if (residType == "ResXvsYProfile") rType = ResXvsYProfile;
  else if (residType == "ResYvsYProfile") rType = ResYvsYProfile;
  else {
    std::cout << "PlotAlignmentValidation::addHists: Unknown residual type "
	      << residType << std::endl; 
    return 0;
  }

  TFile *f = (*sourceList.begin())->getFile();
  TTree *tree= (*sourceList.begin())->getTree();
  if (!f || !tree) {
    std::cout << "PlotAlignmentValidation::addHists: no tree or no file" << std::endl;
    return 0;
  }
  
  // first loop on tree to find out which entries (i.e. modules) fulfill the selection
  // 'Entry$' gives the entry number in the tree
  Long64_t nSel = tree->Draw("Entry$", selection, "goff");
  if (nSel == -1) return 0; // error in selection
  if (nSel == 0) {
    std::cout << "PlotAlignmentValidation::addHists: no selected module." << std::endl;
    return 0;
  }
  // copy entry numbers that fulfil the selection
  const std::vector<double> selected(tree->GetV1(), tree->GetV1() + nSel);

  TH1 *h = 0;       // becomes result
  UInt_t nEmpty = 0;// selected, but empty hists
  Long64_t nentries =  tree->GetEntriesFast();
  std::vector<double>::const_iterator iterEnt = selected.begin();

  // second loop on tree:
  // for each selected entry get the hist from the file and merge
  TkOffTreeVariables *treeMem = 0; // ROOT will initialise
  tree->SetBranchAddress("TkOffTreeVariables", &treeMem);
  for (Long64_t i = 0; i < nentries; i++){
    if (i < *iterEnt - 0.1             // smaller index (with tolerance): skip
	|| iterEnt == selected.end()) { // at the end: skip 
      continue;
    } else if (TMath::Abs(i - *iterEnt) < 0.11) {
      ++iterEnt; // take this entry!
    } else std::cout << "Must not happen: " << i << " " << *iterEnt << std::endl;

    tree->GetEntry(i);
    if (printModuleIds) {
      std::cout << treeMem->moduleId << ": " << treeMem->entries << " entries" << std::endl;
    }
    if (treeMem->entries <= 0) {  // little speed up: skip empty hists
      ++nEmpty;
      continue;
    }
    TString hName;
    switch(rType) {
    case xPrimeRes:     hName = treeMem->histNameX.c_str();          break;
    case yPrimeRes:     hName = treeMem->histNameY.c_str();          break;
    case xPrimeNormRes: hName = treeMem->histNameNormX.c_str();      break;
    case yPrimeNormRes: hName = treeMem->histNameNormY.c_str();      break;
    case xRes:          hName = treeMem->histNameLocalX.c_str();     break;
    case yRes:          hName = treeMem->histNameLocalY.c_str();     break;
    case xNormRes:      hName = treeMem->histNameNormLocalX.c_str(); break;
      /*case yResNorm:      hName = treeMem->histNameNormLocalY.c_str(); break;*/
    case ResXvsXProfile: hName = treeMem->profileNameResXvsX.c_str();    break;
    case ResXvsYProfile: hName = treeMem->profileNameResXvsY.c_str();    break;
    case ResYvsXProfile: hName = treeMem->profileNameResYvsX.c_str();    break;
    case ResYvsYProfile: hName = treeMem->profileNameResYvsY.c_str();    break;
   }
    TH1 *newHist = static_cast<TH1*>(f->FindKeyAny(hName)->ReadObj());
    if (!newHist) {
      std::cout << "Hist " << hName << " not found in file, break loop." << std::endl;
      break;
    }
    if (!h) { // first hist: clone, but rename keeping only first part of name
      TString name(newHist->GetName());
      Ssiz_t pos_ = 0;
      for (UInt_t i2 = 0; i2 < 3; ++i2) pos_ = name.Index("_", pos_+1);
      name = name(0, pos_); // only up to three '_'
	h = static_cast<TH1*>(newHist->Clone("summed_"+name));
	h->SetTitle(Form("%s: %lld modules", selection, nSel));
    } else { // otherwise just add
      h->Add(newHist);
    }
    delete newHist;
  }

  std::cout << "PlotAlignmentValidation::addHists" << "Result is merged from " << nSel-nEmpty
	    << " modules, " << nEmpty << " hists were empty." << std::endl;
  return h;
}

//------------------------------------------------------------------------------
std::pair<float,float> 
PlotAlignmentValidation::fitGauss(TH1 *hist,int color) 
{
  //1. fits a Gauss function to the inner range of abs(2 rms)
  //2. repeates the Gauss fit in a 2 sigma range around mean of first fit
  //returns mean and sigma from fit in micron
  std::pair<float,float> fitResult(9999., 9999.);
  if (!hist || hist->GetEntries() < 20) return fitResult;

  float mean  = hist->GetMean();
  float sigma = hist->GetRMS();

 
  TF1 func("tmp", "gaus", mean - 2.*sigma, mean + 2.*sigma); 
 
  func.SetLineColor(color);
  func.SetLineStyle(2);
  if (0 == hist->Fit(&func,"QNR")) { // N: do not blow up file by storing fit!
    mean  = func.GetParameter(1);
    sigma = func.GetParameter(2);
    // second fit: three sigma of first fit around mean of first fit
    func.SetRange(mean - 2.*sigma, mean + 2.*sigma);
    // I: integral gives more correct results if binning is too wide
    // L: Likelihood can treat empty bins correctly (if hist not weighted...)
    if (0 == hist->Fit(&func, "Q0ILR")) {
      if (hist->GetFunction(func.GetName())) { // Take care that it is later on drawn:
	//hist->GetFunction(func.GetName())->ResetBit(TF1::kNotDraw);
      }
      fitResult.first = func.GetParameter(1)*10000;//convert from cm to micron
      fitResult.second = func.GetParameter(2)*10000;//convert from cm to micron
    }
  }
 
  
  return fitResult;
}


//------------------------------------------------------------------------------
void  PlotAlignmentValidation::setCanvasStyle( TCanvas& canv )
{
  canv.SetFillStyle   ( 4000 );
  canv.SetLeftMargin  ( 0.15 );
  canv.SetRightMargin ( 0.05 );
  canv.SetBottomMargin( 0.15 );
  canv.SetTopMargin   ( 0.10 );
}

//------------------------------------------------------------------------------
void  PlotAlignmentValidation::setLegendStyle( TLegend& leg )
{
  leg.SetFillStyle ( 0 );
  leg.SetFillColor ( 0 );
  leg.SetBorderSize( 0 ); 
}

//------------------------------------------------------------------------------
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

  MyStyle->SetTitleOffset (1.5,      "X");
  MyStyle->SetTitleOffset (1.2,      "Y");
  MyStyle->SetTitleOffset (0.9,     "Z");

  MyStyle->SetTitleSize   (0.045,    "X");
  MyStyle->SetTitleSize   (0.045,    "Y");
  MyStyle->SetTitleSize   (0.045,    "Z");

  MyStyle->SetLabelFont   (132,      "X");
  MyStyle->SetLabelFont   (132,      "Y");
  MyStyle->SetLabelFont   (132,      "Z");

  MyStyle->SetPalette(1);

  MyStyle->cd();
}

//------------------------------------------------------------------------------
void  PlotAlignmentValidation::setTitleStyle( TNamed &hist,const char* titleX, const char* titleY,int subDetId)
{
  std::stringstream titel_Xaxis;
  std::stringstream titel_Yaxis;
  TString titelXAxis=titleX;
  TString titelYAxis=titleY;
  cout<<"plot "<<titelXAxis<<" vs "<<titelYAxis<<endl;
  
  if ( titelXAxis.Contains("medianX")||titelXAxis.Contains("medianY")||titelXAxis.Contains("meanX")||titelXAxis.Contains("rmsX")||titelXAxis.Contains("meanY") ){
    std::string histTitel="";
    if (titelXAxis.Contains("medianX")) histTitel="Distribution of the median of the residuals in ";
    if (titelXAxis.Contains("medianY")) histTitel="Distribution of the median of the y residuals in ";
    if (titelXAxis.Contains("meanX")) histTitel="Distribution of the mean of the residuals in ";
    if (titelXAxis.Contains("meanY")) histTitel="Distribution of the mean of the residuals in ";
    if (titelXAxis.Contains("rmsX")) histTitel="Distribution of the rms of the residuals in ";
    
    switch (subDetId) {
    case 1: histTitel+="TPB";break;
    case 2: histTitel+="TPE";break;
    case 3: histTitel+="TIB";break;
    case 4: histTitel+="TID";break;
    case 5: histTitel+="TOB";break;
    case 6: histTitel+="TEC";break;
    }
    hist.SetTitle(histTitel.c_str());
  } else {
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


//------------------------------------------------------------------------------
void  PlotAlignmentValidation::setHistStyle( TH1& hist,const char* titleX, const char* titleY, int color)
{
  std::stringstream titel_Xaxis;
  std::stringstream titel_Yaxis;
  TString titelXAxis=titleX;
  TString titelYAxis=titleY;
  
  if ( titelXAxis.Contains("Phi") )titel_Xaxis<<titleX<<"[rad]";
  else if( titelXAxis.Contains("meanX") )titel_Xaxis<<"#LTx'_{pred}-x'_{hit}#GT[cm]";
  else if( titelXAxis.Contains("meanY") )titel_Xaxis<<"#LTy'_{pred}-y'_{hit}#GT[cm]";
  else if( titelXAxis.Contains("rmsX") )titel_Xaxis<<"RMS(x'_{pred}-x'_{hit})[cm]";
  else if( titelXAxis.Contains("rmsY") )titel_Xaxis<<"RMS(y'_{pred}-y'_{hit})[cm]";
  else if( titelXAxis.Contains("meanNormX") )titel_Xaxis<<"#LTx'_{pred}-x'_{hit}/#sigma#GT";
  else if( titelXAxis.Contains("meanNormY") )titel_Xaxis<<"#LTy'_{pred}-y'_{hit}/#sigma#GT";
  else if( titelXAxis.Contains("rmsNormX") )titel_Xaxis<<"RMS(x'_{pred}-x'_{hit}/#sigma)";
  else if( titelXAxis.Contains("rmsNormY") )titel_Xaxis<<"RMS(y'_{pred}-y'_{hit}/#sigma)";
  else if( titelXAxis.Contains("meanLocalX") )titel_Xaxis<<"#LTx_{pred}-x_{hit}#GT[cm]";
  else if( titelXAxis.Contains("rmsLocalX") )titel_Xaxis<<"RMS(x_{pred}-x_{hit})[cm]";
  else if( titelXAxis.Contains("meanNormLocalX") )titel_Xaxis<<"#LTx_{pred}-x_{hit}/#sigma#GT[cm]";
  else if( titelXAxis.Contains("rmsNormLocalX") )titel_Xaxis<<"RMS(x_{pred}-x_{hit}/#sigma)[cm]";
  else if( titelXAxis.Contains("medianX") )titel_Xaxis<<"median(x'_{pred}-x'_{hit})[cm]";
  else if( titelXAxis.Contains("medianY") )titel_Xaxis<<"median(y'_{pred}-y'_{hit})[cm]";
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
  hist.GetXaxis()->SetTitleOffset(  1.2   );
  hist.GetXaxis()->SetTitleFont  (   62 );
  hist.GetXaxis()->SetLabelSize  ( 0.05 );
  hist.GetXaxis()->SetLabelFont  (   62 );
  //hist.GetXaxis()->CenterTitle   (      );
  hist.GetXaxis()->SetNdivisions (  505 );

  if /*( titelYAxis.Contains("meanX") )titel_Yaxis<<"#LTx'_{pred}-x'_{hit}#GT[cm]";
  else if ( titelYAxis.Contains("rmsX") )titel_Yaxis<<"RMS(x'_{pred}-x'_{hit})[cm]";
  else if( titelYAxis.Contains("meanNormX") )titel_Yaxis<<"#LTx'_{pred}-x'_{hit}/#sigma#GT";
  else if( titelYAxis.Contains("rmsNormX") )titel_Yaxis<<"RMS(x_'{pred}-x'_{hit}/#sigma)";
  else if( titelYAxis.Contains("meanLocalX") )titel_Yaxis<<"#LTx_{pred}-x_{hit}#GT[cm]";
  else if( titelYAxis.Contains("rmsLocalX") )titel_Yaxis<<"RMS(x_{pred}-x_{hit})[cm]";
  else if*/ ( (titelYAxis.Contains("layer") && titelYAxis.Contains("subDetId"))
	      || titelYAxis.Contains("#modules") )titel_Yaxis<<"#modules";
  else if ( (titelYAxis.Contains("ring") && titelYAxis.Contains("subDetId"))
	    || titelYAxis.Contains("#modules") )titel_Yaxis<<"#modules";
  else titel_Yaxis<<titleY<<"[cm]";

  hist.GetYaxis()->SetTitle( (titel_Yaxis.str()).c_str()  );
  //hist.SetMinimum(1);
  hist.GetYaxis()->SetTitleSize  ( 0.05 );
  hist.GetYaxis()->SetTitleColor (    1 );
  if ( hist.IsA()->InheritsFrom( TH2::Class() ) ) hist.GetYaxis()->SetTitleOffset( 0.95 );
  else hist.GetYaxis()->SetTitleOffset( 1.2 );
  hist.GetYaxis()->SetTitleFont  (   62 );
  hist.GetYaxis()->SetLabelSize  ( 0.03 );
  hist.GetYaxis()->SetLabelFont  (   62 );

}

//------------------------------------------------------------------------------

std::string PlotAlignmentValidation::
getSelectionForDMRPlot(int minHits, int subDetId, int direction)
{
  std::ostringstream builder;
  builder << "entries >= " << minHits;
  builder << " && subDetId == " << subDetId;
  if (direction != 0) {
    if (subDetId == 2) { // FPIX is split by zDirection
      builder << " && zDirection == " << direction;
    } else {
      builder << " && rDirection == " << direction;
    }
  }
  builder.flush();
  return builder.str();
}

std::string PlotAlignmentValidation::
getVariableForDMRPlot(const std::string& histoname, const std::string& variable, int nbins, double min,
		      double max)
{
  std::ostringstream builder;
  builder << variable << ">>" << histoname << "(" << nbins << "," << min <<
    "," << max << ")";
  builder.flush();
  return builder.str();
}

void PlotAlignmentValidation::
setDMRHistStyleAndLegend(TH1F* h, PlotAlignmentValidation::DMRPlotInfo& plotinfo, int direction)
{
  std::pair<float,float> fitResults(9999., 9999.);

  h->SetDirectory(0);

  // The whole DMR plot is plotted with wider line than the split plots
  // If only split plots are plotted, they will be stronger too, though
  h->SetLineWidth((direction == 0 || (plotinfo.plotSplits && !plotinfo.plotPlain)) ? 2 : 1);

  // These lines determine the style of the plots according to rules:
  // -If the plot is for direction != 0, +1 or +2 is added to the given style for distinction
  // -However if only direction split plots are to be plotted, the additions should be 0 and +1 respectively
  // -Modulo 4 arithmetic, because the styles run from 1..4
  int linestyle = plotinfo.vars->getLineStyle() - 1, linestyleplus = 0;
  if (direction == -1) { linestyleplus = 1; }
  if (direction == 1) { linestyleplus = 2; }
  if (direction != 0 && plotinfo.plotSplits && !plotinfo.plotPlain) { linestyleplus--; }
  linestyle = (linestyle + linestyleplus) % 4 + 1;

  if (plotinfo.firsthisto) {
    setHistStyle(*h, plotinfo.variable.c_str(), "#modules", 1); //set color later
    plotinfo.firsthisto = false;
  }

  h->SetLineColor( plotinfo.vars->getLineColor() );
  h->SetLineStyle( linestyle );
	  
  if (plotinfo.maxY<h->GetMaximum()){
    plotinfo.maxY=h->GetMaximum();
  }
	  
  //fit histogram for median and mean
  if (plotinfo.variable == "medianX" || plotinfo.variable == "meanX") {
    fitResults = fitGauss(h, plotinfo.vars->getLineColor() );
  }
	  
  plotinfo.hstack->Add(h);

  std::ostringstream legend;
  legend.precision(2);

  // Legend: header part
  if (direction == -1 && plotinfo.subDetId != 2) { legend << "rDirection < 0: "; }
  else if (direction == 1 && plotinfo.subDetId != 2) { legend << "rDirection > 0: "; }
  else if (direction == -1 && plotinfo.subDetId == 2) { legend << "zDirection < 0: "; }
  else if (direction == 1 && plotinfo.subDetId == 2) { legend << "zDirection > 0: "; }
  else { legend  << plotinfo.vars->getName() << ": "; }

  // Legend: Statistics
  if (plotinfo.variable == "medianX" || plotinfo.variable == "meanX" ||
      plotinfo.variable == "medianY" || plotinfo.variable == "meanY") {
    if (useFit_) {
      legend << "#mu = " << fitResults.first << " #mum, #sigma = " << fitResults.second << " #mum";
    } else {
      legend << "#mu = " << h->GetMean(1)*10000 << " #mum, rms = " << h->GetRMS(1)*10000 << " #mum";
    }
  } else if (plotinfo.variable == "rmsX" || plotinfo.variable == "rmsY") {
    legend << "#mu = " << h->GetMean(1)*10000 << " #mum, rms = " << h->GetRMS(1)*10000 << " #mum";
  } else if (plotinfo.variable == "meanNormX" || plotinfo.variable == "meanNormY" ||
	     plotinfo.variable == "rmsNormX" || plotinfo.variable == "rmsNormY") {
    legend << "#mu = " << h->GetMean(1) << ", rms = " << h->GetRMS(1);
  }

  // Legend: Delta mu for split plots
  if (plotinfo.h1 != 0 && plotinfo.h2 != 0 && plotinfo.plotSplits &&
      plotinfo.plotPlain && direction == 0) {
    std::string unit = " #mum";
    float factor = 10000.0f;
    if (plotinfo.variable == "meanNormX" || plotinfo.variable == "meanNormY" ||
	plotinfo.variable == "rmsNormX" || plotinfo.variable == "rmsNormY") {
      factor = 1.0f;
      unit = "";
    }
    float deltamu = factor*(plotinfo.h2->GetMean(1) - plotinfo.h1->GetMean(1));
    legend << ", #Delta#mu = " << deltamu << unit;
  }

  plotinfo.legend->AddEntry(h, legend.str().c_str(), "l");

}

void PlotAlignmentValidation::
plotDMRHistogram(PlotAlignmentValidation::DMRPlotInfo& plotinfo, int direction)
{
  TH1F* h = 0;
  std::string histoname;
  if (direction == -1) { histoname = "myhisto1"; }
  else if (direction == 1) { histoname = "myhisto2"; }
  else { histoname = "myhisto"; }
  std::string plotVariable = getVariableForDMRPlot(histoname, plotinfo.variable, plotinfo.nbins, plotinfo.min, plotinfo.max);
  std::string selection = getSelectionForDMRPlot(plotinfo.minHits, plotinfo.subDetId, direction);
  plotinfo.vars->getTree()->Draw(plotVariable.c_str(), selection.c_str(), "goff");
  if (gDirectory) gDirectory->GetObject(histoname.c_str(), h);
  if (h && h->GetEntries() > 0) {
    if (direction == -1) { plotinfo.h1 = h; }
    else if (direction == 1) { plotinfo.h2 = h; }
    else { plotinfo.h = h; }
  }
}
