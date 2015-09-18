#include <TStyle.h>
#include <TSystem.h>
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include "TTree.h"
#include "TString.h"
#include "TAxis.h"
#include "TGaxis.h"
#include "TProfile.h"
#include "TH2F.h"
#include "TROOT.h"
#include "TDirectory.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TDirectoryFile.h"
#include "TLegend.h"
#include "TLegendEntry.h"
#include "THStack.h"
#include <exception>
#include "TKey.h"
#include "TPad.h"
#include "TPaveText.h"
#include "TPaveStats.h"
#include "TF1.h"
#include "TRegexp.h"
#include "TLatex.h"

// This line works only if we have a CMSSW environment...
#include "Alignment/OfflineValidation/interface/TkOffTreeVariables.h"

#include "Alignment/OfflineValidation/macros/PlotAlignmentValidation.h"

//------------------------------------------------------------------------------
PlotAlignmentValidation::PlotAlignmentValidation(const char *inputFile,std::string legendName, int lineColor, int lineStyle)
{
  setOutputDir("$TMPDIR");
  setTreeBaseDir();
  sourcelist = NULL;
  
  loadFileList( inputFile, legendName, lineColor, lineStyle);
  moreThanOneSource=false;
  useFit_ = false;

  // Force ROOT to use scientific notation even with smaller datasets
  TGaxis::SetMaxDigits(4);
  // (This sets a static variable: correct in .eps images but must be set
  // again manually when viewing the .root files)

  // Make ROOT calculate histogram statistics using all data,
  // regardless of displayed range
  TH1::StatOverflows(kTRUE);
}

//------------------------------------------------------------------------------
PlotAlignmentValidation::~PlotAlignmentValidation()
{

  for(std::vector<TkOfflineVariables*>::iterator it = sourceList.begin();
      it != sourceList.end(); ++it){
    delete (*it);
  }

  delete sourcelist;
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
  sprintf( PlotName, "%s/%s.png", outputDir.c_str(), histoName.Data() );
  c->Print(PlotName);
  sprintf( PlotName, "%s/%s.eps", outputDir.c_str(), histoName.Data() );
  c->Print(PlotName);
  sprintf( PlotName, "%s/%s.pdf", outputDir.c_str(), histoName.Data() );
  c->Print(PlotName);
  sprintf( PlotName, "%s/%s.root", outputDir.c_str(), histoName.Data() );
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
  sprintf( PlotName, "%s/%s.png", outputDir.c_str(), histName_.c_str() );
  c->Print(PlotName);
  sprintf( PlotName, "%s/%s.eps", outputDir.c_str(), histName_.c_str() );
  c->Print(PlotName);
  sprintf( PlotName, "%s/%s.pdf", outputDir.c_str(), histName_.c_str() );
  c->Print(PlotName);
  sprintf( PlotName, "%s/%s.root", outputDir.c_str(), histName_.c_str() );
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
  //TList* treelist=getTreeList();
  
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
TList* PlotAlignmentValidation::getTreeList()
{
  TList *treeList = new TList();
  TFile *first_source = (TFile*)sourcelist->First();
  std::cout<<first_source->GetName()<<std::endl;
  TDirectoryFile *d=(TDirectoryFile*)first_source->Get( treeBaseDir.c_str() ); 
  treeList->Add( (TTree*)(*d).Get("TkOffVal") );
  
  if( moreThanOneSource ==true ){
    TFile *nextsource = (TFile*)sourcelist->After( first_source );
    while ( nextsource ) {
      std::cout<<nextsource->GetName()<<std::endl;
      d=(TDirectoryFile*)nextsource->Get("TrackerOfflineValidation"); 
      
      treeList->Add((TTree*)(*d).Get("TkOffVal"));
      
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
void PlotAlignmentValidation::plotSurfaceShapes( const std::string& options, const std::string& residType )
{
  cout << "-------- plotSurfaceShapes called with " << options << endl;
  if (options == "none")
    return;
  else if (options == "coarse"){
    plotSS("subdet=1");
    plotSS("subdet=2");
    plotSS("subdet=3");
    plotSS("subdet=4");
    plotSS("subdet=5");
    plotSS("subdet=6");
  }
  // else if (options == "fine") ...
  else 
    plotSS( options, residType );

  return;
}

//------------------------------------------------------------------------------
void PlotAlignmentValidation::plotSS( const std::string& options, const std::string& residType )
{
  if (residType == "") {
    plotSS( options, "ResXvsXProfile");
    plotSS( options, "ResXvsYProfile");
    return;
  }

  int plotLayerN = 0;
  //  int plotRingN  = 0;
  //  bool plotPlain = false;
  bool plotLayers = false;  // overrides plotLayerN
  //  bool plotRings  = false;  // Todo: implement this?
  bool plotSplits = false;
  int plotSubDetN = 0;     // if zero, plot all

  TRegexp layer_re("layer=[0-9]+");
  Ssiz_t index, len;
  if (options.find("split") != std::string::npos) { plotSplits = true; }
  if (options.find("layers") != std::string::npos) { plotLayers = true; }
  if ((index = layer_re.Index(options, &len)) != -1) {
    if (plotLayers) {
      std::cerr << "Warning: option 'layers' overrides 'layer=N'" << std::endl;
    } else {
      std::string substr = options.substr(index+6, len-6);
      plotLayerN = atoi(substr.c_str());
    }
  }

  TRegexp subdet_re("subdet=[1-6]+");
  if ((index = subdet_re.Index(options, &len)) != -1) {
    std::string substr = options.substr(index+7, len-7);
    plotSubDetN = atoi(substr.c_str());
  }

  // If layers are plotted, these are the numbers of layers for each subdetector
  static int numberOfLayers[6] = { 3, 2, 4, 3, 6, 9 };

  setNiceStyle(); 
  gStyle->SetOptStat(0);
  
  TCanvas c("canv", "canv", 600, 600);
  setCanvasStyle( c );

  // todo: title, min/max, nbins?

  // Loop over detectors
  for (int iSubDet=1; iSubDet<=6; ++iSubDet) {

    // TEC requires special care since rings 1-4 and 5-7 are plotted separately
    bool isTEC = (iSubDet==6);

    // if subdet is specified, skip other subdets
    if (plotSubDetN!=0 && iSubDet!=plotSubDetN)
      continue;

    // Skips plotting too high layers
    if (plotLayerN > numberOfLayers[iSubDet-1]) {
      continue;
    }

    int minlayer = plotLayers ? 1 : plotLayerN;
    int maxlayer = plotLayers ? numberOfLayers[iSubDet-1] : plotLayerN;
    
    for (int layer = minlayer; layer <= maxlayer; layer++) {

      // two plots for TEC, skip first 
      for (int iTEC = 0; iTEC<2; iTEC++) {
	if (!isTEC && iTEC==0) continue;
	
	char  selection[1000];
	if (!isTEC){
	  if (layer==0)
	    sprintf(selection,"subDetId==%d",iSubDet); 
	  else
	    sprintf(selection,"subDetId==%d && layer == %d",iSubDet,layer); 
	}
	else{	  // TEC
	  if (iTEC==0)  // rings 
	    sprintf(selection,"subDetId==%d && ring <= 4",iSubDet); 
	  else
	    sprintf(selection,"subDetId==%d && ring > 4",iSubDet); 
	}


	// Title for plot and name for the file

	TString subDetName;
	switch (iSubDet) {
	case 1: subDetName = "BPIX"; break;
	case 2: subDetName = "FPIX"; break;
	case 3: subDetName = "TIB"; break;
	case 4: subDetName = "TID"; break;
	case 5: subDetName = "TOB"; break;
	case 6: subDetName = "TEC"; break;
	}

	TString myTitle = "Surface Shape, ";
	myTitle += subDetName;
	if (layer!=0) {
	  // TEC and TID have discs, the rest have layers
	  if (iSubDet==4 || iSubDet==6)
	    myTitle += TString(", disc ");
	  else {
	    myTitle += TString(", layer ");
	  }
	  myTitle += Form("%d",layer); 
	}
	if (isTEC && iTEC==0)
	  myTitle += TString(" R1-4");
	if (isTEC && iTEC>0)
	  myTitle += TString(" R5-7");
	
	// Generate histograms with selection
	TLegend* legend = 0;
	THStack *hs = addHists(selection, residType, &legend);
	if (!hs || hs->GetHists()==0 || hs->GetHists()->GetSize()==0) {
	  std::cout << "No histogram for " << subDetName <<
	               ", perhaps not enough data? Creating default histogram." << std::endl;
	  if(hs == 0)
	    hs = new THStack("hstack", "");

	  TProfile* defhist = new TProfile("defhist", "Empty default histogram", 100, -1, 1, -1, 1);
	  hs->Add(defhist);
	  hs->SetTitle( myTitle );
	  hs->Draw();
	}
	else {
	  hs->SetTitle( myTitle );
	  hs->Draw("nostack PE");
	  modifySSHistAndLegend(hs, legend);
	  legend->Draw();

	  // Adjust Labels
	  TH1* firstHisto = (TH1*) hs->GetHists()->First();
	  TString xName = firstHisto->GetXaxis()->GetTitle();
	  TString yName = firstHisto->GetYaxis()->GetTitle();
	  hs->GetHistogram()->GetXaxis()->SetTitleColor( kBlack ); 
	  hs->GetHistogram()->GetXaxis()->SetTitle( xName ); 
	  hs->GetHistogram()->GetYaxis()->SetTitleColor( kBlack );
	  // micrometers:
	  yName.ReplaceAll("cm", "#mum");
	  hs->GetHistogram()->GetYaxis()->SetTitle( yName ); 
	}

	// Save plot to file
	std::ostringstream plotName;
	plotName << outputDir << "/SurfaceShape_" << subDetName << "_";
	plotName << residType; 
	if (layer!=0) {
	  plotName << "_";
	  // TEC and TID have discs, the rest have layers
	  if (iSubDet==4 || iSubDet==6)
	    plotName << "disc";
	  else {
	    plotName << "layer";
	  }
	  plotName << layer;
	}
	if (isTEC && iTEC==0)
	  plotName << "_" << "R1-4";
	if (isTEC && iTEC>0)
	  plotName << "_" << "R5-7";

	// PNG,EPS,PDF files
	c.Update();
	c.Print((plotName.str() + ".png").c_str());
	c.Print((plotName.str() + ".eps").c_str());
	c.Print((plotName.str() + ".pdf").c_str());

	// ROOT file
	TFile f((plotName.str() + ".root").c_str(), "recreate");
	c.Write();
	f.Close();

	delete legend;
	delete hs;
      }
    }
  }

  return;
}


//------------------------------------------------------------------------------
void PlotAlignmentValidation::plotDMR(const std::string& variable, Int_t minHits, const std::string& options)
{
  // If several, comma-separated values are given in 'variable',
  // call plotDMR with each value separately.
  // If a comma is found, the string is divided to two.
  // (no space allowed)
  std::size_t findres = variable.find(",");
  if ( findres != std::string::npos) {
    std::string substring1 = variable.substr(0,         findres);
    std::string substring2 = variable.substr(findres+1, std::string::npos);
    plotDMR(substring1, minHits, options);
    plotDMR(substring2, minHits, options);
    return;
   }

  // Variable name should end with X or Y. If it doesn't, recursively calls plotDMR twice with
  // X and Y added, respectively
  if (variable == "mean" || variable == "median" || variable == "meanNorm" ||
      variable == "rms" || variable == "rmsNorm") {
    plotDMR(variable+"X", minHits, options);
    plotDMR(variable+"Y", minHits, options);
    return;
  }

  // options: 
  // -plain (default, the whole distribution)
  // -split (distribution splitted to two)
  // -layers (plain db for each layer/disc superimposed in one plot)
  // -layersSeparate (plain db for each layer/disc in separate plots)
  // -layersSplit (splitted db for each layers/disc in one plot)
  // -layersSplitSeparate (splitted db, for each layers/disc in separate plots)

  TRegexp layer_re("layer=[0-9]+");
  bool plotPlain = false, plotSplits = false, plotLayers = false;
  int plotLayerN = 0;
  Ssiz_t index, len;
  if (options.find("plain") != std::string::npos) { plotPlain = true; }
  if (options.find("split") != std::string::npos) { plotSplits = true; }
  if (options.find("layers") != std::string::npos) { plotLayers = true; }
  if ((index = layer_re.Index(options, &len)) != -1) {
    if (plotLayers) {
      std::cerr << "Warning: option 'layers' overrides 'layer=N'" << std::endl;
    } else {
      std::string substr = options.substr(index+6, len-6);
      plotLayerN = atoi(substr.c_str());
    }
  }

  // Defaults to plotting only plain plot if empty (or invalid)
  // option string is given
  if (!plotPlain && !plotSplits) { plotPlain = true; }

  // This boolean array tells for which detector modules to plot split DMR plots
  // They are plotted for BPIX, FPIX, TIB and TOB
  static bool plotSplitsFor[6] = { true, true, true, false, true, false };

  // If layers are plotted, these are the numbers of layers for each subdetector
  static int numberOfLayers[6] = { 3, 2, 4, 3, 6, 9 };

  DMRPlotInfo plotinfo;

  setNiceStyle(); 
  gStyle->SetOptStat(0);
  
  TCanvas c("canv", "canv", 600, 600);
  setCanvasStyle( c );

  plotinfo.variable = variable;
  plotinfo.minHits = minHits;
  plotinfo.plotPlain = plotPlain;
  plotinfo.plotLayers = plotLayers;

  // width in cm
  // for DMRS, use 100 bins in range +-10 um, bin width 0.2um
  // if modified, check also TrackerOfflineValidationSummary_cfi.py and TrackerOfflineValidation_Standalone_cff.py
  if (variable == "meanX") {          plotinfo.nbins = 50;  plotinfo.min = -0.001; plotinfo.max = 0.001; }
  else if (variable == "meanY") {     plotinfo.nbins = 50;  plotinfo.min = -0.005; plotinfo.max = 0.005; }
  else if (variable == "medianX")
    if (plotSplits) {                 plotinfo.nbins = 50;  plotinfo.min = -0.0005; plotinfo.max = 0.0005;}
    else {                            plotinfo.nbins = 100;  plotinfo.min = -0.001; plotinfo.max = 0.001; }
  else if (variable == "medianY")
    if (plotSplits) {                 plotinfo.nbins = 50;  plotinfo.min = -0.0005; plotinfo.max = 0.0005;}
    else {                            plotinfo.nbins = 100;  plotinfo.min = -0.001; plotinfo.max = 0.001; }
  else if (variable == "meanNormX") { plotinfo.nbins = 100; plotinfo.min = -2.0;   plotinfo.max = 2.0; }
  else if (variable == "meanNormY") { plotinfo.nbins = 100; plotinfo.min = -2.0;   plotinfo.max = 2.0; }
  else if (variable == "rmsX") {      plotinfo.nbins = 100; plotinfo.min = 0.0;    plotinfo.max = 0.1; }
  else if (variable == "rmsY") {      plotinfo.nbins = 100; plotinfo.min = 0.0;    plotinfo.max = 0.1; }
  else if (variable == "rmsNormX") {  plotinfo.nbins = 100; plotinfo.min = 0.3;    plotinfo.max = 1.8; }
  else if (variable == "rmsNormY") {  plotinfo.nbins = 100; plotinfo.min = 0.3;    plotinfo.max = 1.8; }
  else {
    std::cerr << "Unknown variable " << variable << std::endl;
    plotinfo.nbins = 100; plotinfo.min = -0.1; plotinfo.max = 0.1;
  }

  for (int i=1; i<=6; ++i) {

    // Skip strip detectors if plotting any "Y" variable
    if (i != 1 && i != 2 && variable.length() > 0 && variable[variable.length()-1] == 'Y') {
      continue;
    }
 
    // Skips plotting too high layers
    if (plotLayerN > numberOfLayers[i-1]) {
      continue;
    }

    plotinfo.plotSplits = plotSplits && plotSplitsFor[i-1];
    if (!plotinfo.plotPlain && !plotinfo.plotSplits) {
      continue;
    }

    // Sets dimension of legend according to the number of plots

    int nPlots = 1;
    if (plotinfo.plotSplits) { nPlots = 3; }
    if (plotinfo.plotLayers) { nPlots *= numberOfLayers[i-1]; }
    nPlots *= sourceList.size();

    double legendY = 0.80;
    if (nPlots > 3) { legendY -= 0.01 * (nPlots - 3); }
    if (legendY < 0.6) {
      std::cerr << "Warning: Huge legend!" << std::endl;
      legendY = 0.6;
    }

    THStack hstack("hstack", "hstack");
    plotinfo.maxY = 0;
    plotinfo.subDetId = i;
    plotinfo.nLayers = numberOfLayers[i-1];
    plotinfo.legend = new TLegend(0.17, legendY, 0.85, 0.88);
    setLegendStyle(*plotinfo.legend);
    plotinfo.hstack = &hstack;
    plotinfo.h = plotinfo.h1 = plotinfo.h2 = 0;
    plotinfo.firsthisto = true;
    
    for(std::vector<TkOfflineVariables*>::iterator it = sourceList.begin();
	it != sourceList.end(); ++it) {

      int minlayer = plotLayers ? 1 : plotLayerN;
      int maxlayer = plotLayers ? plotinfo.nLayers : plotLayerN;

      plotinfo.vars = *it;
      plotinfo.h1 = plotinfo.h2 = plotinfo.h = 0;

      for (int layer = minlayer; layer <= maxlayer; layer++) {

	if (plotinfo.plotPlain) {
	  plotDMRHistogram(plotinfo, 0, layer);
	}

	if (plotinfo.plotSplits) {
	  plotDMRHistogram(plotinfo, -1, layer);
	  plotDMRHistogram(plotinfo, 1, layer);
	}

	if (plotinfo.plotPlain) {
	  if (plotinfo.h) { setDMRHistStyleAndLegend(plotinfo.h, plotinfo, 0, layer); }
	}

	if (plotinfo.plotSplits) {
	  // Add delta mu to the histogram
	  if (plotinfo.h1 != 0 && plotinfo.h2 != 0 && !plotinfo.plotPlain) {
	    std::ostringstream legend;
	    std::string unit = " #mum";
	    legend.precision(3);
	    legend << fixed; // to always show 3 decimals
	    float factor = 10000.0f;
	    if (plotinfo.variable == "meanNormX" || plotinfo.variable == "meanNormY" ||
		plotinfo.variable == "rmsNormX" || plotinfo.variable == "rmsNormY") {
	      factor = 1.0f;
	      unit = "";
	    }
	    float deltamu = factor*(plotinfo.h2->GetMean(1) - plotinfo.h1->GetMean(1));
	    legend << plotinfo.vars->getName();
	    if (layer > 0) {
	      // TEC and TID have discs, the rest have layers
	      if (i==4 || i==6)
	        legend << ", disc ";
	      else
	        legend << ", layer ";
	      legend << layer;
	    }
	    legend << ": #Delta#mu = " << deltamu << unit;
	    plotinfo.legend->AddEntry(static_cast<TObject*>(0), legend.str().c_str(), ""); 
	  }
	  if (plotinfo.h1) { setDMRHistStyleAndLegend(plotinfo.h1, plotinfo, -1, layer); }
	  if (plotinfo.h2) { setDMRHistStyleAndLegend(plotinfo.h2, plotinfo, 1, layer); }
	}

      }

    }

    if (hstack.GetHists()!=0 && hstack.GetHists()->GetSize()!=0) {
      hstack.Draw("nostack");
      hstack.SetMaximum(plotinfo.maxY*1.3);
      setTitleStyle(hstack, variable.c_str(), "#modules", plotinfo.subDetId);
      setHistStyle(*hstack.GetHistogram(), variable.c_str(), "#modules", 1);

      plotinfo.legend->Draw(); 
    }
    else {
      // Draw an empty default histogram
      plotinfo.h = new TH1F("defhist", "Empty default histogram", plotinfo.nbins, plotinfo.min, plotinfo.max);
      plotinfo.h->SetMaximum(10);
      if (plotinfo.variable.find("Norm") == std::string::npos)
        scaleXaxis(plotinfo.h, 10000);
      setTitleStyle(*plotinfo.h, variable.c_str(), "#modules", plotinfo.subDetId);
      setHistStyle(*plotinfo.h, variable.c_str(), "#modules", 1);
      plotinfo.h->Draw();
    }

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
    case 1: plotName << "BPIX"; break;
    case 2: plotName << "FPIX"; break;
    case 3: plotName << "TIB"; break;
    case 4: plotName << "TID"; break;
    case 5: plotName << "TOB"; break;
    case 6: plotName << "TEC"; break;
    }

    if (plotPlain && !plotSplits) { plotName << "_plain"; }
    else if (!plotPlain && plotSplits) { plotName << "_split"; }
    if (plotLayers) {
      // TEC and TID have discs, the rest have layers
      if (i==4 || i==6)
        plotName << "_discs";
      else
        plotName << "_layers";
    }
    if (plotLayerN > 0) {
      // TEC and TID have discs, the rest have layers
      if (i==4 || i==6)
        plotName << "_disc";
      else
        plotName << "_layer";
      plotName << plotLayerN;
    }

    // PNG,EPS,PDF files
    c.Update(); 
    c.Print((plotName.str() + ".png").c_str());
    c.Print((plotName.str() + ".eps").c_str());
    c.Print((plotName.str() + ".pdf").c_str());

    // ROOT file
    TFile f((plotName.str() + ".root").c_str(), "recreate");
    c.Write();
    f.Close();
    
    // Free allocated memory.
    delete plotinfo.h;
    delete plotinfo.h1;
    delete plotinfo.h2;

  }

}

//------------------------------------------------------------------------------
void PlotAlignmentValidation::plotChi2(const char *inputFile)
{
  // Opens the file (it should be OfflineValidation(Parallel)_result.root)
  // and reads and plots the norm_chi^2 and h_chi2Prob -distributions.

  // First set default style: plots are already formatted
  TStyle defStyle("Default","Default Style");
  defStyle.cd();
  gStyle->SetOptStat(1);
  TGaxis::SetMaxDigits(3);

  Bool_t errorflag = kTRUE;
  TFile* fi1 = TFile::Open(inputFile,"read");
  TDirectoryFile* mta1 = NULL;
  TDirectoryFile* mtb1 = NULL;
  TCanvas* normchi = NULL;
  TCanvas* chiprob = NULL;
  if (fi1 != NULL) {
    mta1 = (TDirectoryFile*) fi1->Get("TrackerOfflineValidationStandalone");
    if(mta1 != NULL) {
      mtb1 = (TDirectoryFile*) mta1->Get("GlobalTrackVariables");
      if(mtb1 != NULL) {
        normchi = (TCanvas*) mtb1->Get("h_normchi2");
	chiprob = (TCanvas*) mtb1->Get("h_chi2Prob");
        if (normchi != NULL && chiprob != NULL) {
          errorflag = kFALSE;
        }
      }
    }
  }
  if(errorflag)
  {
    std::cout << "PlotAlignmentValidation::plotChi2: Can't find data from given file,"
              << " no chi^2-plots produced" << std::endl;
    return;
  }

  // Small adjustments: move the legend right and up so that it doesn't block
  // the exponent of the y-axis scale and doesn't cut the histogram border
  TLegend* l = (TLegend*)findObjectFromCanvas(normchi, "TLegend");
  if (l != 0) {
    l->SetX1NDC(0.25);
    l->SetY1NDC(0.86);
  }
  l = (TLegend*)findObjectFromCanvas(chiprob, "TLegend");
  if (l != 0) {
    l->SetX1NDC(0.25);
    l->SetY1NDC(0.86);
  }

  // Move stat boxes slightly right so that the border lines fit in
  int i = 1;
  for (TH1F* h = (TH1F*)findObjectFromCanvas(normchi, "TH1F", i); h != 0;
       h = (TH1F*)findObjectFromCanvas(normchi, "TH1F", ++i)) {
        TPaveStats *s = (TPaveStats*)h->GetListOfFunctions()->FindObject("stats");
        if (s != 0)
          s->SetX2NDC(0.995);
  }
  i = 1;
  for (TH1F* h = (TH1F*)findObjectFromCanvas(chiprob, "TH1F", i); h != 0;
       h = (TH1F*)findObjectFromCanvas(chiprob, "TH1F", ++i)) {
        TPaveStats *s = (TPaveStats*)h->GetListOfFunctions()->FindObject("stats");
        if (s != 0)
          s->SetX2NDC(0.995);
  }

  chiprob->Draw();
  normchi->Draw();

  // PNG,EPS,PDF files
  normchi->Print((outputDir + "/h_normchi2.png").c_str());
  chiprob->Print((outputDir + "/h_chi2Prob.png").c_str());
  normchi->Print((outputDir + "/h_normchi2.eps").c_str());
  chiprob->Print((outputDir + "/h_chi2Prob.eps").c_str());
  normchi->Print((outputDir + "/h_normchi2.pdf").c_str());
  chiprob->Print((outputDir + "/h_chi2Prob.pdf").c_str());

  // ROOT files
  TFile fi2((outputDir + "/h_normchi2.root").c_str(), "recreate");
  normchi->Write();
  fi2.Close();

  TFile fi3((outputDir + "/h_chi2Prob.root").c_str(), "recreate");
  chiprob->Write();
  fi3.Close();

  fi1->Close();
  TGaxis::SetMaxDigits(4);

}

//------------------------------------------------------------------------------
THStack* PlotAlignmentValidation::addHists(const TString& selection, const TString &residType,
					   TLegend **myLegend, bool printModuleIds)
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

  cout << "PlotAlignmentValidation::addHists: using selection " << selection << endl;
  THStack * retHistoStack = new THStack("hstack", "");
  if (myLegend != 0)
    if (*myLegend == 0) {
      *myLegend = new TLegend(0.17, 0.80, 0.85, 0.88);
      setLegendStyle( **myLegend );
    }

  for(std::vector<TkOfflineVariables*>::iterator itSourceFile = sourceList.begin();
      itSourceFile != sourceList.end(); ++itSourceFile) {

    std::vector<TString> histnames;

    TFile *f = (*itSourceFile)->getFile();
    TTree *tree= (*itSourceFile)->getTree();
    int myLineColor = (*itSourceFile)->getLineColor();
    int myLineStyle = (*itSourceFile)->getLineStyle();
    TString myLegendName = (*itSourceFile)->getName();
    TH1 *h = 0;       // becomes result
    UInt_t nEmpty = 0;// selected, but empty hists
    Long64_t nentries =  tree->GetEntriesFast();
    if (!f || !tree) {
      std::cout << "PlotAlignmentValidation::addHists: no tree or no file" << std::endl;
      return 0;
    }

    bool histnamesfilled = false;
    if (residType.Contains("Res") && residType.Contains("Profile"))
    {
      TString basename = TString(residType).ReplaceAll("Res","p_res")
                                           .ReplaceAll("vs","")
                                           .ReplaceAll("Profile","_");   //gives e.g.: p_resXX_
      if (selection == "subDetId==1") {
        histnames.push_back(TString(basename) += "TPBBarrel_1");
        histnamesfilled = true;
      } else if (selection == "subDetId==2") {
        histnames.push_back(TString(basename) += "TPEEndcap_2");
        histnames.push_back(TString(basename) += "TPEEndcap_3");
        histnamesfilled = true;
      } else if (selection == "subDetId==3") {
        histnames.push_back(TString(basename) += "TIBBarrel_1");
        histnamesfilled = true;
      } else if (selection == "subDetId==4") {
        histnames.push_back(TString(basename) += "TIDEndcap_2");
        histnames.push_back(TString(basename) += "TIDEndcap_3");
        histnamesfilled = true;
      } else if (selection == "subDetId==5") {
        histnames.push_back(TString(basename) += "TOBBarrel_4");
        histnamesfilled = true;
      } else if (selection == "subDetId==6") { //whole TEC - doesn't happen by default but easy enough to account for
        histnames.push_back(TString(basename) += "TECEndcap_5");
        histnames.push_back(TString(basename) += "TECEndcap_6");
        histnamesfilled = true;
      } else if (selection == "subDetId==6 && ring <= 4") {
        //There are multiple with the same name and all are needed, so give the full path.  For these TFile::Get is used later instead of FindKeyAny.
        for (int iEndcap = 5; iEndcap <= 6; iEndcap++)
          for (int iDisk = 1; iDisk <= 9; iDisk++)
            for (int iSide = 1; iSide <= 2; iSide++)
              for (int iPetal = 1; iPetal <= 8; iPetal++)
                for (int iRing = 1; iRing <= 4 - (iDisk>=4) - (iDisk>=7) - (iDisk>=9); iRing++)
                //in the higher disks, the inner rings go away.  But the numbering in the file structure removes the higher numbers
                // so the numbers there do not correspond to the actual ring numbers
                {
                  stringstream s;
                  s << "TrackerOfflineValidationStandalone/Strip/TECEndcap_" << iEndcap
                                                            << "/TECDisk_"   << iDisk
                                                            << "/TECSide_"   << iSide
                                                            << "/TECPetal_"  << iPetal
                                         << "/" << basename <<  "TECRing_"   << iRing;
                  histnames.push_back(TString(s.str()));
                }
        histnamesfilled = true;
      } else if (selection == "subDetId==6 && ring > 4") {
        //There are multiple with the same name and all are needed, so give the full path.  For these TFile::Get is used later instead of FindKeyAny.
        for (int iEndcap = 5; iEndcap <= 6; iEndcap++)
          for (int iDisk = 1; iDisk <= 9; iDisk++)
            for (int iSide = 1; iSide <= 2; iSide++)
              for (int iPetal = 1; iPetal <= 8; iPetal++)
                for (int iRing = 5 - (iDisk>=4) - (iDisk>=7) - (iDisk>=9); iRing <= 7 - (iDisk>=4) - (iDisk>=7) - (iDisk>=9); iRing++)
                //in the higher disks, the inner rings go away.  But the numbering in the file structure removes the higher numbers
                // so the numbers there do not correspond to the actual ring numbers
                {
                  stringstream s;
                  s << "TrackerOfflineValidationStandalone/Strip/TECEndcap_" << iEndcap
                                                            << "/TECDisk_"   << iDisk
                                                            << "/TECSide_"   << iSide
                                                            << "/TECPetal_"  << iPetal
                                         << "/" << basename <<  "TECRing_"   << iRing;
                  histnames.push_back(TString(s.str()));
                }
        histnamesfilled = true;
      }
    }


    Long64_t nSel = 0;
    if (histnamesfilled && histnames.size() > 0) {
      nSel = (Long64_t)histnames.size();

      //============================================================
      //for compatibility - please remove this at some point
      //it's now the end of August 2015
      TH1 *firstHist = 0;
      if (histnames[0].Contains("/")) {
        firstHist = (TH1*)f->Get(histnames[0]);
      } else {
        TKey *histKey = f->FindKeyAny(histnames[0]);
        if (histKey)
          firstHist = (histKey ? static_cast<TH1*>(histKey->ReadObj()) : 0);
      }
      if (!firstHist)             //then the validation was done with an older version of TrackerOfflineValidation
      {                           // ==> have to make the plots the old (long) way
        histnamesfilled = false;
        histnames.clear();
      }
      //============================================================

    }
    if (!histnamesfilled) {
      // first loop on tree to find out which entries (i.e. modules) fulfill the selection
      // 'Entry$' gives the entry number in the tree
      nSel = tree->Draw("Entry$", selection, "goff");
      if (nSel == -1) return 0; // error in selection
      if (nSel == 0) {
        std::cout << "PlotAlignmentValidation::addHists: no selected module." << std::endl;
        return 0;
      }
      // copy entry numbers that fulfil the selection
      const std::vector<double> selected(tree->GetV1(), tree->GetV1() + nSel);

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
        histnames.push_back(hName);
      }
    }

    for (std::vector<TString>::iterator ithistname = histnames.begin();
      ithistname != histnames.end(); ++ithistname) {
      TH1 *newHist;
      if (ithistname->Contains("/")) {
        newHist = (TH1*)f->Get(*ithistname);
      } else {
        TKey *histKey = f->FindKeyAny(*ithistname);
        newHist = (histKey ? static_cast<TH1*>(histKey->ReadObj()) : 0);
      }
      if (!newHist) {
	std::cout << "Hist " << *ithistname << " not found in file, break loop." << std::endl;
	break;
      }
      if (newHist->GetEntries() == 0) {
        nEmpty++;
        continue;
      }
      newHist->SetLineColor(myLineColor);
      newHist->SetLineStyle(myLineStyle);
      if (!h) { // first hist: clone, but rename keeping only first part of name
	TString name(newHist->GetName());
	Ssiz_t pos_ = 0;
	for (UInt_t i2 = 0; i2 < 3; ++i2) pos_ = name.Index("_", pos_+1);
	name = name(0, pos_); // only up to three '_'
	h = static_cast<TH1*>(newHist->Clone("summed_"+name));
	//      TString myTitle = Form("%s: %lld modules", selection, nSel);
	//	h->SetTitle( myTitle );
      } else { // otherwise just add
	h->Add(newHist);
      }
      delete newHist;
    }

    std::cout << "PlotAlignmentValidation::addHists" << "Result is merged from " << nSel-nEmpty
	      << " hists, " << nEmpty << " hists were empty." << std::endl;

    if (nSel-nEmpty == 0) continue;

    if (myLegend != 0)
      (*myLegend)->AddEntry(h, myLegendName, "L");

    retHistoStack->Add(h);
  }

  return retHistoStack;
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
  canv.SetTopMargin   ( 0.12 );
}

//------------------------------------------------------------------------------
void  PlotAlignmentValidation::setLegendStyle( TLegend& leg )
{
  leg.SetFillStyle ( 0 );
  leg.SetFillColor ( 0 );
  leg.SetBorderSize( 0 ); 
}

//------------------------------------------------------------------------------
void PlotAlignmentValidation::scaleXaxis(TH1* hist, Int_t scale)
{
  Double_t xmin = hist->GetXaxis()->GetXmin();
  Double_t xmax = hist->GetXaxis()->GetXmax();
  hist->GetXaxis()->SetLimits(xmin*scale, xmax*scale);
}

//------------------------------------------------------------------------------
TObject* PlotAlignmentValidation::findObjectFromCanvas(TCanvas* canv, const char* className, Int_t n) {
  // Finds the n-th instance of the given class from the canvas
  TIter next(canv->GetListOfPrimitives());
  TObject* obj = 0;
  Int_t found = 0;
  while ((obj = next())) {
    if(strncmp(obj->ClassName(), className, 10) == 0) {
      if (++found == n)
        return obj;
    }
  }

  return 0;
}

//------------------------------------------------------------------------------
void  PlotAlignmentValidation::setNiceStyle() {
  TStyle *MyStyle = new TStyle ("MyStyle", "My style for nicer plots");
  
  Float_t xoff = MyStyle->GetLabelOffset("X"),
    yoff = MyStyle->GetLabelOffset("Y"),
    zoff = MyStyle->GetLabelOffset("Z");

  MyStyle->SetCanvasBorderMode ( 0 );
  MyStyle->SetFrameBorderMode ( 0 );
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
    case 1: histTitel+="BPIX";break;
    case 2: histTitel+="FPIX";break;
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
  else if( titelXAxis.Contains("meanX") )titel_Xaxis<<"#LTx'_{pred}-x'_{hit}#GT[#mum]";
  else if( titelXAxis.Contains("meanY") )titel_Xaxis<<"#LTy'_{pred}-y'_{hit}#GT[#mum]";
  else if( titelXAxis.Contains("rmsX") )titel_Xaxis<<"RMS(x'_{pred}-x'_{hit})[#mum]";
  else if( titelXAxis.Contains("rmsY") )titel_Xaxis<<"RMS(y'_{pred}-y'_{hit})[#mum]";
  else if( titelXAxis.Contains("meanNormX") )titel_Xaxis<<"#LTx'_{pred}-x'_{hit}/#sigma#GT";
  else if( titelXAxis.Contains("meanNormY") )titel_Xaxis<<"#LTy'_{pred}-y'_{hit}/#sigma#GT";
  else if( titelXAxis.Contains("rmsNormX") )titel_Xaxis<<"RMS(x'_{pred}-x'_{hit}/#sigma)";
  else if( titelXAxis.Contains("rmsNormY") )titel_Xaxis<<"RMS(y'_{pred}-y'_{hit}/#sigma)";
  else if( titelXAxis.Contains("meanLocalX") )titel_Xaxis<<"#LTx_{pred}-x_{hit}#GT[#mum]";
  else if( titelXAxis.Contains("rmsLocalX") )titel_Xaxis<<"RMS(x_{pred}-x_{hit})[#mum]";
  else if( titelXAxis.Contains("meanNormLocalX") )titel_Xaxis<<"#LTx_{pred}-x_{hit}/#sigma#GT[#mum]";
  else if( titelXAxis.Contains("rmsNormLocalX") )titel_Xaxis<<"RMS(x_{pred}-x_{hit}/#sigma)[#mum]";
  else if( titelXAxis.Contains("medianX") )titel_Xaxis<<"median(x'_{pred}-x'_{hit})[#mum]";
  else if( titelXAxis.Contains("medianY") )titel_Xaxis<<"median(y'_{pred}-y'_{hit})[#mum]";
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
getSelectionForDMRPlot(int minHits, int subDetId, int direction, int layer)
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
  if (layer > 0) {
    builder << " && layer == " << layer;
  }
  return builder.str();
}

std::string PlotAlignmentValidation::
getVariableForDMRPlot(const std::string& histoname, const std::string& variable, int nbins, double min,
		      double max)
{
  std::ostringstream builder;
  builder << variable << ">>" << histoname << "(" << nbins << "," << min <<
    "," << max << ")";
  return builder.str();
}

void PlotAlignmentValidation::
setDMRHistStyleAndLegend(TH1F* h, PlotAlignmentValidation::DMRPlotInfo& plotinfo, int direction, int layer)
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
  if (direction == 1) { linestyleplus = 1; }
  if (direction == -1) { linestyleplus = 2; }
  if (direction != 0 && plotinfo.plotSplits && !plotinfo.plotPlain) { linestyleplus--; }
  linestyle = (linestyle + linestyleplus) % 4 + 1;

  int linecolor = plotinfo.vars->getLineColor();
  if (plotinfo.plotLayers && layer > 0) { linecolor += layer - 1; }

  if (plotinfo.firsthisto) {
    setHistStyle(*h, plotinfo.variable.c_str(), "#modules", 1); //set color later
    plotinfo.firsthisto = false;
  }

  h->SetLineColor( linecolor );
  h->SetLineStyle( linestyle );
	  
  if (plotinfo.maxY<h->GetMaximum()){
    plotinfo.maxY=h->GetMaximum();
  }
	  
  //fit histogram for median and mean
  if (plotinfo.variable == "medianX" || plotinfo.variable == "meanX") {
    fitResults = fitGauss(h, linecolor );
  }
	  
  plotinfo.hstack->Add(h);

  std::ostringstream legend;
  legend.precision(3);
  legend << fixed; // to always show 3 decimals

  // Legend: header part
  if (direction == -1 && plotinfo.subDetId != 2) { legend << "rDirection < 0: "; }
  else if (direction == 1 && plotinfo.subDetId != 2) { legend << "rDirection > 0: "; }
  else if (direction == -1 && plotinfo.subDetId == 2) { legend << "zDirection < 0: "; }
  else if (direction == 1 && plotinfo.subDetId == 2) { legend << "zDirection > 0: "; }
  else {
    legend  << plotinfo.vars->getName();
    if (layer > 0) {
      // TEC and TID have discs, the rest have layers
      if (plotinfo.subDetId==4 || plotinfo.subDetId==6)
        legend << ", disc ";
      else
        legend << ", layer ";
      legend << layer << "";
    }
    legend << ":";
  }

  // Legend: Statistics
  if (plotinfo.variable == "medianX" || plotinfo.variable == "meanX" ||
      plotinfo.variable == "medianY" || plotinfo.variable == "meanY") {
    if (useFit_) {
      legend << " #mu = " << fitResults.first << " #mum, #sigma = " << fitResults.second << " #mum";
    } else {
      legend << " #mu = " << h->GetMean(1)*10000 << " #mum, rms = " << h->GetRMS(1)*10000 << " #pm " << h->GetRMSError(1)*10000 << " #mum, " << (int) h->GetEntries() << " modules" ;
    }
  } else if (plotinfo.variable == "rmsX" || plotinfo.variable == "rmsY") {
    legend << " #mu = " << h->GetMean(1)*10000 << " #mum, rms = " << h->GetRMS(1)*10000 << " #mum";
  } else if (plotinfo.variable == "meanNormX" || plotinfo.variable == "meanNormY" ||
	     plotinfo.variable == "rmsNormX" || plotinfo.variable == "rmsNormY") {
    legend << " #mu = " << h->GetMean(1) << ", rms = " << h->GetRMS(1);
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

  // Scale the x-axis (cm to um), if needed
  if (plotinfo.variable.find("Norm") == std::string::npos)
    scaleXaxis(h, 10000);

}

void PlotAlignmentValidation::
plotDMRHistogram(PlotAlignmentValidation::DMRPlotInfo& plotinfo, int direction, int layer)
{
  TH1F* h = 0;
  std::string histoname;
  if (direction == -1) { histoname = "myhisto1"; }
  else if (direction == 1) { histoname = "myhisto2"; }
  else { histoname = "myhisto"; }
  std::string plotVariable = getVariableForDMRPlot(histoname, plotinfo.variable, plotinfo.nbins, plotinfo.min, plotinfo.max);
  std::string selection = getSelectionForDMRPlot(plotinfo.minHits, plotinfo.subDetId, direction, layer);
  plotinfo.vars->getTree()->Draw(plotVariable.c_str(), selection.c_str(), "goff");
  if (gDirectory) gDirectory->GetObject(histoname.c_str(), h);
  if (h && h->GetEntries() > 0) {
    if (direction == -1) { plotinfo.h1 = h; }
    else if (direction == 1) { plotinfo.h2 = h; }
    else { plotinfo.h = h; }
  }
}

void PlotAlignmentValidation::modifySSHistAndLegend(THStack* hs, TLegend* legend)
{
  // Add mean-y-values to the legend and scale the histograms.

  Double_t legendY = 0.80;
  if (hs->GetHists()->GetSize() > 3)
    legendY -= 0.01 * (hs->GetHists()->GetSize() - 3);
  if (legendY < 0.6) {
    std::cerr << "Warning: Huge legend!" << std::endl;
    legendY = 0.6;
  }
  legend->SetY1(legendY);

  // Loop over all profiles
  TProfile* prof = 0;
  TIter next(hs->GetHists());
  Int_t index = 0;
  while ((prof = (TProfile*)next())) {
    //Scaling: from cm to um
    Double_t scale = 10000;
    prof->Scale(scale);

    Double_t stats[6] = {0};
    prof->GetStats(stats);

    std::ostringstream legendtext;
    legendtext.precision(3);
    legendtext << fixed; // to always show 3 decimals
    legendtext << ": y mean = " << stats[4]/stats[0]*scale << " #mum";

    TLegendEntry* entry = (TLegendEntry*)legend->GetListOfPrimitives()->At(index);
    if (entry == 0)
      cout << "PlotAlignmentValidation::PlotAlignmentValidation::modifySSLegend: Bad legend!" << endl;
    else
      entry->SetLabel((entry->GetLabel() + legendtext.str()).c_str());
    index++;
  }

  // Make some room for the legend
  hs->SetMaximum(hs->GetMaximum("nostack PE")*1.3);
}
