#include "Alignment/OfflineValidation/macros/PlotAlignmentValidation.h"

#include "Alignment/OfflineValidation/macros/TkAlStyle.cc"
#include "Alignment/OfflineValidation/interface/TkOffTreeVariables.h"

#include "Math/ProbFunc.h"

#include "TAxis.h"
#include "TCanvas.h"
#include "TDirectory.h"
#include "TDirectoryFile.h"
#include "TF1.h"
#include "TFile.h"
#include "TGaxis.h"
#include "TH2F.h"
#include "THStack.h"
#include "TKey.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TLegendEntry.h"
#include "TPad.h"
#include "TPaveStats.h"
#include "TPaveText.h"
#include "TProfile.h"
#include "TRandom3.h"
#include "TRegexp.h"
#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TTree.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>


/*! \class PlotAlignmentValidation
 *  \brief Class PlotAlignmentValidation
 *         Class used as the last step for Offline Track Validation tool.
 *         The main goal of this class is creating the plots regarding DMRs and Surface Deformations for modules and substructures.
 */


//------------------------------------------------------------------------------
/*! \fn PlotAlignmentValidation
 *  \brief Constructor for the class
 */

PlotAlignmentValidation::PlotAlignmentValidation(bool bigtext) : bigtext_(bigtext)
{
  setOutputDir(".");
  setTreeBaseDir();
  sourcelist = NULL;
  
  moreThanOneSource=false;
  useFit_ = false;

  // Force ROOT to use scientific notation even with smaller datasets
  TGaxis::SetMaxDigits(4);
  // (This sets a static variable: correct in .eps images but must be set
  // again manually when viewing the .root files)

  // Make ROOT calculate histogram statistics using all data,
  // regardless of displayed range
  TH1::StatOverflows(kTRUE);

  //show all information in the legend by default
  legendOptions(TkAlStyle::legendoptions);
}

//------------------------------------------------------------------------------
/*! \fn PlotAlignmentValidation
 *  \brief Constructor for the class. This function also retrieves the list of root files used to produce DMRs and Surface Deformations
 */
PlotAlignmentValidation::PlotAlignmentValidation(const char *inputFile,std::string legendName, int lineColor, int lineStyle, bool bigtext) : PlotAlignmentValidation(bigtext)
{
  loadFileList(inputFile, legendName, lineColor, lineStyle);
}

//------------------------------------------------------------------------------
/*! \fn ~PlotAlignmentValidation
 *  \brief Default destructor
 */
PlotAlignmentValidation::~PlotAlignmentValidation()
{

  for(std::vector<TkOfflineVariables*>::iterator it = sourceList.begin();
      it != sourceList.end(); ++it){
    delete (*it);
  }

  delete sourcelist;

}

//------------------------------------------------------------------------------
/*!
 * \fn openSummaryFile
 * \brief Create/open the root and txt summary files, where the DMR histograms and the associtated mean/sigma are stored respectively
 */


void PlotAlignmentValidation::openSummaryFile()
{
  if (!openedsummaryfile) {
    openedsummaryfile = true;
    summaryfile.open(outputDir+"/"+summaryfilename+".txt");
    //Rootfile introduced to store the DMR histograms
    rootsummaryfile= new TFile(outputDir+"/"+summaryfilename+".root","RECREATE");

    for (auto vars : sourceList) {
      summaryfile << "\t" << vars->getName();
    }
    summaryfile << "\tformat={}\n";
  }else{
    //Check for the rootfile to be open, and open it in case it is not already.
    if (!rootsummaryfile->IsOpen()) rootsummaryfile->Open(outputDir+"/"+summaryfilename+".root","UPDATE");

  }
}

//------------------------------------------------------------------------------
/*! \fn loadFileList
 *  \brief Add to the list of sources the rootfile associated to a particular geometry
 */
void PlotAlignmentValidation::loadFileList(const char *inputFile, std::string legendName, int lineColor, int lineStyle)
{

  if (openedsummaryfile) {
    std::cout << "Can't load a root file after opening the summary file!" << std::endl;
    assert(0);
  }
  sourceList.push_back( new TkOfflineVariables( inputFile, treeBaseDir, legendName, lineColor, lineStyle ) );

}

//------------------------------------------------------------------------------
/*! \fn useFitForDMRplots
 *  \brief Store the selected boolean in one of the private members of the class
 */
void PlotAlignmentValidation::useFitForDMRplots(bool usefit)
{

  useFit_ = usefit;
  
}

//------------------------------------------------------------------------------
/*! \fn numberOfLayers
 *  \brief Select the number of layers associated to a subdetector.
 */
//TODO Possible improvement: reduce the number of switches in the code by implementing a map
int PlotAlignmentValidation::numberOfLayers(int phase, int subdetector) {
  switch (phase) {
  case 0:
    switch (subdetector) {
      case 1: return 3;
      case 2: return 2;
      case 3: return 4;
      case 4: return 3;
      case 5: return 6;
      case 6: return 9;
      default: assert(false);
    }
  case 1:
    switch (subdetector) {
      case 1: return 4;
      case 2: return 3;
      case 3: return 4;
      case 4: return 3;
      case 5: return 6;
      case 6: return 9;
      default: assert(false);
    }
    default: assert(false);
  }
  return 0;
}

//------------------------------------------------------------------------------
/*! \fn maxNumberOfLayers
 *  \brief Return the number of layers of a subdetector
 */
int PlotAlignmentValidation::maxNumberOfLayers(int subdetector) {
  int result = 0;
  for (auto it = sourceList.begin(); it != sourceList.end(); ++it) {
    result = max(result, numberOfLayers((*it)->getPhase(), subdetector));
  }
  return result;
}

//------------------------------------------------------------------------------
/*! \fn legendOptions
 *  \brief Assign legend options to members of the class
 */
void PlotAlignmentValidation::legendOptions(TString options)
{

  showMean_ = false;
  showRMS_ = false;
  showMeanError_ = false;
  showRMSError_ = false;
  showModules_ = false;
  showUnderOverFlow_ = false;
  options.ReplaceAll(" ","").ToLower();
  if (options.Contains("mean") || options.Contains("all"))
    showMean_ = true;
  if (options.Contains("meanerror") || options.Contains("all"))
    showMeanError_ = true;
  if (options.Contains("rms") || options.Contains("all"))
    showRMS_ = true;
  if (options.Contains("rmserror") || options.Contains("all"))
    showRMSError_ = true;
  if (options.Contains("modules") || options.Contains("all"))
    showModules_ = true;
  if (options.Contains("under") || options.Contains("over") || options.Contains("outside") || options.Contains("all"))
    showUnderOverFlow_ = true;

  twolines_ = (showUnderOverFlow_ && (showMean_ + showMeanError_ + showRMS_ + showRMSError_ >= 1) && bigtext_);
}

//------------------------------------------------------------------------------
/*! \fn setOutputDir
 *  \brief Set the output direcotry
 */
void PlotAlignmentValidation::setOutputDir( std::string dir )
{
  if (openedsummaryfile) {
    std::cout << "Can't set the output dir after opening the summary file!" << std::endl;
    assert(0);
  }
  outputDir = dir;
  gSystem->mkdir(outputDir.data(), true);
}

//------------------------------------------------------------------------------
/*! \fn plotSubDetResiduals
 *  \brief Function used to plot residuals for a subdetector
 */
void PlotAlignmentValidation::plotSubDetResiduals(bool plotNormHisto,unsigned int subDetId)
{
  gStyle->SetOptStat(11111);
  gStyle->SetOptFit(0000);

  TCanvas *c = new TCanvas("c", "c");
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
  
  //gStyle->SetOptStat(0);
  
  TCanvas *c = new TCanvas("c", "c", 1200,400);
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
  
  gStyle->SetOptStat(111111);
  gStyle->SetStatY(0.9);
  //TList* treelist=getTreeList();
  
  TCanvas *c1 = new TCanvas("canv", "canv", 800, 500);
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
/*! \fn getTreeList
 *  \brief Extract from the rootfiles stored in the sourcelist the corresponding trees.
 */
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

  int bkperrorx = gStyle->GetErrorX();
  gStyle->SetErrorX(1);   //regardless of style settings, we want x error bars here

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

  gStyle->SetOptStat(0);
  
  TCanvas c("canv", "canv");

  // todo: title, min/max, nbins?

  // Loop over detectors
  for (int iSubDet=1; iSubDet<=6; ++iSubDet) {

    // TEC requires special care since rings 1-4 and 5-7 are plotted separately
    bool isTEC = (iSubDet==6);

    // if subdet is specified, skip other subdets
    if (plotSubDetN!=0 && iSubDet!=plotSubDetN)
      continue;

    // Skips plotting too high layers
    // if it's a mixture of phase 0 and 1, the phase 0 files will be skipped
    //  when plotting the higher layers of BPIX and FPIX
    if (plotLayerN > maxNumberOfLayers(iSubDet)) {
      continue;
    }

    int minlayer = plotLayers ? 1 : plotLayerN;
    int maxlayer = plotLayers ? maxNumberOfLayers(iSubDet) : plotLayerN;
    // see later where this is used
    int maxlayerphase0 = plotLayers ? numberOfLayers(0, iSubDet) : plotLayerN;
    
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

        TString secondline = "";
	if (layer!=0) {
	  // TEC and TID have discs, the rest have layers
	  if (iSubDet==4 || iSubDet==6)
	    secondline = "disc ";
	  else {
	    secondline = "layer ";
	  }
	  secondline += Form("%d",layer);
	  secondline += " ";
	}
	if (isTEC && iTEC==0)
	  secondline += TString("R1-4");
	if (isTEC && iTEC>0)
	  secondline += TString("R5-7");

	// Generate histograms with selection
	TLegend* legend = 0;
        // Any file from phase 0 will be skipped if the last argument is false
	THStack *hs = addHists(selection, residType, &legend, false, /*validforphase0 = */layer <= maxlayerphase0);
	if (!hs || hs->GetHists()==0 || hs->GetHists()->GetSize()==0) {
	  std::cout << "No histogram for " << subDetName <<
	               ", perhaps not enough data? Creating default histogram." << std::endl;
	  if(hs == 0)
	    hs = new THStack("hstack", "");

	  TProfile* defhist = new TProfile("defhist", "Empty default histogram", 100, -1, 1, -1, 1);
	  hs->Add(defhist);
	  hs->Draw();
	}
	else {
	  hs->Draw("nostack PE");
	  modifySSHistAndLegend(hs, legend);
	  legend->Draw();
	  setTitleStyle(*hs, "", "", iSubDet, true, secondline);

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
  gStyle->SetErrorX(bkperrorx);

  return;
}


//------------------------------------------------------------------------------
/*! \fn plotDMR
 *  \brief Main function used to plot DMRs for a single IOV printing the canvases in the output directory and saving histograms and fit funtions in a root file.
 */
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

  DMRPlotInfo plotinfo;

  gStyle->SetOptStat(0);
  
  TCanvas c("canv", "canv");

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
  //Begin loop on structures
  for (int i=1; i<=6; ++i) {

    // Skip strip detectors if plotting any "Y" variable
    if (i != 1 && i != 2 && variable.length() > 0 && variable[variable.length()-1] == 'Y') {
      continue;
    }
 
    // Skips plotting too high layers
    if (plotLayerN > maxNumberOfLayers(i)) {
      continue;
    }

    plotinfo.plotSplits = plotSplits && plotSplitsFor[i-1];
    if (!plotinfo.plotPlain && !plotinfo.plotSplits) {
      continue;
    }

    // Sets dimension of legend according to the number of plots

    bool hasheader = (TkAlStyle::legendheader != "");

    int nPlots = 1;
    if (plotinfo.plotSplits) { nPlots = 3; }
    // This will make the legend a bit bigger than necessary if there is a mixture of phase 0 and phase 1.
    // Not worth it to implement more complicated logic.
    if (plotinfo.plotLayers) { nPlots *= maxNumberOfLayers(i); }
    nPlots *= sourceList.size();
    if (twolines_) { nPlots *= 2; }
    nPlots += hasheader;

    double legendY = 0.80;
    if (nPlots > 3) { legendY -= 0.01 * (nPlots - 3); }
    if (bigtext_) { legendY -= 0.05; }
    if (legendY < 0.6) {
      std::cerr << "Warning: Huge legend!" << std::endl;
      legendY = 0.6;
    }

    THStack hstack("hstack", "hstack");
    plotinfo.maxY = 0;
    plotinfo.subDetId = i;
    plotinfo.legend = new TLegend(0.17, legendY, 0.85, 0.88);
    plotinfo.legend->SetNColumns(2);
    if (hasheader) plotinfo.legend->SetHeader(TkAlStyle::legendheader);
    if (bigtext_) plotinfo.legend->SetTextSize(TkAlStyle::textSize);
    plotinfo.legend->SetFillStyle(0);
    plotinfo.hstack = &hstack;
    plotinfo.h = plotinfo.h1 = plotinfo.h2 = 0;
    plotinfo.firsthisto = true;

    openSummaryFile();
    vmean.clear(); vrms.clear(); vdeltamean.clear(); vmeanerror.clear(); vPValueEqualSplitMeans.clear(), vAlignmentUncertainty.clear(); vPValueMeanEqualIdeal.clear(); vPValueRMSEqualIdeal.clear();

    std::string stringsubdet;
    switch (i) {
      case 1: stringsubdet = "BPIX"; break;
      case 2: stringsubdet = "FPIX"; break;
      case 3: stringsubdet = "TIB"; break;
      case 4: stringsubdet = "TID"; break;
      case 5: stringsubdet = "TOB"; break;
      case 6: stringsubdet = "TEC"; break;
    }

    for(std::vector<TkOfflineVariables*>::iterator it = sourceList.begin();
	it != sourceList.end(); ++it) {

      plotinfo.vars = *it;
      plotinfo.h1 = plotinfo.h2 = plotinfo.h = 0;

      int minlayer = plotLayers ? 1 : plotLayerN;
      //Layer 0 is associated to the entire structure, this check ensures that even when both the plotLayers and the plotPlain options are active, also the histogram for the entire structure is made.
      if(plotinfo.plotPlain) minlayer=0; 
      int maxlayer = plotLayers ? numberOfLayers(plotinfo.vars->getPhase(), plotinfo.subDetId) : plotLayerN;

      for (int layer = minlayer; layer <= maxlayer; layer++) {

	if (plotinfo.plotPlain) {
	  plotDMRHistogram(plotinfo, 0, layer, stringsubdet);
	}

	if (plotinfo.plotSplits) {
	  plotDMRHistogram(plotinfo, -1, layer, stringsubdet);
	  plotDMRHistogram(plotinfo, 1, layer, stringsubdet);
	}

	if (plotinfo.plotPlain) {
	  if (plotinfo.h) {
            setDMRHistStyleAndLegend(plotinfo.h, plotinfo, 0, layer);
          } else {
            if ((plotinfo.variable == "medianX" || plotinfo.variable == "medianY") && /*!plotinfo.plotLayers && */layer==0) {
              vmean.push_back(nan(""));
              vrms.push_back(nan(""));
              vmeanerror.push_back(nan(""));
              vAlignmentUncertainty.push_back(nan(""));
              vPValueMeanEqualIdeal.push_back(nan(""));
              vPValueRMSEqualIdeal.push_back(nan(""));
              if (plotinfo.plotSplits && plotinfo.plotPlain) {
                vdeltamean.push_back(nan(""));
                vPValueEqualSplitMeans.push_back(nan(""));
              }
            }
          }
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
	    plotinfo.legend->AddEntry(static_cast<TObject*>(0), legend.str().c_str(), "");
	    legend.str("");
	    legend << "#Delta#mu = " << deltamu << unit;
	    plotinfo.legend->AddEntry(static_cast<TObject*>(0), legend.str().c_str(), "");

            if ((plotinfo.variable == "medianX" || plotinfo.variable == "medianY") && !plotLayers && layer==0) {
              vdeltamean.push_back(deltamu);
            }
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


    TString subdet;
    switch (i) {
      case 1: subdet = "BPIX"; break;
      case 2: subdet = "FPIX"; break;
      case 3: subdet = "TIB"; break;
      case 4: subdet = "TID"; break;
      case 5: subdet = "TOB"; break;
      case 6: subdet = "TEC"; break;
    }

    plotName << subdet;

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

    if (vmean.size()) {
      summaryfile << "   mu_" << subdet;
      if (plotinfo.variable == "medianY") summaryfile << "_y";
      summaryfile << " (um)\t"
                  << "latexname=$\\mu_\\text{" << subdet << "}";
      if (plotinfo.variable == "medianY") summaryfile << "^{y}";
      summaryfile << "$ ($\\mu$m)\t"
                  << "format={:.3g}\t"
                  << "latexformat=${:.3g}$";
      for (auto mu : vmean) summaryfile << "\t" << mu;
      summaryfile << "\n";
    }
    if (vrms.size()) {
      summaryfile << "sigma_" << subdet;
      if (plotinfo.variable == "medianY") summaryfile << "_y";
      summaryfile << " (um)\t"
                  << "latexname=$\\sigma_\\text{" << subdet << "}";
      if (plotinfo.variable == "medianY") summaryfile << "^{y}";
      summaryfile << "$ ($\\mu$m)\t"
                  << "format={:.3g}\t"
                  << "latexformat=${:.3g}$";
      for (auto sigma : vrms) summaryfile << "\t" << sigma;
      summaryfile << "\n";
    }
    if (vdeltamean.size()) {
      summaryfile << "  dmu_" << subdet ;
      if (plotinfo.variable == "medianY") summaryfile << "_y";
      summaryfile << " (um)\t"
                  << "latexname=$\\Delta\\mu_\\text{" << subdet << "}";
      if (plotinfo.variable == "medianY") summaryfile << "^{y}";
      summaryfile << "$ ($\\mu$m)\t"
                  << "format={:.3g}\t"
                  << "latexformat=${:.3g}$";
      for (auto dmu : vdeltamean) summaryfile << "\t" << dmu;
      summaryfile << "\n";
    }
    if (vmeanerror.size()) {
      summaryfile << "  sigma_mu_" << subdet ;
      if (plotinfo.variable == "medianY") summaryfile << "_y";
      summaryfile << " (um)\t"
                  << "latexname=$\\sigma\\mu_\\text{" << subdet << "}";
      if (plotinfo.variable == "medianY") summaryfile << "^{y}";
      summaryfile << "$ ($\\mu$m)\t"
                  << "format={:.3g}\t"
                  << "latexformat=${:.3g}$";
      for (auto dmu : vmeanerror) summaryfile << "\t" << dmu;
      summaryfile << "\n";
    }
    if (vPValueEqualSplitMeans.size()) {
      summaryfile << "  p_delta_mu_equal_zero_" << subdet ;
      if (plotinfo.variable == "medianY") summaryfile << "_y";
      summaryfile << "\t"
                  << "latexname=$P(\\delta\\mu_\\text{" << subdet << "}=0)";
      if (plotinfo.variable == "medianY") summaryfile << "^{y}";
      summaryfile << "$\t"
                  << "format={:.3g}\t"
                  << "latexformat=${:.3g}$";
      for (auto dmu : vPValueEqualSplitMeans) summaryfile << "\t" << dmu;
      summaryfile << "\n";
    }
    if (vAlignmentUncertainty.size()) {
      summaryfile << "  alignment_uncertainty_" << subdet ;
      if (plotinfo.variable == "medianY") summaryfile << "_y";
      summaryfile << " (um)\t"
                  << "latexname=$\\sigma_\\text{align}_\\text{" << subdet << "}";
      if (plotinfo.variable == "medianY") summaryfile << "^{y}";
      summaryfile << "$ ($\\mu$m)\t"
                  << "format={:.3g}\t"
                  << "latexformat=${:.3g}$";
      for (auto dmu : vAlignmentUncertainty) summaryfile << "\t" << dmu;
      summaryfile << "\n";
    }
    if (vPValueMeanEqualIdeal.size()) {
      summaryfile << "  p_mean_equals_ideal_" << subdet ;
      if (plotinfo.variable == "medianY") summaryfile << "_y";
      summaryfile << "\t"
                  << "latexname=$P(\\mu_\\text{" << subdet << "}=\\mu_\\text{ideal})";
      if (plotinfo.variable == "medianY") summaryfile << "^{y}";
      summaryfile << "$\t"
                  << "format={:.3g}\t"
                  << "latexformat=${:.3g}$";
      for (auto dmu : vPValueMeanEqualIdeal) summaryfile << "\t" << dmu;
      summaryfile << "\n";
    }
    if (vPValueRMSEqualIdeal.size()) {
      summaryfile << "  p_RMS_equals_ideal_" << subdet ;
      if (plotinfo.variable == "medianY") summaryfile << "_y";
      summaryfile << "\t"
                  << "latexname=$P(\\sigma_\\text{" << subdet << "}=\\sigma_\\text{ideal})";
      if (plotinfo.variable == "medianY") summaryfile << "^{y}";
      summaryfile << "$\t"
                  << "format={:.3g}\t"
                  << "latexformat=${:.3g}$";
      for (auto dmu : vPValueRMSEqualIdeal) summaryfile << "\t" << dmu;
      summaryfile << "\n";
    }
  }
}

//------------------------------------------------------------------------------
void PlotAlignmentValidation::plotChi2(const char *inputFile)
{
  // Opens the file (it should be OfflineValidation(Parallel)_result.root)
  // and reads and plots the norm_chi^2 and h_chi2Prob -distributions.

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
        normchi = dynamic_cast<TCanvas*>(mtb1->Get("h_normchi2"));
        chiprob = dynamic_cast<TCanvas*>(mtb1->Get("h_chi2Prob"));
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

  TLegend *legend = 0;
  for (auto primitive : *normchi->GetListOfPrimitives()) {
    legend = dynamic_cast<TLegend*>(primitive);
    if (legend) break;
  }
  if (legend) {
    openSummaryFile();
    summaryfile << "ntracks";
    for (auto alignment : sourceList) {
      summaryfile << "\t";
      TString title = alignment->getName();
      int color = alignment->getLineColor();
      int style = alignment->getLineStyle();
      bool foundit = false;
      for (auto entry : *legend->GetListOfPrimitives()) {
        TLegendEntry *legendentry = dynamic_cast<TLegendEntry*>(entry);
        assert(legendentry);
        TH1 *h = dynamic_cast<TH1*>(legendentry->GetObject());
        if (!h) continue;
        if (legendentry->GetLabel() == title && h->GetLineColor() == color && h->GetLineStyle() == style) {
          foundit = true;
          summaryfile << h->GetEntries();
          break;
        }
      }
      if (!foundit) {
        summaryfile << 0;
      }
    }
    summaryfile << "\n";
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

  delete fi1;

}

//------------------------------------------------------------------------------
THStack* PlotAlignmentValidation::addHists(const TString& selection, const TString &residType,
					   TLegend **myLegend, bool printModuleIds, bool validforphase0)
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
    int phase = (bool)(f->Get("TrackerOfflineValidationStandalone/Pixel/P1PXBBarrel_1"));
    if (residType.Contains("Res") && residType.Contains("Profile"))
    {
      TString basename = TString(residType).ReplaceAll("Res","p_res")
                                           .ReplaceAll("vs","")
                                           .ReplaceAll("Profile","_");   //gives e.g.: p_resXX_
      if (selection == "subDetId==1") {
        if (phase==1)
          histnames.push_back(TString(basename) += "P1PXBBarrel_1");
        else
          histnames.push_back(TString(basename) += "TPBBarrel_1");
        histnamesfilled = true;
      } else if (selection == "subDetId==2") {
        if (phase==1) {
          histnames.push_back(TString(basename) += "P1PXECEndcap_2");
          histnames.push_back(TString(basename) += "P1PXECEndcap_3");
        } else {
          histnames.push_back(TString(basename) += "TPEEndcap_2");
          histnames.push_back(TString(basename) += "TPEEndcap_3");
        }
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
      if (phase == 0 && !validforphase0) break;
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
/*! \fn fitGauss
 *  \brief Operate a Gaussian fit to the given histogram
 */
TF1 *
PlotAlignmentValidation::fitGauss(TH1 *hist,int color) 
{
  //1. fits a Gauss function to the inner range of abs(2 rms)
  //2. repeates the Gauss fit in a 2 sigma range around mean of first fit
  //returns mean and sigma from fit in micron
  if (!hist || hist->GetEntries() < 20) return 0;

  float mean  = hist->GetMean();
  float sigma = hist->GetRMS();
  string functionname="gaussian_";
  functionname+=hist->GetName();
  TF1 *func = new TF1(functionname.c_str(), "gaus", mean - 2.*sigma, mean + 2.*sigma); 
 
  func->SetLineColor(color);
  func->SetLineStyle(2);
  if (0 == hist->Fit(func,"QNR")) { // N: do not blow up file by storing fit!
    mean  = func->GetParameter(1);
    sigma = func->GetParameter(2);
    // second fit: three sigma of first fit around mean of first fit
    func->SetRange(mean - 3.*sigma, mean + 3.*sigma);
    // I: integral gives more correct results if binning is too wide
    // L: Likelihood can treat empty bins correctly (if hist not weighted...)
    if (0 == hist->Fit(func, "Q0ILR")) {
      if (hist->GetFunction(func->GetName())) { // Take care that it is later on drawn:
	//hist->GetFunction(func->GetName())->ResetBit(TF1::kNotDraw);
      }
    }
  }
  return func;
}


//------------------------------------------------------------------------------
/*! \fn storeHistogramInRootfile
 *  \brief Store the histogram and the gaussian function resulting from the fitGauss function into a root file
 */
void PlotAlignmentValidation::storeHistogramInRootfile(TH1* hist)
{
  //Store histogram and fit function in the root summary file
  rootsummaryfile->cd();
  hist->Write();
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
void  PlotAlignmentValidation::setTitleStyle( TNamed &hist,const char* titleX, const char* titleY,int subDetId, bool isSurfaceDeformation, TString secondline)
{
  std::stringstream title_Xaxis;
  std::stringstream title_Yaxis;
  TString titleXAxis=titleX;
  TString titleYAxis=titleY;
  if (titleXAxis != "" && titleYAxis != "")
    cout<<"plot "<<titleXAxis<<" vs "<<titleYAxis<<endl;
  
  hist.SetTitle("");
  TkAlStyle::drawStandardTitle();

  //Thanks Candice!
  TString subD;
  switch (subDetId) {
    case 1: subD="BPIX"; break;
    case 2: subD="FPIX"; break;
    case 3: subD="TIB"; break;
    case 4: subD="TID"; break;
    case 5: subD="TOB"; break;
    case 6: subD="TEC"; break;
  }

  TPaveText *text2;
  if (!isSurfaceDeformation) {
    text2 = new TPaveText(0.7, 0.3, 0.9, 0.6, "brNDC");
  } else {
    cout << "Surface Deformation" << endl;
    text2 = new TPaveText(0.8, 0.75, 0.9, 0.9, "brNDC");
  }
  text2->SetTextSize(0.06);
  text2->SetTextFont(42);
  text2->SetFillStyle(0);
  text2->SetBorderSize(0);
  text2->SetMargin(0.01);
  text2->SetTextAlign(12); // align left
  text2->AddText(0.01,0.75,subD);
  if (secondline != "") {
    text2->AddText(0.01, 0.25, secondline);
  }
  text2->Draw();
}


//------------------------------------------------------------------------------
/*! \fn 
 *  \brief 
 */
void  PlotAlignmentValidation::setHistStyle( TH1& hist,const char* titleX, const char* titleY, int color)
{
  std::stringstream title_Xaxis;
  std::stringstream title_Yaxis;
  TString titleXAxis=titleX;
  TString titleYAxis=titleY;
  
  if ( titleXAxis.Contains("Phi") )title_Xaxis<<titleX<<"[rad]";
  else if( titleXAxis.Contains("meanX") )title_Xaxis<<"#LTx'_{pred}-x'_{hit}#GT[#mum]";
  else if( titleXAxis.Contains("meanY") )title_Xaxis<<"#LTy'_{pred}-y'_{hit}#GT[#mum]";
  else if( titleXAxis.Contains("rmsX") )title_Xaxis<<"RMS(x'_{pred}-x'_{hit})[#mum]";
  else if( titleXAxis.Contains("rmsY") )title_Xaxis<<"RMS(y'_{pred}-y'_{hit})[#mum]";
  else if( titleXAxis.Contains("meanNormX") )title_Xaxis<<"#LTx'_{pred}-x'_{hit}/#sigma#GT";
  else if( titleXAxis.Contains("meanNormY") )title_Xaxis<<"#LTy'_{pred}-y'_{hit}/#sigma#GT";
  else if( titleXAxis.Contains("rmsNormX") )title_Xaxis<<"RMS(x'_{pred}-x'_{hit}/#sigma)";
  else if( titleXAxis.Contains("rmsNormY") )title_Xaxis<<"RMS(y'_{pred}-y'_{hit}/#sigma)";
  else if( titleXAxis.Contains("meanLocalX") )title_Xaxis<<"#LTx_{pred}-x_{hit}#GT[#mum]";
  else if( titleXAxis.Contains("rmsLocalX") )title_Xaxis<<"RMS(x_{pred}-x_{hit})[#mum]";
  else if( titleXAxis.Contains("meanNormLocalX") )title_Xaxis<<"#LTx_{pred}-x_{hit}/#sigma#GT[#mum]";
  else if( titleXAxis.Contains("rmsNormLocalX") )title_Xaxis<<"RMS(x_{pred}-x_{hit}/#sigma)[#mum]";
  else if( titleXAxis.Contains("medianX") )title_Xaxis<<"median(x'_{pred}-x'_{hit})[#mum]";
  else if( titleXAxis.Contains("medianY") )title_Xaxis<<"median(y'_{pred}-y'_{hit})[#mum]";
  else title_Xaxis<<titleX<<"[cm]";
  
  if (hist.IsA()->InheritsFrom( TH1F::Class() ) )hist.SetLineColor(color);
  if (hist.IsA()->InheritsFrom( TProfile::Class() ) ) {
    hist.SetMarkerStyle(20);
    hist.SetMarkerSize(0.8);
    hist.SetMarkerColor(color);
  }
  
  hist.GetXaxis()->SetTitle( (title_Xaxis.str()).c_str() );

  double binning = (hist.GetXaxis()->GetXmax() - hist.GetXaxis()->GetXmin()) / hist.GetNbinsX();
  title_Yaxis.precision(2);

  if ( ((titleYAxis.Contains("layer") || titleYAxis.Contains("ring"))
                    && titleYAxis.Contains("subDetId"))
	      || titleYAxis.Contains("#modules")) {
    title_Yaxis<<"number of modules";
    if (TString(title_Xaxis.str()).Contains("[#mum]"))
      title_Yaxis << " / " << binning << " #mum";
    else if (TString(title_Xaxis.str()).Contains("[cm]"))
      title_Yaxis << " / " << binning << " cm";
    else
      title_Yaxis << " / " << binning;
  }
  else title_Yaxis<<titleY<<"[cm]";

  hist.GetYaxis()->SetTitle( (title_Yaxis.str()).c_str()  );

  hist.GetXaxis()->SetTitleFont(42);
  hist.GetYaxis()->SetTitleFont(42);
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

//------------------------------------------------------------------------------
void PlotAlignmentValidation::
setDMRHistStyleAndLegend(TH1F* h, PlotAlignmentValidation::DMRPlotInfo& plotinfo, int direction, int layer)
{
  TF1 *fitResults = 0;

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
  if (plotinfo.variable == "medianX" || plotinfo.variable == "meanX" || plotinfo.variable == "medianY" || plotinfo.variable == "meanY") {
    fitResults = fitGauss(h, linecolor );
  }
	  
  plotinfo.hstack->Add(h);

  std::ostringstream legend;
  legend.precision(3);
  legend << fixed; // to always show 3 decimals

  // Legend: header part
  if (direction == -1 && plotinfo.subDetId != 2) { legend << "rDirection < 0"; }
  else if (direction == 1 && plotinfo.subDetId != 2) { legend << "rDirection > 0"; }
  else if (direction == -1 && plotinfo.subDetId == 2) { legend << "zDirection < 0"; }
  else if (direction == 1 && plotinfo.subDetId == 2) { legend << "zDirection > 0"; }
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
  }

  plotinfo.legend->AddEntry(h, legend.str().c_str(), "l");
  legend.str("");

  // Legend: Statistics
  double mean, meanerror, rms, rmserror;
  TString rmsname, units;
  bool showdeltamu = (plotinfo.h1 != 0 && plotinfo.h2 != 0 && plotinfo.plotSplits && plotinfo.plotPlain && direction == 0);
  if (plotinfo.variable == "medianX" || plotinfo.variable == "meanX" ||
      plotinfo.variable == "medianY" || plotinfo.variable == "meanY" ||
      plotinfo.variable == "rmsX"    || plotinfo.variable == "rmsY") {
    if (useFit_ && fitResults) {
      mean = fitResults->GetParameter(1)*10000;
      meanerror = fitResults->GetParError(1)*10000;
      rms = fitResults->GetParameter(2)*10000;
      rmserror = fitResults->GetParError(2)*10000;
      rmsname = "#sigma";
      delete fitResults;
    } else {
      mean = h->GetMean(1)*10000;
      meanerror = h->GetMeanError(1)*10000;
      rms = h->GetRMS(1)*10000;
      rmserror = h->GetRMSError(1)*10000;
      rmsname = "rms";
    }
    units = " #mum";
  } else if (plotinfo.variable == "meanNormX" || plotinfo.variable == "meanNormY" ||
	     plotinfo.variable == "rmsNormX" || plotinfo.variable == "rmsNormY") {
    mean = h->GetMean(1);
    meanerror = h->GetMeanError(1);
    rms = h->GetRMS(1);
    rmserror = h->GetRMSError(1);
    rmsname = "rms";
    units = "";
  }
  if (showMean_)
  {
    legend << " #mu = " << mean;
    if (showMeanError_)
      legend << " #pm " << meanerror;
    legend << units;
    if (showRMS_ || showdeltamu || ((showModules_ || showUnderOverFlow_) && !twolines_))
      legend << ", ";
  }
  if (showRMS_)
  {
    legend << " " << rmsname << " = " << rms;
    if (showRMSError_)
      legend << " #pm " << rmserror;
    legend << units;
    if (showdeltamu || ((showModules_ || showUnderOverFlow_) && !twolines_))
      legend << ", ";
  }

  if ((plotinfo.variable == "medianX" || plotinfo.variable == "medianY") && /*!plotinfo.plotLayers && */layer==0 && direction==0) {
    vmean.push_back(mean);
    vrms.push_back(rms);
    vmeanerror.push_back(meanerror);
  TH1F* ideal = (TH1F*)plotinfo.hstack->GetHists()->At(0);
  TH1F* h = plotinfo.h;
  if(h->GetRMS() >= ideal->GetRMS())
  {
  vAlignmentUncertainty.push_back(sqrt(pow(h->GetRMS(),2)-pow(ideal->GetRMS(),2)));
  }
  else{
  vAlignmentUncertainty.push_back(nan(""));
  }
  float p = (float)resampleTestOfEqualMeans(ideal, h, 10000);
  vPValueMeanEqualIdeal.push_back(p);
  p=resampleTestOfEqualRMS(ideal, h, 10000);
  vPValueRMSEqualIdeal.push_back(p);

  }

  // Legend: Delta mu for split plots
  if (showdeltamu) {
    float factor = 10000.0f;
    if (plotinfo.variable == "meanNormX" || plotinfo.variable == "meanNormY" ||
	plotinfo.variable == "rmsNormX" || plotinfo.variable == "rmsNormY") {
      factor = 1.0f;
    }
    float deltamu = factor*(plotinfo.h2->GetMean(1) - plotinfo.h1->GetMean(1));
    legend << "#Delta#mu = " << deltamu << units;
    if ((showModules_ || showUnderOverFlow_) && !twolines_)
      legend << ", ";

    if ((plotinfo.variable == "medianX" || plotinfo.variable == "medianY") && /*!plotinfo.plotLayers && */layer==0 && direction==0) {
      vdeltamean.push_back(deltamu);
      if(plotinfo.h1->GetEntries()&&plotinfo.h2->GetEntries()){
          float p = (float)resampleTestOfEqualMeans(plotinfo.h1,plotinfo.h2, 10000);
          vPValueEqualSplitMeans.push_back(p);
          
      }
    }
  }

  if (twolines_) {
    plotinfo.legend->AddEntry((TObject*)0, legend.str().c_str(), "");
    plotinfo.legend->AddEntry((TObject*)0, "", "");
    legend.str("");
  }

  if (!showUnderOverFlow_ && showModules_) {
    legend << (int) h->GetEntries() << " modules";
  }
  if (showUnderOverFlow_) {
    if (showModules_) {
      legend << (int) h->GetEntries() << " modules (" << (int) h->GetBinContent(0) + (int)h->GetBinContent(h->GetNbinsX()+1) << " outside range)";
    } else {
      legend << (int) h->GetBinContent(0) + (int)h->GetBinContent(h->GetNbinsX()+1) << " modules outside range";
    }
  }
  plotinfo.legend->AddEntry((TObject*)0, legend.str().c_str(), "");

  // Scale the x-axis (cm to um), if needed
  if (plotinfo.variable.find("Norm") == std::string::npos)
    scaleXaxis(h, 10000);

}

/*!
 * \fn plotDMRHistogram 
 * \brief Create the DMR histrogram using data stored in trees and store them in the plotinfo structure.
 */

void PlotAlignmentValidation::
plotDMRHistogram(PlotAlignmentValidation::DMRPlotInfo& plotinfo, int direction, int layer, std::string subdet)
{
  TH1F* h = 0;
  //Create a name for the histogram that summarize all relevant information: name of the geometry, variable plotted, structure, layer, and whether the modules considered point inward or outward.

  TString histoname="";
  if(plotinfo.variable == "medianX" || plotinfo.variable == "medianY" )histoname="median";
  else if(plotinfo.variable == "rmsNormX" || plotinfo.variable == "rmsNormY")histoname="DrmsNR";
  histoname+="_";  histoname+=plotinfo.vars->getName();
  histoname.ReplaceAll(" ","_");
  histoname+="_";  histoname+=subdet.c_str();
  if (plotinfo.variable == "medianY" || plotinfo.variable == "rmsNormY")histoname+="_y";
  if(layer!=0){
    if(subdet=="TID"||subdet=="TEC")histoname+="_disc";
    else histoname+="_layer";
    histoname+=to_string(layer);
  }
  if (direction == -1) { histoname += "_minus"; }
  else if (direction == 1) { histoname += "_plus"; }
  else { histoname += ""; }
  std::string plotVariable = getVariableForDMRPlot(histoname.Data(), plotinfo.variable, plotinfo.nbins, plotinfo.min, plotinfo.max);
  std::string selection = getSelectionForDMRPlot(plotinfo.minHits, plotinfo.subDetId, direction, layer);
  plotinfo.vars->getTree()->Draw(plotVariable.c_str(), selection.c_str(), "goff");
  if (gDirectory) gDirectory->GetObject(histoname.Data(), h);
  if (h && h->GetEntries() > 0) {
    if (direction == -1) { plotinfo.h1 = h; }
    else if (direction == 1) { plotinfo.h2 = h; }
    else { plotinfo.h = h; }
  }
  if(plotinfo.variable == "medianX" || plotinfo.variable == "medianY" || plotinfo.variable == "rmsNormX" || plotinfo.variable == "rmsNormY")
    storeHistogramInRootfile(h);

}

void PlotAlignmentValidation::modifySSHistAndLegend(THStack* hs, TLegend* legend)
{
  // Add mean-y-values to the legend and scale the histograms.

  Double_t legendY = 0.80;
  bool hasheader = (TkAlStyle::legendheader != "");
  if (hasheader) legend->SetHeader(TkAlStyle::legendheader);
  legend->SetFillStyle(0);
  int legendsize = hs->GetHists()->GetSize() + hasheader;

  if (legendsize > 3)
    legendY -= 0.01 * (legendsize - 3);
  if (bigtext_) { legendY -= 0.05; }
  if (legendY < 0.6) {
    std::cerr << "Warning: Huge legend!" << std::endl;
    legendY = 0.6;
  }
  legend->SetY1(legendY);
  if (bigtext_) legend->SetTextSize(TkAlStyle::textSize);

  // Loop over all profiles
  TProfile* prof = 0;
  TIter next(hs->GetHists());
  Int_t index = hasheader;  //if hasheader, first entry is the header itself
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


//random variable: \sigma_{X_1}-\sigma_{X_2}-\delta_{RMS}
//is centered approx around 0
//null hypothesis: \delta_{RMS}=0
//so \delta_\sigma is a realization of this random variable
//how probable is it to get our value of \delta_\sigma?
//->p-value
double PlotAlignmentValidation::resampleTestOfEqualRMS(TH1F* h1, TH1F* h2, int numSamples){
//vector to store realizations of random variable
    vector<double> diff;
    diff.clear();
//"true" (in bootstrap terms) difference of the samples' RMS
    double rmsdiff = abs(h1->GetRMS()-h2->GetRMS());
//means of the samples to calculate RMS
    double m1 = h1->GetMean();
    double m2 = h2->GetMean();
//realization of random variable
    double d1 = 0;
    double d2 = 0;
//mean of random variable
    double test_mean=0;
    for(int i=0;i<numSamples;i++){
        d1=0;
        d2=0;
        for(int i=0;i<h1->GetEntries();i++){
            d1+=h1->GetRandom()-m1;
        }
        for(int i=0;i<h2->GetEntries();i++){
            d2+=h2->GetRandom()+m2;
        }
        d1/=h1->GetEntries();
        d2/=h2->GetEntries();
        diff.push_back(abs(d1-d2-rmsdiff));
        test_mean+=abs(d1-d2-rmsdiff);
    }
    test_mean/=numSamples;
//p value
    double p=0;
    for(double d:diff){
        if(d>rmsdiff){
            p+=1;
        }
    }
    
    p/=numSamples;
    return p;
}



//random variable: (\overline{X_1}-\mu_1)-(\overline{X_2}-\mu_2)
//is centered approx around 0
//null hypothesis: \mu_1-\mu_2=0
//so \delta_\mu is a realization of this random variable
//how probable is it to get our value of \delta_\mu?
//->p-value
double PlotAlignmentValidation::resampleTestOfEqualMeans(TH1F* h1, TH1F* h2, int numSamples){
//vector to store realization of random variable
    vector<double> diff;
    diff.clear();
//"true" (in bootstrap terms) difference of the samples' means
    double meandiff = abs(h1->GetMean()-h2->GetMean());
//realization of random variable
    double d1 = 0;
    double d2=0;
//mean of random variable
    double test_mean=0;
    for(int i=0;i<numSamples;i++){
        d1=0;
        d2=0;
        for(int i=0;i<h1->GetEntries();i++){
            d1+=h1->GetRandom();
        }
        for(int i=0;i<h2->GetEntries();i++){
            d2+=h2->GetRandom();
        }
        d1/=h1->GetEntries();
        d2/=h2->GetEntries();
        diff.push_back(abs(d1-d2-meandiff));
        test_mean+=abs(d1-d2-meandiff);
    }
    test_mean/=numSamples;
//p-value
    double p=0;
    for(double d:diff){
        if(d>meandiff){
            p+=1;
        }
    }
    
    p/=numSamples;
    return p;
}



float PlotAlignmentValidation::twotailedStudentTTestEqualMean(float t, float v){
return 2*(1-ROOT::Math::tdistribution_cdf(abs(t),v));
}

const TString PlotAlignmentValidation::summaryfilename = "OfflineValidationSummary";



vector <TH1*>  PlotAlignmentValidation::findmodule (TFile* f, unsigned int moduleid){
		
		
		//TFile *f = TFile::Open(filename, "READ");
		TString histnamex;
		TString histnamey;
        //read necessary branch/folder
        auto t = (TTree*)f->Get("TrackerOfflineValidationStandalone/TkOffVal");

        TkOffTreeVariables *variables=0;
        t->SetBranchAddress("TkOffTreeVariables", &variables);
        unsigned int number_of_entries=t->GetEntries();
        for (unsigned int i=0;i<number_of_entries;i++){
                t->GetEntry(i);
                 if (variables->moduleId==moduleid){
                        histnamex=variables->histNameX;
                        histnamey=variables->histNameY;
						break;
                        }
        }
		 
	vector <TH1*> h;
		
        auto h1 = (TH1*)f->FindObjectAny(histnamex);
	auto h2 = (TH1*)f->FindObjectAny(histnamey);
        
	h1->SetDirectory(0);
	h2->SetDirectory(0);
        
	h.push_back(h1);
	h.push_back(h2);
		
	return h;
 }

void PlotAlignmentValidation::residual_by_moduleID( unsigned int moduleid){
	TCanvas *cx = new TCanvas("x_residual");
	TCanvas *cy = new TCanvas("y_residual");
	TLegend *legendx =new TLegend(0.55, 0.7, 1, 0.9);
        TLegend *legendy =new TLegend(0.55, 0.7, 1, 0.9);
	
    	legendx->SetTextSize(0.016);
	legendx->SetTextAlign(12);
        legendy->SetTextSize(0.016);
        legendy->SetTextAlign(12);

	
	
	
	for (auto it : sourceList) {
		TFile* file = it->getFile();
		int color = it->getLineColor();
		int linestyle = it->getLineStyle();   //this you set by doing h->SetLineStyle(linestyle)
		TString legendname = it->getName(); //this goes in the legend
	    	vector<TH1*> hist = findmodule(file, moduleid);
			
		TString histnamex = legendname+" NEntries: "+to_string(int(hist[0]->GetEntries()));
        	hist[0]->SetTitle(histnamex);
        	hist[0]->SetStats(0);
        	hist[0]->Rebin(50);
        	hist[0]->SetBit(TH1::kNoTitle);
        	hist[0]->SetLineColor(color);
        	hist[0]->SetLineStyle(linestyle);
        	cx->cd();
		hist[0]->Draw("Same");
		legendx->AddEntry(hist[0], histnamex, "l");
				
			
		TString histnamey = legendname+" NEntries: "+to_string(int(hist[1]->GetEntries()));
        	hist[1]->SetTitle(histnamey);
        	hist[1]->SetStats(0);
        	hist[1]->Rebin(50);
        	hist[1]->SetBit(TH1::kNoTitle);
        	hist[1]->SetLineColor(color);
        	hist[1]->SetLineStyle(linestyle);
        	cy->cd();
		hist[1]->Draw("Same");
		legendy->AddEntry(hist[1], histnamey, "l");
	
	}
	
	TString filenamex = "x_residual_"+to_string(moduleid);
        TString filenamey = "y_residual_"+to_string(moduleid);
        cx->cd();
	legendx->Draw();
        cx->SaveAs(outputDir + "/" +filenamex+".root");
        cx->SaveAs(outputDir + "/" +filenamex+".pdf");
        cx->SaveAs(outputDir + "/" +filenamex+".png");
        cx->SaveAs(outputDir + "/" +filenamex+".eps");

        cy->cd();
	legendy->Draw();
        cy->SaveAs(outputDir + "/" +filenamey+".root");
        cy->SaveAs(outputDir + "/" +filenamey+".pdf");
        cy->SaveAs(outputDir + "/" +filenamey+".png");
        cy->SaveAs(outputDir + "/" +filenamey+".eps");
	

}
