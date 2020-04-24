/// saveDQMHistograms.C
/// This is a simple macro to make plots of the histograms
/// produced by the hltExoticaPostProcessor_cfg.py file
/// Author: Thiago R. F. P. Tomei

#include "TFile.h"
#include "TH1.h"
#include "TString.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TKey.h"
#include "TList.h"
#include "TSystem.h"

// *******************************************
// Variables

TString drawOptions1D("");   // Drawing options for 
                             // 1D histograms.
TString drawOptions2D("box");// Drawing options for
                             // 2D histograms.

int lineColor = 1; // 1 = Black, 2 = Red, 3 = Green, 4 = Blue
int lineWidth = 2; // Line width for 1D histograms.
int rebinFactor = 1; // Rebinning factor

double histoxmin = 0.0;
double histoxmax = 1.2;
double titleSize = 0.03;
double titleOffset = 1.5;

Bool_t displayStatsBox = 0;  // 0 = false, 1 = true
Bool_t autoLogYaxis    = 0;
Bool_t forceLogYaxis   = 0;

// End of Variables
// *******************************************

void saveDQMHistograms(const TString fileName="histos.root", 
		       TString imageType="pdf",
		       double outputWidth=600, 
		       double outputHeight=600)
{
  TString outputFolder = fileName;
  TFile* fin = new TFile(fileName.Data()) ;

  TCanvas* canvasDefault;
  TString outputType = "."+imageType;
  TH1* h = 0;

  if (!fin->IsOpen()) {
    printf("<E> Cannot open input file %s\n",fileName.Data()) ;
    exit(1) ;
  }
  
  outputFolder = fileName+"/"; // Blank to use current directory,
                                          // or, for a specific dir type
                                          // something like "images/"
  outputFolder.ReplaceAll(".root","");
  outputFolder.ReplaceAll("__","_"); // Just a bit of cleanup here
  gSystem->MakeDirectory(outputFolder);

  canvasDefault = new TCanvas("canvasDefault","testCanvas",outputWidth,outputHeight);
  canvasDefault->SetGridx();
  canvasDefault->SetGridy();
  
  // Change settings on plotting histograms
  gStyle->SetOptStat(111111);  // This will cause overflow and underflow to be shown
  gStyle->SetHistLineWidth(lineWidth);
  gStyle->SetHistLineColor(lineColor);
  gStyle->SetTitleSize(titleSize,"X");
  gStyle->SetTitleSize(titleSize,"Y");
  gStyle->SetTitleXOffset(titleOffset);
  gStyle->SetTitleYOffset(titleOffset);

  // FIXME - this should also be configurable... right?
  fin->cd("DQMData/Run 1/HLT/Run summary/Exotica");
  TDirectory* baseDir = gDirectory;
  TDirectory* subDir;

  TList* thelist = baseDir->GetListOfKeys() ;
  if (!thelist) { printf("<E> No keys found in file\n") ; exit(1) ; }

  TIter next(thelist) ;
  TKey* key ;
  TObject* obj ;
      
  while ( (key = (TKey*)next()) ) {
    obj = key->ReadObj() ;
    printf("%s\n",obj->IsA()->GetName());

    if (strcmp(obj->IsA()->GetName(),"TDirectoryFile")==0) {

      // Okay, it is a directory, let's make a subfolder
      // and them loop over it
      printf("<W> Found subdirectory %s\n",obj->GetName()) ;
      TString path = outputFolder+"/"+obj->GetName();
      gSystem->MakeDirectory(outputFolder+"/"+obj->GetName());
      subDir = (TDirectory*)obj;
      TList* thesublist = subDir->GetListOfKeys() ;
      TIter subnext(thesublist) ;
      
      while ( (key = (TKey*)subnext()) ) {
	obj = key->ReadObj();
	
	if ( (strcmp(obj->IsA()->GetName(),"TProfile")==0)
	     || (!obj->InheritsFrom("TH2"))
	     || (!obj->InheritsFrom("TH1")) 
	     ) {
	  
	  // Okay, it is a histogram, let's draw it
	  printf("Histo name:%s title:%s\n",obj->GetName(),obj->GetTitle());
	  h = (TH1*)obj;

	  // But only if it is an efficiency
	  TString histName = h->GetName(); 
	  if(!histName.Contains("Eff")) continue;
	  
	  h->SetStats(displayStatsBox);
	  if(rebinFactor!=1)
	    h->Rebin(rebinFactor);
	  
	  ////////////////////////////////////////////////////////
	  // A trick to see if we want logscale in y-axis or not
	  if (autoLogYaxis) {
	    Double_t testYvalue = h->GetMaximum();
	    //cout << testYvalue << endl;
	    
	    if (testYvalue > 1.0) {
	      Double_t maxy = log10(testYvalue);
	      Double_t miny = log10(h->GetMinimum(1.0));
		
	      // log scale if more than 2 powers of 10 between low and high bins
	      if ( (maxy-miny) > 2.0 ) {
		canvasDefault->SetLogy(1);
	      }
	    }
	  }
	  
	  // or, alternatively, do it unconditionally.
	  if (forceLogYaxis) {
	    canvasDefault->SetLogy(1);
	  } 
	  // End of log or no-log y axis decision
	  ////////////////////////////////////////////////////////

	  h->Draw(drawOptions1D);
	  h->GetYaxis()->SetRangeUser(histoxmin, histoxmax);
	  canvasDefault->Modified();
	  canvasDefault->Update();
	  
	  canvasDefault->Print(path+"/"+histName+outputType);
	  canvasDefault->SetLogy(0); // reset to no-log - prevents errors

	} // Closes "if is a histogram" conditional
      } // Closes loop in keys of subdir
      
    } // Closes "if is a subdir" conditional
  } // Closes loop in base dir
  
  fin->Close();
}
