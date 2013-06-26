/*************************************************
  Drawing plots and saving images of histograms
  can be tedious. This automates the task.
  
  Finds all the directories in a given ROOT file
  and creates a directory structure exactly like 
  it and fills it with image files of hitograms 
  found.

  Can be run from a bash prompt as well:
    root -b -l -q "saveHistograms.C(\"aFile.root\",\"gif\",640,480)"

  Michael B. Anderson
  May 23, 2008
*************************************************/

#include <string.h>
#include "TChain.h"
#include "TFile.h"
#include "TH1.h"
#include "TTree.h"
#include "TKey.h"
#include "Riostream.h"

TFile *sourceFile;
TObject *obj;
TString outputFolder;
TH1 *h;
TCanvas *canvasDefault;

// *******************************************
// Variables

TString outputType;
Int_t outputXsize;
Int_t outputYsize;

TString drawOptions1D("");   // Drawing options for 
                             // 1D histograms.
TString drawOptions2D("box");// Drawing options for
                             // 2D histograms.

int lineColor = 4; // 2 = Red, 3 = Green, 4 = Blue
int lineWidth = 2; // Line width for 1D histograms.

Bool_t displayStatsBox = 1;  // 0 = false, 1 = true
Bool_t autoLabelXaxis  = 1;
Bool_t autoLogYaxis    = 0;
Bool_t printOutput     = 1;

// End of Variables
// *******************************************

void recurseOverKeys( TDirectory *target );

void saveHistograms(TString fileName,
	            TString imageType = "gif", 
	            int outputWidth  = 640, 
	            int outputHeight = 480) {

  sourceFile = TFile::Open( fileName );

  outputFolder = fileName+"/"; // Blank to use current directory,
                                          // or, for a specific dir type
                                          // something like "images/"
  outputFolder.ReplaceAll(".root","");
  gSystem->MakeDirectory(outputFolder);

  outputType = "."+imageType;
  outputXsize = outputWidth; 
  outputYsize = outputHeight;
  canvasDefault = new TCanvas("canvasDefault","testCanvas",outputWidth,outputHeight);

  // Change settings on plotting histograms
  gStyle->SetOptStat(111111);  // This will cause overflow and underflow to be shown
  gStyle->SetHistLineWidth(2); // Set the line width to 2 pixels
  //gStyle->SetTitleFontSize(0.035); // Shrink Histogram Title Size

  // Now actually find all the directories, histograms, and save them..
  recurseOverKeys(sourceFile);  

  sourceFile->Close();

  TString currentDir = gSystem->pwd();
  cout << "Done. See images in:" << endl << currentDir << "/" << outputFolder << endl;
}

void recurseOverKeys( TDirectory *target ) {
 
  TString path( (char*)strstr( target->GetPath(), ":" ) );
  path.Remove( 0, 2 );

  cout << path << endl;

  sourceFile->cd( path );
  TDirectory *current_sourcedir = gDirectory;

  TKey *key;
  TIter nextkey(current_sourcedir->GetListOfKeys());

  while (key = (TKey*)nextkey()) {

    obj = key->ReadObj();

    if (obj->IsA()->InheritsFrom("TH1")) {

      // **************************
      // Plot & Save this Histogram
      h = (TH1*)obj;
      h->SetStats(displayStatsBox);

      TString histName = h->GetName(); 


      ///////////////////////////////////////////
      // Special & optional drawing commands

      // Now to label the X-axis!
      if (autoLabelXaxis) {
	if ( histName.Contains("Phi") ) {
	  h->GetXaxis()->SetTitle("#phi");
	} else if ( histName.Contains("eta") || histName.Contains("eta2") ) {
	  h->GetXaxis()->SetTitle("#eta");
	} else if ( histName.Contains("Pt") ) {
	  h->GetXaxis()->SetTitle("p_{T} (GeV)");
	} else if ( histName.Contains("et") || histName.Contains("et2") ) {
	  h->GetXaxis()->SetTitle("E_{T} (GeV)");
	}
      }

      // Tricky work-around.
      //  Some plots have text labels that are too big.
      //  Alter a margin to keep them in the picture.
      if (histName.Contains("total ") || histName.Contains("efficiency by step")) {
        canvasDefault->SetBottomMargin(0.24);
	canvasDefault->SetRightMargin(0.15);
      } else {
	canvasDefault->SetBottomMargin(0.1);
	canvasDefault->SetRightMargin(0.1);
      }

      h->SetLineColor(lineColor);
      h->SetLineWidth(lineWidth);


      // ********************************
      // A trick to decide whether to have log or no-log y axis
      // get hist max y value
      if (autoLogYaxis) {
        Double_t testYvalue = h->GetMaximum();
        //cout << testYvalue << endl;

        if (testYvalue > 1.0) {
          Double_t maxy = log10(testYvalue);

          // get hist min y value
          Double_t miny = log10(h->GetMinimum(1.0));

          // log scale if more than 2 powers of 10 between low and high bins
          if ( (maxy-miny) > 2.0 ) {
            canvasDefault->SetLogy(1);
          }
        }
      }
      // End of log or no-log y axis decision
      // ********************************

      // END Special commands
      ///////////////////////////////////////////

      h->Draw(drawOptions1D);
      canvasDefault->Modified();
      canvasDefault->Update();
      
      canvasDefault->Print(outputFolder+path+"/"+histName+outputType);
      // To store the root file name in image file name:
      //canvasDefault->Print(outputFolder+histFileName+histName+outputType);
      if (printOutput) cout << outputFolder+path+"/"+histName+outputType << endl;

      canvasDefault->SetLogy(0); // reset to no-log - prevents errors
      // **************************

    } else if ( obj->IsA()->InheritsFrom( "TDirectory" ) ) {
      // it's a subdirectory

      cout << "Found subdirectory " << obj->GetName() << endl;
      gSystem->MakeDirectory(outputFolder+path+"/"+obj->GetName());

      // obj is now the starting point of another round of merging
      // obj still knows its depth within the target file via
      // GetPath(), so we can still figure out where we are in the recursion
      recurseOverKeys( (TDirectory*)obj );

    } // end of IF a TDriectory
  } // end of LOOP over keys
}
