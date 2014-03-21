/**
 * This macro draws the resolutions for single muons quantities: pt, cotgTheta and phi.
 *
 * It does a rebin(4) if NbinsX > 50 and in the case of PtGenVsMu_ResoVSPt also a rebin(8) in y. <br>
 * It takes a new histogram (a TH1D) equal to the projection in X of the starting histogram (a TH2F) and it empties it
 * (so as to have the binning and the axis already set and an empty histogram). <br>
 * Takes also a profileX of the TH2F. <br>
 * For the fit in eta it takes the events with eta < 0 on those with eta > 0. <br>
 * In any case it takes the projection in y (ProjectionY) of the TH2F in each bin (from x to x, that is a single bin). <br>
 * It extracts the rms and its error from the gaussian fit and writes them in the TH1D described above.
 */

// Needed to use gROOT in a compiled macro
#include "TROOT.h"
#include "TStyle.h"

#include <string>
#include <vector>
#include <map>
#include <sstream>
#include "TChain.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TTree.h"
#include "TKey.h"
#include "Riostream.h"
#include "THStack.h"
#include "TCanvas.h"
#include "TF1.h"

// Use this with the ResolutionAnalyzer files
// TString mainNamePt("PtResolutionGenVSMu");
// TString mainNameCotgTheta("CotgThetaResolutionGenVSMu");
// TString mainNamePhi("PhiResolutionGenVSMu");
TString mainNamePt("hResolPtGenVSMu");
TString mainNameCotgTheta("hResolCotgThetaGenVSMu");
TString mainNamePhi("hResolPhiGenVSMu");

void draw( TDirectory *target, TList *sourcelist, const std::vector<TString> & vecNames,
	   const bool doHalfEta, const int minEntries = 100, const int rebinX = 0, const int rebinY = 0 );

void ResolDraw(const TString numString = "0", const bool doHalfEta = false,
	       const int minEntries = 100, const int rebinX = 0, const int rebinY = 0)
{
  // in an interactive ROOT session, edit the file names
  // Target and FileList, then
  // root > .L hadd.C
  // root > hadd()

  TList *FileList;
  // vector of names of the histograms to fit
  std::vector<TString> vecNames;

  TFile *Target;

  vecNames.push_back("DeltaMassOverGenMassVsPt");
  vecNames.push_back("DeltaMassOverGenMassVsEta");

  vecNames.push_back(mainNamePt + "_ResoVSPt");
  vecNames.push_back(mainNamePt + "_ResoVSPt_Bar");
  vecNames.push_back(mainNamePt + "_ResoVSPt_Endc_1.7");
  vecNames.push_back(mainNamePt + "_ResoVSPt_Endc_2.0");
  vecNames.push_back(mainNamePt + "_ResoVSPt_Endc_2.4");
  vecNames.push_back(mainNamePt + "_ResoVSPt_Ovlap");
  vecNames.push_back(mainNamePt + "_ResoVSEta");
  vecNames.push_back(mainNamePt + "_ResoVSPhiMinus");
  vecNames.push_back(mainNamePt + "_ResoVSPhiPlus");

  vecNames.push_back(mainNameCotgTheta + "_ResoVSPt");
  vecNames.push_back(mainNameCotgTheta + "_ResoVSEta");
  vecNames.push_back(mainNameCotgTheta + "_ResoVSPhiMinus");
  vecNames.push_back(mainNameCotgTheta + "_ResoVSPhiPlus");

  vecNames.push_back(mainNamePhi + "_ResoVSPt");
  vecNames.push_back(mainNamePhi + "_ResoVSEta");
  vecNames.push_back(mainNamePhi + "_ResoVSPhiMinus");
  vecNames.push_back(mainNamePhi + "_ResoVSPhiPlus");

  gROOT->SetBatch(true);
//   gROOT->SetStyle("Plain");
//   gStyle->SetCanvasColor(kWhite);
//   gStyle->SetCanvasBorderMode(0);
//   gStyle->SetPadBorderMode(0);
//   gStyle->SetTitleFillColor(kWhite);
//   gStyle->SetTitleColor(kWhite);
//   gStyle->SetOptStat("nemruoi");

  Target = TFile::Open( "redrawed_"+numString+".root", "RECREATE" );

  FileList = new TList();

  // ************************************************************
  // List of Files
  FileList->Add( TFile::Open(numString+"_MuScleFit.root") );    // 1

  draw( Target, FileList, vecNames, doHalfEta, minEntries, rebinX, rebinY );

  Target->Close();
}

void draw( TDirectory *target, TList *sourcelist, const std::vector<TString> & vecNames,
	   const bool doHalfEta, const int minEntries, const int rebinX, const int rebinY )
{
  //  std::cout << "Target path: " << target->GetPath() << std::endl;
  TString path( (char*)strstr( target->GetPath(), ":" ) );
  path.Remove( 0, 2 );

  TFile *first_source = (TFile*)sourcelist->First();

  // Stores all the fits
  std::map<TString, std::vector<TH1D*> > fitHistograms;

  first_source->cd( path );
  TDirectory *current_sourcedir = gDirectory;
  //gain time, do not add the objects in the list in memory
  Bool_t status = TH1::AddDirectoryStatus();
  TH1::AddDirectory(kFALSE);

  // loop over all keys in this directory
  TIter nextkey( current_sourcedir->GetListOfKeys() );
  TKey *key, *oldkey=0;
  while ( (key = (TKey*)nextkey())) {

    //keep only the highest cycle number for each key
    if (oldkey && !strcmp(oldkey->GetName(),key->GetName())) continue;

    // read object from first source file
    first_source->cd( path );
    TObject *obj = key->ReadObj();

    if ( obj->IsA()->InheritsFrom( "TH1" ) ) {
      // descendant of TH1 -> redraw it

      TString objName = obj->GetName();
      std::vector<TString>::const_iterator namesIt = vecNames.begin();
      for( ; namesIt != vecNames.end(); ++namesIt ) {
        if( *namesIt == objName ) {
          std::cout << "found histogram: " << *namesIt << std::endl;

          TDirectory * fits = (TDirectory*) target->Get("fits");
          if( fits == 0 ) fits = target->mkdir("fits");

          // Perform different fits for different profiles
          TH2F *h2 = (TH2F*)obj;

	  // Rebin X
          int xBins = h2->GetNbinsX();
	  if( rebinX != 0 ) {
            h2->RebinX(rebinX);
            xBins /= rebinX;
	  }
          else if( xBins > 500 ) {
            h2->RebinX(2);
            xBins /= 2;
          }
	  // Rebin Y
	  if( rebinY != 0 ) {
            h2->RebinY(rebinY);
	  }

          TH1D * h1 = h2->ProjectionX();
          TH1D * h1RMS = h2->ProjectionX();
          // h1->Clear();
          h1->Reset();
          h1->SetName(*namesIt+"_resol");
          h1RMS->Reset();
          h1RMS->SetName(*namesIt+"_resolRMS");
          TProfile * profile = h2->ProfileX();

          // This is used to fit half eta, needed to get the resolution function by value
          // with greater precision assuming symmetry in eta and putting all values in
          // -3,0 (as if -fabs(eta) is used).
          if( *namesIt == mainNamePt+"_ResoVSEta" && doHalfEta ) {
            std::cout << mainNamePt+"_ResoVSEta" << std::endl;
            std::cout << "bins%2 = " << xBins%2 << std::endl;
            for( int x=1; x<=xBins/2; ++x ) {
              std::stringstream fitNum;
              fitNum << x;
              TString fitName(*namesIt);
              fitName += "_fit_"; 
              TH1D * temp = h2->ProjectionY(fitName+fitNum.str(),x,x);
              TH1D * temp2 = h2->ProjectionY(fitName+fitNum.str(),xBins+1-x,xBins+1-x);
              temp->Add(temp2);
              if( temp->GetEntries() > minEntries ) {
                fitHistograms[*namesIt].push_back(temp);
                temp->Fit("gaus");
                double sigma = temp->GetFunction("gaus")->GetParameter(2);
                double sigmaError = temp->GetFunction("gaus")->GetParError(2);
                double rms = temp->GetRMS();
                double rmsError = temp->GetRMSError();
                // std::cout << "sigma = " << rms << std::endl;
                // std::cout << "sigma error = " << rmsError << std::endl;
                // std::cout << "rms = " << rms << std::endl;
                // std::cout << "rms error = " << rmsError << std::endl;
                // Reverse x in the first half to the second half.
                int xToFill = x;
                // Bin 0 corresponds to bin=binNumber(the last bin, which is also considered in the loop).
                if( *namesIt == mainNamePt+"_ResoVSEta" ) {
                  std::cout << mainNamePt+"_ResoVSEta" << std::endl;
                  if( x<xBins/2+1 ) xToFill = xBins+1 - x;
                }
                // std::cout << "x = " << x << ", xToFill = " << xToFill << std::endl;
                // std::cout << "rms = " << rms << ", rmsError = " << rmsError << std::endl;
                h1->SetBinContent(x, sigma);
                h1->SetBinError(x, sigmaError);
                h1->SetBinContent(xBins+1-x, 0);
                h1->SetBinError(xBins+1-x, 0);

                h1RMS->SetBinContent(x, rms);
                h1RMS->SetBinError(x, rmsError);
                h1RMS->SetBinContent(xBins+1-x, 0);
                h1RMS->SetBinError(xBins+1-x, 0);
                // h2->ProjectionY("_px",x,x)->Write();
                fits->cd();
                TCanvas * canvas = new TCanvas(temp->GetName()+TString("_canvas"), temp->GetTitle(), 1000, 800);
                temp->Draw();
                TF1 * gaussian = temp->GetFunction("gaus");
                gaussian->SetLineColor(kRed);
                gaussian->Draw("same");
                //canvas->Write();
                temp->Write();
              }
            }
            target->cd();
            profile->Write();
            // for ( int i=1; i<=h1->GetNbinsX(); ++i ) {
            //   std::cout << "bin["<<i<<"] = " << h1->GetBinContent(i) << std::endl;
            // }
            h1->Write();
            h1RMS->Write();
          }
          else {
            for( int x=1; x<=xBins; ++x ) {
              std::stringstream fitNum;
              fitNum << x;
              TString fitName(*namesIt);
              fitName += "_fit_";
              TH1D * temp = h2->ProjectionY(fitName+fitNum.str(),x,x);
              if( temp->GetEntries() > minEntries ) {
                fitHistograms[*namesIt].push_back(temp);
                temp->Fit("gaus");

                // double rms = temp->GetRMS();
                // double rmsError = temp->GetRMSError();

                double sigma = temp->GetFunction("gaus")->GetParameter(2);
                double sigmaError = temp->GetFunction("gaus")->GetParError(2);
                double rms = temp->GetRMS();
                double rmsError = temp->GetRMSError();
                if( sigma != sigma ) std::cout << "value is NaN: rms = " << rms << std::endl; 
                if( sigma == sigma ) {

                  std::cout << "rms = " << rms << std::endl;
                  std::cout << "rms error = " << rmsError << std::endl;

                  // NaN is the only value different from itself. Infact NaN is "not a number"
                  // and it is not equal to any value, including itself.
                  h1->SetBinContent(x, sigma);
                  h1->SetBinError(x, sigmaError);
                  h1RMS->SetBinContent(x, rms);
                  h1RMS->SetBinError(x, rmsError);
                }
                // h2->ProjectionY("_px",x,x)->Write();
                fits->cd();
                TCanvas * canvas = new TCanvas(temp->GetName()+TString("_canvas"), temp->GetTitle(), 1000, 800);
                temp->Draw();
                TF1 * gaussian = temp->GetFunction("gaus");
                gaussian->SetLineColor(kRed);
                gaussian->Draw("same");
                // canvas->Write();
                temp->Write();
              }
            }
            target->cd();
            profile->Write();
            h1->Write();
            h1RMS->Write();
          }
        }
      }
    }
    else if ( obj->IsA()->InheritsFrom( "TDirectory" ) ) {
      // it's a subdirectory

      // std::cout << "Found subdirectory " << obj->GetName() << std::endl;

      // create a new subdir of same name and title in the target file
      target->cd();
      TDirectory *newdir = target->mkdir( obj->GetName(), obj->GetTitle() );

      // newdir is now the starting point of another round of merging
      // newdir still knows its depth within the target file via
      // GetPath(), so we can still figure out where we are in the recursion
      draw( newdir, sourcelist, vecNames, doHalfEta, minEntries, rebinX, rebinY );
    }
    else {
      // object is of no type that we know or can handle
      // std::cout << "Unknown object type, name: "
      //      << obj->GetName() << " title: " << obj->GetTitle() << std::endl;
    }

    // now write the compared histograms (which are "in" obj) to the target file
    // note that this will just store obj in the current directory level,
    // which is not persistent until the complete directory itself is stored
    // by "target->Write()" below
    if ( obj ) {
      target->cd();

      if( obj->IsA()->InheritsFrom( "TH1" ) ) {
        // Write the superimposed histograms to file
        // obj->Write( key->GetName() );
      }
    }

  } // while ( ( TKey *key = (TKey*)nextkey() ) )

  // Save canvases of the fitted histograms
  std::map<TString, std::vector<TH1D*> >::const_iterator mapIt = fitHistograms.begin();
  for( ; mapIt != fitHistograms.end(); ++mapIt ) {
    TCanvas * canvas = new TCanvas(mapIt->first, mapIt->first, 1000, 800);

    int sizeCheck = mapIt->second.size();
    int x = int(sqrt(sizeCheck));
    int y = x;
    if( x*y < sizeCheck ) y += 1;
    if( x*y < sizeCheck ) x += 1;
    canvas->Divide(x,y);
    std::vector<TH1D*>::const_iterator histoIt = mapIt->second.begin();
    int histoNum = 1;
    for( ; histoIt != mapIt->second.end(); ++histoIt, ++histoNum ) {
      canvas->cd(histoNum);
      (*histoIt)->Draw();
    }
    canvas->Write();
  }

  // save modifications to target file
  target->SaveSelf(kTRUE);
  TH1::AddDirectory(status);
}
