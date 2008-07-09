#include <string.h>
#include <cstring>
#include "TChain.h"
#include "TFile.h"
#include "TH1.h"
#include "TTree.h"
#include "TKey.h"
#include "Riostream.h"
#include <vector>
#include <sstream>
#include "TCanvas.h"
#include "TLegend.h"
#include "TROOT.h"
#include "TPaveStats.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TStyle.h"
#include "TEnv.h"


TList *FileList;
TList *LabelList;
TFile *Target;
std::vector< std::string > lowestlevels;
std::vector< int > theColors;
std::vector< int > theStyles;

void MergeRootfile( TDirectory *target, TList *sourcelist, TList *labellist );
void nicePad(Int_t logx,Int_t logy);
void SetMinMaxRange(TObjArray *hists);

void ColourStatsBoxes(TObjArray *hists);

void compareAlignments(TString namesandlabels="readFromFile") 
{
  cout << "Comparing using: >"<<namesandlabels<<"<"<<endl;

  gStyle->SetOptStat(111110);
  gStyle->SetTitleFillColor(10);
  gStyle->SetTitleBorderSize(0);

  Target = TFile::Open( "result.root", "RECREATE" );
  FileList = new TList();
  LabelList = new TList();
  
  int formatCounter = 1;
  //TObjArray* stringarray = namesandlabels.Tokenize(",");  
  TObjArray *nameandlabelpairs = namesandlabels.Tokenize(",");
  for (Int_t i = 0; i < nameandlabelpairs->GetEntries(); ++i) {
    TObjArray *aFileLegPair = TString(nameandlabelpairs->At(i)->GetName()).Tokenize("=");
    
    if(aFileLegPair->GetEntries() == 2) {
      TFile* currentFile = TFile::Open(aFileLegPair->At(0)->GetName());
      if( currentFile != NULL && !currentFile->IsZombie() ){
	FileList->Add( currentFile  );  // 2
	if(TString(aFileLegPair->At(1)->GetName()).Contains("|")){
	  TObjArray* formatedLegendEntry = TString(aFileLegPair->At(1)->GetName()).Tokenize("|");
	  LabelList->Add( formatedLegendEntry->At(0) );
	  if(formatedLegendEntry->GetEntries() > 1){
	    theColors.push_back(atoi(formatedLegendEntry->At(1)->GetName()));
	    
	    if(formatedLegendEntry->GetEntries() > 2)
	      theStyles.push_back(atoi(formatedLegendEntry->At(2)->GetName()));
	    else 
	      theStyles.push_back( formatCounter );
	  }else{
	  std::cout <<"if you give a \"|\" in the legend name you will need to at least give a int for the color"<<std::endl;
	  }
	  formatCounter++;
	}else{
	  LabelList->Add( aFileLegPair->At(1) );
	  theColors.push_back(formatCounter);
	  theStyles.push_back(formatCounter);
	  formatCounter++;
	}
      }else{
	std::cout << "Could not open: "<<aFileLegPair->At(0)->GetName()<<std::endl;
      }
    }
    else {
      std::cout << "Please give file name and legend entry in the following form:\n" 
		<< " filename1=legendentry1,filename2=legendentry2[|color[|style]]"<<std::endl;

    }
    
  }

  // ************************************************************
  // List of Files
  //FileList->Add( TFile::Open("../test/AlignmentValidation_Elliptical.root") ); 
  //FileList->Add( TFile::Open("../test/AlignmentValidation_10pb.root")  );  // 2
  //FileList->Add( TFile::Open("../test/AlignmentValidation_custom.root")  );  // 2
  // ************************************************************

  // put here the lowest level up to which you want to combine the 
  // histogramms
  lowestlevels.push_back("TPBLadder");
  lowestlevels.push_back("TPEPanel");
  lowestlevels.push_back("TIBHalfShell");
  lowestlevels.push_back("TIDRing");
  lowestlevels.push_back("TOBRod");
  lowestlevels.push_back("TECSide");
//  lowestlevels.push_back("Det");
  
  
  
   MergeRootfile( Target, FileList, LabelList );

}

void MergeRootfile( TDirectory *target, TList *sourcelist, TList *labellist ) {

  if( sourcelist->GetSize() == 0){
    std::cout<< "Cowardly refuse to merge empty SourceList! " <<std::endl;
    return;
  }

  TString path( (char*)strstr( target->GetPath(), ":" ) );
  path.Remove( 0, 2 );

  TFile *first_source = (TFile*)sourcelist->First();
  TObjString *first_label = (TObjString*)labellist->First();

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

    if ( obj->IsA()->InheritsFrom( TH1::Class() ) ) {
      // descendant of TH1 -> merge it
      TCanvas c(obj->GetName(),obj->GetName(),500,500);
      c.SetFillColor(10);
      
      bool is2d = false;
      if(strstr(obj->ClassName() ,"TH2") != NULL )
	is2d = true;
      TH1 *h1 = static_cast<TH1*>(obj);

      int q = 1; 
      TObjArray *histarray = new TObjArray;
      
      h1->SetLineStyle(theStyles.at(q-1));
      h1->SetLineWidth(2);
   
      h1->SetLineColor(theColors.at(q-1));
      h1->GetYaxis()->SetTitleOffset(1.5);
      if(strstr(h1->GetName(),"summary") != NULL )
	h1->Draw("x0e1*H");
      else if(is2d)
	h1->Draw();
      else 
	h1->Draw();

      TLegend leg(0.2,0.85,0.775,0.93);
      leg.AddEntry(h1,first_label->String().Data(),"L");
      leg.SetBorderSize(0);
      leg.SetFillColor(10);
      // loop over all source files and add the content of the
      // correspondant histogram to the one pointed to by "h1"
      TFile *nextsource = (TFile*)sourcelist->After( first_source );
      TObjString *nextlabel = (TObjString*)labellist->After( labellist->First() );
      
      histarray->Add(h1);
      while ( nextsource ) {

        // make sure we are at the correct directory level by cd'ing to path
	
        nextsource->cd( path );
        TKey *key2 = (TKey*)gDirectory->GetListOfKeys()->FindObject(h1->GetName());
        if (key2) {
	  ++q;
	  TH1 *h2 = (TH1*)key2->ReadObj();	  

	  if(!is2d){
	    h2->SetLineStyle(theStyles.at(q-1));
	    h2->SetLineWidth(2);
	  }

	  h2->SetLineColor(theColors.at(q-1));
	  std::stringstream newname;
	  newname << h2->GetName() << q;
	  
	  h2->SetName(newname.str().c_str());
	  if(strstr(newname.str().c_str(),"summary") != NULL )	    
	    h2->DrawClone("x0*He1sames");
	  else if(is2d) 
	    h2->DrawClone("sames");
	  else
	    h2->DrawClone("sames");
	  leg.AddEntry(c.FindObject(h2->GetName()),nextlabel->String().Data(),"L");
	  histarray->Add(c.FindObject(h2->GetName()));
	  delete h2;	  
        } else {
	  std::cerr << "Histogram "<< key2->GetTitle() << " is not present in file " << nextsource->GetName() << std::endl;
	}
	
        nextsource = (TFile*)sourcelist->After( nextsource );
	nextlabel = (TObjString*)labellist->After(nextlabel);
      }
      nicePad(0,0);
      leg.Draw();
      c.Update();
      if(strstr(h1->GetName(),"summary") == NULL )
	SetMinMaxRange(histarray);
      ColourStatsBoxes(histarray);
      target->cd();
      c.Write();
      histarray->Delete();
      

      

    } else if ( obj->IsA()->InheritsFrom( TDirectory::Class() ) ) {
      // it's a subdirectory

      std::string dirname = obj->GetName();
      for( std::vector< std::string >::const_iterator lowlevelit = lowestlevels.begin(), 
	     lowlevelitend = lowestlevels.end(); lowlevelit != lowlevelitend; ++lowlevelit) 
	if(   dirname.find(*lowlevelit) != std::string::npos ) 
	  return;
	  
      // create a new subdir of same name and title in the target file
      target->cd();
      TDirectory *newdir = target->mkdir( obj->GetName(), obj->GetTitle() );
      
      // newdir is now the starting point of another round of merging
      // newdir still knows its depth within the target file via
      // GetPath(), so we can still figure out where we are in the recursion
      MergeRootfile( newdir, sourcelist, labellist );
      

    } else {

      // object is of no type that we know or can handle
      cout << "Unknown object type, name: "
           << obj->GetName() << " title: " << obj->GetTitle() << endl;
    }


  } // while ( ( TKey *key = (TKey*)nextkey() ) )

  // save modifications to target file
  target->SaveSelf(kTRUE);
  TH1::AddDirectory(status);
}




void nicePad(Int_t logx,Int_t logy)
{
    gPad->SetBottomMargin(0.10);
    gPad->SetRightMargin(0.1);
    gPad->SetLeftMargin(0.15);
    gPad->SetTopMargin(0.15);
    gPad->SetTickx(1);
    gPad->SetTicky(1);
    if(logy==1)
      {
        gPad->SetLogy();
      }
    else
      {
        gPad->SetLogy(0);
      }
    if(logx==1)
      {
        gPad->SetLogx();
      }
    else
      {
        gPad->SetLogx(0);
      }
}


void ColourStatsBoxes(TObjArray *hists) 
{

  Double_t fStatsX1 = 0.85, fStatsX2 = 1., fStatsY1 = 0.85, fStatsY2 = 1.;
  // colours stats boxes like hists' line colors and moves the next to each other
  if (!hists) return;
  Double_t x1 = fStatsX1, x2 = fStatsX2, y1 = fStatsY1, y2 = fStatsY2;
  for (Int_t iH = 0; iH < hists->GetEntries(); ++iH) {
    TH1 *h = static_cast<TH1*>(hists->At(iH));
    if (!h) continue;
    TObject *statObj = h->GetListOfFunctions()->FindObject("stats");
    if (statObj && statObj->InheritsFrom(TPaveStats::Class())) {
      TPaveStats *stats = static_cast<TPaveStats*>(statObj);
      stats->SetLineColor(static_cast<TH1*>(hists->At(iH))->GetLineColor());
      stats->SetTextColor(static_cast<TH1*>(hists->At(iH))->GetLineColor());
      stats->SetFillColor(10);
      stats->SetX1NDC(x1);
      stats->SetX2NDC(x2);
      stats->SetY1NDC(y1);
      stats->SetY2NDC(y2);
      y2 = y1 - 0.005; // shift down 2
      y1 = y2 - (fStatsY2 - fStatsY1); // shift down 1
      if (y1 < 0.) {
	y1 = fStatsY1; y2 = fStatsY2; // restart y-positions
	x2 = x1 - 0.005; // shift left 2
	x1 = x2 - (fStatsX2 - fStatsX1); // shift left 1
	if (x1 < 0.) { // give up, start again:
	  x1 = fStatsX1, x2 = fStatsX2, y1 = fStatsY1, y2 = fStatsY2;
	}
      }
      //} else if (gStyle->GetOptStat() != 0) { // failure in case changed in list via TExec....
      //this->Warning("ColourStatsBoxes", "No stats found for %s", hists->At(iH)->GetName());
    }
  }
}


void SetMinMaxRange(TObjArray *hists)
{
  Double_t min = 100000;
  Double_t max = -100000;
   for (Int_t iH = 0; iH < hists->GetEntries(); ++iH) {
     TH1 *h = static_cast<TH1*>(hists->At(iH));
     if (!h) continue;
     for(int i = 1; i <= h->GetNbinsX(); ++i) {
       if(h->GetBinContent(i) + h->GetBinError(i) > max ) max = h->GetBinContent(i) + h->GetBinError(i);
       if(h->GetBinContent(i) - h->GetBinError(i) < min ) min = h->GetBinContent(i) - h->GetBinError(i);
     }
   }

   TH1 *h_first = static_cast<TH1*>(hists->At(0));
   h_first->SetMaximum(max+max*0.1);
   if(min = 0.) {
     min = -1111;
     h_first->SetMinimum(min);
   } else {
     h_first->SetMinimum(min-min*0.1);
   }
}
