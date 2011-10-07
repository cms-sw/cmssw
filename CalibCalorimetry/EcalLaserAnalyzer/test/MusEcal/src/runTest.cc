#include <cassert>
#include <iostream>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <vector>
#include <map>
using namespace std;

#include "TMath.h"
#include "TRint.h"
#include "TGClient.h"
#include "TGraphErrors.h"
#include "TGraph.h"
#include "TStyle.h"

#include <TGWindow.h>
#include <TGDockableFrame.h>

#include "MusEcal.hh"
#include "MusEcalGUI.hh"
#include "../../interface/MEGeom.h"
#include "MEVarVector.hh"
#include "MEEBDisplay.hh"
#include "MEPlotWindow.hh"
#include "../../interface/MEChannel.h" 
#include "MERunManager.hh"
#include "MENormManager.hh"


void setTDRStyle(int isIt1D) {
  TStyle *tdrStyle = new TStyle("tdrStyle","Style for P-TDR");

// For the canvas:
  tdrStyle->SetCanvasBorderMode(0);
  tdrStyle->SetCanvasColor(kWhite);
  tdrStyle->SetCanvasDefH(600); //Height of canvas
  tdrStyle->SetCanvasDefW(600); //Width of canvas
  tdrStyle->SetCanvasDefX(0);   //POsition on screen
  tdrStyle->SetCanvasDefY(0);

// For the Pad:
  tdrStyle->SetPadBorderMode(0);
  tdrStyle->SetPadColor(kWhite);
  tdrStyle->SetPadGridX(false);
  tdrStyle->SetPadGridY(false);
  tdrStyle->SetGridColor(0);
  tdrStyle->SetGridStyle(3);
  tdrStyle->SetGridWidth(1);

// For the frame:
  tdrStyle->SetFrameBorderMode(0);
  tdrStyle->SetFrameBorderSize(1);
  tdrStyle->SetFrameFillColor(0);
  tdrStyle->SetFrameFillStyle(0);
  tdrStyle->SetFrameLineColor(1);
  tdrStyle->SetFrameLineStyle(1);
  tdrStyle->SetFrameLineWidth(1);

// For the histo:
  tdrStyle->SetHistLineColor(1);
  tdrStyle->SetHistLineStyle(0);
  tdrStyle->SetHistLineWidth(1);

  tdrStyle->SetEndErrorSize(2);
  tdrStyle->SetErrorX(0.);

  tdrStyle->SetMarkerStyle(20);

//For the fit/function:
  tdrStyle->SetOptFit(1);
  tdrStyle->SetFitFormat("5.4g");
  tdrStyle->SetFuncColor(2);
  tdrStyle->SetFuncStyle(1);
  tdrStyle->SetFuncWidth(1);

//For the date:
  tdrStyle->SetOptDate(0);

// For the statistics box:
  tdrStyle->SetOptFile(0);
  tdrStyle->SetOptStat(0); // To display the mean and RMS:   SetOptStat("mr");
  tdrStyle->SetStatColor(kWhite);
  tdrStyle->SetStatFont(42);
  tdrStyle->SetStatFontSize(0.025);
  tdrStyle->SetStatTextColor(1);
  tdrStyle->SetStatFormat("6.4g");
  tdrStyle->SetStatBorderSize(1);
  tdrStyle->SetStatH(0.1);
  tdrStyle->SetStatW(0.15);

// Margins:
  tdrStyle->SetPadTopMargin(0.05);
  tdrStyle->SetPadBottomMargin(0.13);
  tdrStyle->SetPadLeftMargin(0.13);
  tdrStyle->SetPadRightMargin(0.05);

// For the Global title:
  tdrStyle->SetOptTitle(0);
  tdrStyle->SetTitleFont(42);
  tdrStyle->SetTitleColor(1);
  tdrStyle->SetTitleTextColor(1);
  tdrStyle->SetTitleFillColor(10);
  tdrStyle->SetTitleFontSize(0.05);

// For the axis titles:
  tdrStyle->SetTitleColor(1, "XYZ");
  tdrStyle->SetTitleFont(42, "XYZ");
  tdrStyle->SetTitleSize(0.06, "XYZ");
  tdrStyle->SetTitleXOffset(0.9);
  tdrStyle->SetTitleYOffset(1.05);

// For the axis labels:
  tdrStyle->SetLabelColor(1, "XYZ");
  tdrStyle->SetLabelFont(42, "XYZ");
  tdrStyle->SetLabelOffset(0.007, "XYZ");
  tdrStyle->SetLabelSize(0.05, "XYZ");

// For the axis:
  tdrStyle->SetAxisColor(1, "XYZ");
  tdrStyle->SetStripDecimals(kTRUE);
  tdrStyle->SetTickLength(0.03, "XYZ");
  tdrStyle->SetNdivisions(510, "XYZ");
  tdrStyle->SetPadTickX(1);  // To get tick marks on the opposite side of the frame
  tdrStyle->SetPadTickY(1);

// Change for log plots:
  tdrStyle->SetOptLogx(0);
  tdrStyle->SetOptLogy(0);
  tdrStyle->SetOptLogz(0);

// Postscript options:
  tdrStyle->SetPaperSize(20.,20.);

  // set the paper & margin sizes
 if( isIt1D==0 ){
   tdrStyle->SetPaperSize(20,26);
   tdrStyle->SetPadTopMargin(0.05);
   tdrStyle->SetPadTopMargin(0.10);
   tdrStyle->SetPadRightMargin(0.16);
   tdrStyle->SetPadBottomMargin(0.16);
   tdrStyle->SetPadLeftMargin(0.16);
 }

 tdrStyle->SetMarkerStyle(2);
 tdrStyle->SetHistLineWidth(2);
 tdrStyle->SetLineStyleString(2,"[12 12]"); // postscript dashes

 tdrStyle->SetOptStat(0);
 tdrStyle->SetOptFit(0);

 tdrStyle->SetPadTickX(1);
 tdrStyle->SetPadTickY(1);
 tdrStyle->SetPalette(1);

  tdrStyle->cd();
}
int main(int argc, char **argv)
{
  
  int c;
  int first=0;
  int last=999999;

  while ( (c = getopt( argc, argv, "f:l:m:d:" ) ) != EOF ) 
    {
      switch (c) 
	{
	case 'f': 
	  first       =  atoi ( optarg ) ;    
	  cout <<" first run "<<first << endl;
	  break;
	case 'l': 
	  last       =  atoi ( optarg ) ;    
	  cout <<" last run "<<last << endl;
	  break;
	}
    }
 

  for (int ilmr=1;ilmr<93;ilmr++){
  
    MENormManager test(ilmr);
    
  }


  //  cout<< "nChannels blue: "<<nChanBlue<<" red:"<<nChanRed<< endl;
  
  // Loop on channels inside LMR:
  //=============================
  
//   for( int ii=0; ii<nChanBlue; ii++ ) 
//     {
      
//       cout<<"Looking at channel :"<<ii<< endl;

//       vector< ME::Time > timeGlobalBlue;
//       vector< ME::Time > timeGlobalRed;
      
//       vector< float > MIDBlue;
//       vector< bool  > flagMIDBlue;

//       vector< float > MIDRed;
//       vector< bool  > flagMIDRed;      
      
//       MEChannel* _leafBlue = vecBlue[ii];
//       MEChannel* leafBlue_ = _leafBlue;
//       cout<<"Got Blue Leafs"<< endl;

//       int iX=leafBlue_->ix();
//       int iY=leafBlue_->iy();
//       cout<<"Channel coordinates: "<<iX<<" "<<iY<< endl;

//       MEChannel* _leafRed = vecRed[ii];
//       MEChannel* leafRed_ = _leafRed;
//       cout<<"Got Red Leafs"<< endl;
      
      
      
//       // Get back variables:
//       //====================
      

//       MEVarVector* apdVectorBlue_ =  MusEcalBlue->curMgr()->apdVector(leafBlue_);
//       apdVectorBlue_->getTime( timeGlobalBlue );
//       MEVarVector* midVectorBlue_=MusEcalBlue->midVector(leafBlue_);
//       midVectorBlue_->getValAndFlag(ME::iMID_MEAN, timeGlobalBlue,MIDBlue,flagMIDBlue);
//       cout<<"blue time size:"<<timeGlobalBlue.size()<< endl;

//       MEVarVector* apdVectorRed_ ;
//       apdVectorRed_ =  MusEcalRed->curMgr()->apdVector(leafRed_);
//       apdVectorRed_->getTime( timeGlobalRed );
      
//       MEVarVector* midVectorRed_  =  MusEcalRed->midVector(leafRed_);
//       midVectorRed_->getValAndFlag(ME::iMID_MEAN, timeGlobalRed,MIDRed,flagMIDRed);
//       cout<<"red time size:"<<timeGlobalRed.size()<< endl;
      
//       for (int it=0;it<timeGlobalBlue.size();it++){
	
// 	cout<< " it="<<it<<" timeBlue="<<timeGlobalBlue[it]<<" closestRed="<<
// 	  MusEcalRed->curMgr()->closestKey(timeGlobalBlue[it])<<" closestRedInFuture="<<
// 	  MusEcalRed->curMgr()->closestKeyInFuture(timeGlobalBlue[it])<<
// 	  " blueMid="<<MIDBlue[it]<<" blueFlag="<<flagMIDBlue[it]<<endl;
	
	
//       }
      
//     }
}
