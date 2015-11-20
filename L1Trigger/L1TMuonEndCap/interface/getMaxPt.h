// For DT PT Assignment. 
// getMaxPT() will solve for the maximum allowable PT as defined by the "corridors". 
// The corridor root files were generated with "runBasicCuts.C", which defines the 
// number of PT bins. The parameter NPTBins below MUST match NPTBins in runBasicCuts.C. 

#include <iomanip>
#include <vector>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <map>
#include <string>
#include <iomanip>

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "sstream"
#include "TNtuple.h"
#include "TGraphErrors.h"
#include "TH1F.h"
#include "TH1.h"
#include "TH2.h"
#include "TMatrixD.h"
#include "TGraph.h"
#include "TF1.h"
#include "TCanvas.h"
#include "sstream"
#include "TMath.h"
#include "TGraphAsymmErrors.h"
#include "TLegend.h"

using namespace std;


int NETABITS = 5;
int NETABINS = 1<<NETABITS;
int NPTBins = 200;

// The TGraphs which define the corridors for dPhi vs PT. The arrays are in the form dPhi_Cut[stationA-1][stationB-1].  
TGraph *gdPhi12_Cut__[32];
TGraph *gdPhi23_Cut__[32];
TGraph *gdPhi34_Cut__[32];
TGraph *gdEta12_Cut__[32];
TGraph *gdEta23_Cut__[32];
TGraph *gdEta34_Cut__[32];
TGraph *gCLCT1_Cut__[32];
TGraph *gCLCT2_Cut__[32];
TGraph *gCLCT3_Cut__[32];
TGraph *gCLCT4_Cut__[32];

int loaded__ = false;

// Histogram used to rebin PT. 
TH1F *hPT__;

// load() reads in the corridor TGraphs from the root files. 
void load(int perCUT=90)
{
  // "perCUT" is the fraction of events under the corridor curve.
  // For DTs, you currently have three options: perCUT = 85, perCUT = 90, or perCUT = 98. 
  // You'll need to use "runBasicCuts.C" to generate other tail fraction scenarios. 
  
 
  hPT__ = new TH1F("hPT__","",NPTBins,0,200); 
 
  TString name = "L1Trigger/L1TMuon/data/emtf_luts/dPhi_Cuts_";
  name += perCUT; name += ".root";

  TFile f(name);
  
   for (int j=0;j<NETABINS;j++)
      {
	TString name = "gCut_dPhi12_eta_";
	name += j;
	gdPhi12_Cut__[j] = (TGraph *)f.Get(name);
	cout << gdPhi12_Cut__[j]->GetName() << " loaded " << endl; 
      }
    for (int j=0;j<NETABINS;j++)
      {
	TString name = "gCut_dPhi23_eta_";
	name += j;
	gdPhi23_Cut__[j] = (TGraph *)f.Get(name);
	cout << gdPhi23_Cut__[j]->GetName() << " loaded " << endl; 
      }
    for (int j=0;j<NETABINS;j++)
      {
	TString name = "gCut_dPhi34_eta_";
	name+=j;
	gdPhi34_Cut__[j] = (TGraph *)f.Get(name);
	cout << gdPhi34_Cut__[j]->GetName() << " loaded " << endl; 
      }
  for (int j=0;j<NETABINS;j++)
      {
	TString name = "gCut_dEta12_eta_";
	name+=j;
	gdEta12_Cut__[j] = (TGraph *)f.Get(name);
	cout << gdEta12_Cut__[j]->GetName() << " loaded " << endl; 
      }
 for (int j=0;j<NETABINS;j++)
      {
	TString name = "gCut_dEta23_eta_";
	name+=j;
	gdEta23_Cut__[j] = (TGraph *)f.Get(name);
	cout << gdEta23_Cut__[j]->GetName() << " loaded " << endl; 
      }
 for (int j=0;j<NETABINS;j++)
      {
	TString name = "gCut_dEta34_eta_";
	name+=j;
	gdEta34_Cut__[j] = (TGraph *)f.Get(name);
	cout << gdEta34_Cut__[j]->GetName() << " loaded " << endl; 
      }
 for (int j=0;j<NETABINS;j++)
      {
	TString name = "gCut_CLCT1_eta_";
	name+=j;
	gCLCT1_Cut__[j] = (TGraph *)f.Get(name);
	cout << gCLCT1_Cut__[j]->GetName() << " loaded " << endl; 
      }
for (int j=0;j<NETABINS;j++)
      {
	TString name = "gCut_CLCT2_eta_";
	name+=j;
	gCLCT2_Cut__[j] = (TGraph *)f.Get(name);
	cout << gCLCT2_Cut__[j]->GetName() << " loaded " << endl; 
      }
for (int j=0;j<NETABINS;j++)
      {
	TString name = "gCut_CLCT3_eta_";
	name+=j;
	gCLCT3_Cut__[j] = (TGraph *)f.Get(name);
	cout << gCLCT3_Cut__[j]->GetName() << " loaded " << endl; 
      }
for (int j=0;j<NETABINS;j++)
      {
	TString name = "gCut_CLCT4_eta_";
	name+=j;
	gCLCT4_Cut__[j] = (TGraph *)f.Get(name);
	cout << gCLCT4_Cut__[j]->GetName() << " loaded " << endl; 
      }
  loaded__ = true;

  
}

// solve() takes the PT hypothesis (generally from BDT) and checks for the 
// maximum PT consistent with the input value (such as dPhiAB) and input corridor 
// given by the TGraph pointer. For the given input value, the maximum acceptable 
// PT is returned such that the corridor condition is satisfied.   
float solve(float ptHyp, float val, TGraph *g)
{
  Int_t theBIN = hPT__->FindBin( ptHyp); // for the pt hypothesis get the bin
  Float_t thePT = theBIN;
  
  float maxPt = 1e6;		  
  float cut = g->Eval( theBIN ) ; 

  //  cout << "thePT = " << thePT << ", cut = " << cut << endl;

  if (cut>-1) // is a cut defined at this PT? Sometimes the training sample has too few events for a given PT bin. 
    if (fabs(val) > cut ) // check if the value is greater than the corridor cut
      {
	maxPt  = 1; 
	for (float p=thePT-1; p>0; p=p-1) // The corridor TGraphs count PT bins by "0"...sorry. 
	  {
	    float eval = g->Eval((float)p); // Get the maximum acceptable input value at this test-PT 
	    // cout << eval << " " << val << endl;
	    if (eval>=0) // make sure there is corridor value defined here, or else we ignore this test-PT and move on
	      {
		if (fabs(val) < eval) // Finally, is the input value below the corridor at the test-PT? If so, we can stop. 
		  {
		    maxPt = p;
		    break;
		  }
	      }
	  }
      }

  float output = ptHyp;
  if (maxPt <  output) // If the final test-PT is lower than the original hypothesis, we need to use the final test-PT. 
    output = maxPt;

  return output;
}

// getMaxPT() solves for the maximum acceptable PT value given an ensemble of corridors defined for 
// the dPhi and PhiBend input values. For a given station pair, this will call the solve function
// above for dPhiAB and PhiBendA, and will choose the minimum of the best PT solutions. The minimum
// of the PTs will satisfy all corridors simultaniously, I promise. 
float getMaxPT(float ptHyp, Int_t eta, float dPhi12, float dPhi23, float dPhi34, int perCUT=100)
{
  //cout << dPhi12 << " " << dPhi23 << " " << dPhi34 << endl;
  if (!loaded__) // make sure the corridor TGraphs are loaded from the root files! 
    load(perCUT);

  if (eta>NETABINS) eta = NETABINS-1;

  float maxPt_dPhi12 = solve(ptHyp, dPhi12, gdPhi12_Cut__[eta]); // Get the best PT using dPhiAB corridors. 
  float maxPt_dPhi23 = solve(ptHyp, dPhi23, gdPhi23_Cut__[eta]); // Get the best PT using dPhiAB corridors. 
  float maxPt_dPhi34 = solve(ptHyp, dPhi34, gdPhi34_Cut__[eta]); // Get the best PT using dPhiAB corridors. 
      
  // Now take the minimum of the four possible PT values. 
  float ptOut = ptHyp;

  if (maxPt_dPhi12 < ptOut)
    ptOut = maxPt_dPhi12;
  if (maxPt_dPhi23 < ptOut)
    ptOut = maxPt_dPhi23;
  if (maxPt_dPhi34 < ptOut)
    ptOut = maxPt_dPhi34;

  //ptOut = (maxPt_dPhi12 + maxPt_dPhi23 + maxPt_dPhi34)/3;
  
  return ptOut;
}
float getMaxPT_dEta(float ptHyp, Int_t eta, float dEta12, float dEta23, float dEta34, int perCUT=100)
{
  //cout << dPhi12 << " " << dPhi23 << " " << dPhi34 << endl;
  if (!loaded__) // make sure the corridor TGraphs are loaded from the root files! 
    load();

  if (eta>NETABINS) eta = NETABINS-1;

  float maxPt_dEta12 = solve(ptHyp, dEta12, gdEta12_Cut__[eta]); // Get the best PT using dPhiAB corridors. 
  float maxPt_dEta23 = solve(ptHyp, dEta23, gdEta23_Cut__[eta]); // Get the best PT using dPhiAB corridors. 
  float maxPt_dEta34 = solve(ptHyp, dEta34, gdEta34_Cut__[eta]); // Get the best PT using dPhiAB corridors. 
      
  // Now take the minimum of the four possible PT values. 
  float ptOut = ptHyp;

  if (maxPt_dEta12 < ptOut)
    ptOut = maxPt_dEta12;
  if (maxPt_dEta23 < ptOut)
    ptOut = maxPt_dEta23;
  if (maxPt_dEta34 < ptOut)
    ptOut = maxPt_dEta34;

  
  
  return ptOut;
}
float getMaxPT_CLCT(float ptHyp, Int_t eta, float CLCT1, float CLCT2, float CLCT3, float CLCT4, int perCUT=100)
{
  //cout << dPhi12 << " " << dPhi23 << " " << dPhi34 << endl;
  if (!loaded__) // make sure the corridor TGraphs are loaded from the root files! 
    load();

  if (eta>NETABINS) eta = NETABINS-1;

  float maxPt_CLCT1 = solve(ptHyp, CLCT1, gCLCT1_Cut__[eta]); // Get the best PT using dPhiAB corridors. 
  float maxPt_CLCT2 = solve(ptHyp, CLCT2, gCLCT2_Cut__[eta]); // Get the best PT using dPhiAB corridors. 
  float maxPt_CLCT3 = solve(ptHyp, CLCT3, gCLCT3_Cut__[eta]); // Get the best PT using dPhiAB corridors. 
  float maxPt_CLCT4 = solve(ptHyp, CLCT4, gCLCT4_Cut__[eta]); // Get the best PT using dPhiAB corridors. 
      
  // Now take the minimum of the four possible PT values. 
  float ptOut = ptHyp;

  if (maxPt_CLCT1 < ptOut)
    ptOut = maxPt_CLCT1;
 if (maxPt_CLCT2 < ptOut)
    ptOut = maxPt_CLCT2;
 if (maxPt_CLCT3 < ptOut)
    ptOut = maxPt_CLCT3;
 if (maxPt_CLCT4 < ptOut)
    ptOut = maxPt_CLCT4;
 
  return ptOut;
}


// dumpTables just makes a text dump of the corridors as a function of PT. This may be useful
// when ported to CMSSW. 
/*
float dumpTables()
{
  int perDiff = 0;
  for (int itr=0; itr<3; itr++)
    {
      if (itr==0) perDiff = 85;
      if (itr==1) perDiff = 90;
      if (itr==2) perDiff = 98;

      TString pd; pd+=perDiff;

      load(perDiff);
      std::cout << std::endl;
      std::cout << "Writing delta-Phi(MB1-MB2) tables for " << (100-perDiff) << " % tail clip" << std::endl;
      ofstream file1("dPhi_Cut_12_" + pd + "percent.dat");
      for (float pt_=0;pt_<=140; pt_+=0.5)
	{
	  Int_t theBIN = hPT__->FindBin(pt_);
	  float cut = gdPhi_Cut__[0][1]->Eval(theBIN);
	  std::cout << std::setw(10) << pt_ << std::setw(10) << cut << std::endl;
	  file1 << std::setw(10) << pt_ << std::setw(10) << cut << std::endl;
	}
      std::cout << std::endl;
      std::cout << "Writing delta-Phi(MB1-MB3) tables for " << (100-perDiff) << " % tail clip" << std::endl;
      ofstream file2("dPhi_Cut_13_" + pd + "percent.dat");
      for (float pt_=0;pt_<=140; pt_+=0.5)
	{
	  Int_t theBIN = hPT__->FindBin(pt_);
	  float cut = gdPhi_Cut__[0][2]->Eval(theBIN);
	  std::cout << std::setw(10) << pt_ << std::setw(10) << cut << std::endl;
	  file2 << std::setw(10) << pt_ << std::setw(10) << cut << std::endl;
	}
      std::cout << std::endl;
      std::cout << "Writing delta-Phi(MB1-MB4) tables for " << (100-perDiff) << " % tail clip" << std::endl;
      ofstream file3("dPhi_Cut_14_" + pd + "percent.dat");
      for (float pt_=0;pt_<=140; pt_+=0.5)
	{
	  Int_t theBIN = hPT__->FindBin(pt_);
	  float cut = gdPhi_Cut__[0][3]->Eval(theBIN);
	  std::cout << std::setw(10) << pt_ << std::setw(10) << cut << std::endl;
	  file3 << std::setw(10) << pt_ << std::setw(10) << cut << std::endl;
	}
      std::cout << std::endl;
      std::cout << "Writing delta-Phi(MB2-MB3) tables for " << (100-perDiff) << " % tail clip" << std::endl;
      ofstream file4("dPhi_Cut_23_" + pd + "percent.dat");
      for (float pt_=0;pt_<=140; pt_+=0.5)
	{
	  Int_t theBIN = hPT__->FindBin(pt_);
	  float cut = gdPhi_Cut__[1][2]->Eval(theBIN);
	  std::cout << std::setw(10) << pt_ << std::setw(10) << cut << std::endl;
	  file4 << std::setw(10) << pt_ << std::setw(10) << cut << std::endl;
	}
      std::cout << std::endl;
      std::cout << "Writing delta-Phi(MB2-MB4) tables for " << (100-perDiff) << " % tail clip" << std::endl;
      ofstream file5("dPhi_Cut_24_" + pd + "percent.dat");
      for (float pt_=0;pt_<=140; pt_+=0.5)
	{
	  Int_t theBIN = hPT__->FindBin(pt_);
	  float cut = gdPhi_Cut__[1][3]->Eval(theBIN);
	  std::cout << std::setw(10) << pt_ << std::setw(10) << cut << std::endl;
	  file5 << std::setw(10) << pt_ << std::setw(10) << cut << std::endl;
	}
      std::cout << std::endl;
      std::cout << "Writing delta-Phi(MB3-MB4) tables for " << (100-perDiff) << " % tail clip" << std::endl;
      ofstream file6("dPhi_Cut_34_" + pd + "percent.dat");
      for (float pt_=0;pt_<=140; pt_+=0.5)
	{
	  Int_t theBIN = hPT__->FindBin(pt_);
	  float cut = gdPhi_Cut__[2][3]->Eval(theBIN);
	  std::cout << std::setw(10) << pt_ << std::setw(10) << cut << std::endl;
	  file6 << std::setw(10) << pt_ << std::setw(10) << cut << std::endl;
	}

      std::cout << std::endl;
      std::cout << "Writing Phib(MB1) tables for " << (100-perDiff) << " % tail clip" << std::endl;
      ofstream file7("dPhib_Cut_1_" + pd + "percent.dat");
      for (float pt_=0;pt_<=140; pt_+=0.5)
	{
	  Int_t theBIN = hPT__->FindBin(pt_);
	  float cut = gPhib_Cut__[0]->Eval(theBIN);
	  std::cout << std::setw(10) << pt_ << std::setw(10) << cut << std::endl;
	  file7 << std::setw(10) << pt_ << std::setw(10) << cut << std::endl;
	}
      std::cout << std::endl;
      std::cout << "Writing Phib(MB2) tables for " << (100-perDiff) << " % tail clip" << std::endl;
      ofstream file8("dPhib_Cut_2_" + pd + "percent.dat");
      for (float pt_=0;pt_<=140; pt_+=0.5)
	{
	  Int_t theBIN = hPT__->FindBin(pt_);
	  float cut = gPhib_Cut__[1]->Eval(theBIN);
	  std::cout << std::setw(10) << pt_ << std::setw(10) << cut << std::endl;
	  file8 << std::setw(10) << pt_ << std::setw(10) << cut << std::endl;
	}
      std::cout << std::endl;
      std::cout << "Writing Phib(MB4) tables for " << (100-perDiff) << " % tail clip" << std::endl;
      ofstream file9("dPhib_Cut_4_" + pd + "percent.dat");
      for (float pt_=0;pt_<=140; pt_+=0.5)
	{
	  Int_t theBIN = hPT__->FindBin(pt_);
	  float cut = gPhib_Cut__[3]->Eval(theBIN);
	  std::cout << std::setw(10) << pt_ << std::setw(10) << cut << std::endl;
	  file9 << std::setw(10) << pt_ << std::setw(10) << cut << std::endl;
	}


    }

}
*/
