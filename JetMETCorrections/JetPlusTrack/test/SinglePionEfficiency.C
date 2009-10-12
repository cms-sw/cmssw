#define SinglePionEfficiency_cxx
#include "SinglePionEfficiency.h"
#include <fstream>
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void SinglePionEfficiency::setTDRStyle(Int_t xlog, Int_t ylog) {

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
  // tdrStyle->SetPadBorderSize(Width_t size = 1);
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
  // tdrStyle->SetHistFillColor(1);
  // tdrStyle->SetHistFillStyle(0);
  tdrStyle->SetHistLineColor(1);
  tdrStyle->SetHistLineStyle(0);
  tdrStyle->SetHistLineWidth(2);
  // tdrStyle->SetLegoInnerR(Float_t rad = 0.5);
  // tdrStyle->SetNumberContours(Int_t number = 20);

  tdrStyle->SetEndErrorSize(4);
  //  tdrStyle->SetErrorMarker(20);
  //  tdrStyle->SetErrorX(0.);
  
  tdrStyle->SetMarkerStyle(20);

//For the fit/function:
  tdrStyle->SetOptFit(1);
  tdrStyle->SetFitFormat("5.4g");
  tdrStyle->SetFuncColor(1);
  tdrStyle->SetFuncStyle(1);
  tdrStyle->SetFuncWidth(1);

//For the date:
  tdrStyle->SetOptDate(0);
  // tdrStyle->SetDateX(Float_t x = 0.01);
  // tdrStyle->SetDateY(Float_t y = 0.01);

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
  // tdrStyle->SetStatStyle(Style_t style = 1001);
  // tdrStyle->SetStatX(Float_t x = 0);
  // tdrStyle->SetStatY(Float_t y = 0);

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
  // tdrStyle->SetTitleH(0); // Set the height of the title box
  // tdrStyle->SetTitleW(0); // Set the width of the title box
  // tdrStyle->SetTitleX(0); // Set the position of the title box
  // tdrStyle->SetTitleY(0.985); // Set the position of the title box
  // tdrStyle->SetTitleStyle(Style_t style = 1001);
  // tdrStyle->SetTitleBorderSize(2);

// For the axis titles:

  tdrStyle->SetTitleColor(1, "XYZ");
  tdrStyle->SetTitleFont(42, "XYZ");
  tdrStyle->SetTitleSize(0.06, "XYZ");
  // tdrStyle->SetTitleXSize(Float_t size = 0.02); // Another way to set the size?
  // tdrStyle->SetTitleYSize(Float_t size = 0.02);
  tdrStyle->SetTitleXOffset(0.9);
  tdrStyle->SetTitleYOffset(1.05);
  // tdrStyle->SetTitleOffset(1.1, "Y"); // Another way to set the Offset

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
  tdrStyle->SetOptLogx(xlog);
  tdrStyle->SetOptLogy(ylog);
  tdrStyle->SetOptLogz(0);

// Postscript options:

//  tdrStyle->SetPaperSize(7.5,7.5);

  tdrStyle->SetPaperSize(15.,15.);

//  tdrStyle->SetPaperSize(20.,20.);

  // tdrStyle->SetLineScalePS(Float_t scale = 3);
  // tdrStyle->SetLineStyleString(Int_t i, const char* text);
  // tdrStyle->SetHeaderPS(const char* header);
  // tdrStyle->SetTitlePS(const char* pstitle);

  // tdrStyle->SetBarOffset(Float_t baroff = 0.5);
  // tdrStyle->SetBarWidth(Float_t barwidth = 0.5);
  // tdrStyle->SetPaintTextFormat(const char* format = "g");
  // tdrStyle->SetPalette(Int_t ncolors = 0, Int_t* colors = 0);
  // tdrStyle->SetTimeOffset(Double_t toffset);
  // tdrStyle->SetHistMinimumZero(kTRUE);

  tdrStyle->cd();
}


void SinglePionEfficiency::Loop()
{
//   In a ROOT session, you can do:
//      Root > .L SinglePionEfficiency.C
//      Root > SinglePionEfficiency t
//      Root > t.GetEntry(12); // Fill t data members with entry number 12
//      Root > t.Show();       // Show values of entry 12
//      Root > t.Show(16);     // Read and show values of entry 16
//      Root > t.Loop();       // Loop on all entries
//

//     This is the loop skeleton where:
//    jentry is the global entry number in the chain
//    ientry is the entry number in the current Tree
//  Note that the argument to GetEntry must be:
//    jentry for TChain::GetEntry
//    ientry for TTree::GetEntry and TBranch::GetEntry
//
//       To read only selected branches, Insert statements like:
// METHOD1:
//    fChain->SetBranchStatus("*",0);  // disable all branches
//    fChain->SetBranchStatus("branchname",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch
   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();
 
   Long64_t nbytes = 0, nb = 0;

   // number of pt bins and intervals
   const Int_t nptbins = 12;
   const Int_t nptcuts = nptbins+1;
   //                       0    1   2    3    4     5     6     7     8     9    10   11    
   Double_t pt[nptcuts]={0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40., 50.0};
   // number of eta bins and intervals
   const Int_t netabins = 12;
   const Int_t netacuts = netabins+1;
   //                         0    1    2    3    4    5    6    7    8    9    10   11
   Double_t eta[netacuts]={0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4};
   cout <<"  Eta bins " << endl;

   // prepare eta points for graph
   Float_t etagr[netabins], eetagr[netabins];
   for(Int_t i = 0; i < netabins; i++) {
     etagr[i]  = 0.5*(eta[i]+eta[i+1]);
     eetagr[i] = 0.5*(eta[i+1]-eta[i]);
     cout <<" i = " << i <<" Eta = " << etagr[i] <<" err Eta = " << eetagr[i] << endl;
   } 
   // prepare for graph
   Float_t ptgr[nptbins], eptgr[nptbins];
   for(Int_t i = 0; i < nptbins; i++) {
     ptgr[i]  = 0.5*(pt[i]+pt[i+1]);
     eptgr[i] = 0.5*(pt[i+1]-pt[i]);
     cout <<" i = " << i <<" pT = " << ptgr[i] <<" err pT = " << eptgr[i] << endl;
   } 


   TH2D* hResp = new TH2D("hResp","Resp",nptbins, pt, netabins, eta);
   // histos for energy losses
   TH1F* hEnTrkNotInter[netabins];
   const char* namesEnTrkNotInter[netabins] = {"hEnTrkNotInter1",
					       "hEnTrkNotInter2",
					       "hEnTrkNotInter3",
					       "hEnTrkNotInter4",
					       "hEnTrkNotInter5",
					       "hEnTrkNotInter6",
					       "hEnTrkNotInter7",
					       "hEnTrkNotInter8",
					       "hEnTrkNotInter9",
					       "hEnTrkNotInter10",
					       "hEnTrkNotInter11",
					       "hEnTrkNotInter12"};

   const char* titleEnTrkNotInter[netabins] = {"EnTrkNotInter1",
					       "EnTrkNotInter2",
					       "EnTrkNotInter3",
					       "EnTrkNotInter4",
					       "EnTrkNotInter5",
					       "EnTrkNotInter6",
					       "EnTrkNotInter7",
					       "EnTrkNotInter8",
					       "EnTrkNotInter9",
					       "EnTrkNotInter10",
					       "EnTrkNotInter11",
					       "EnTrkNotInter12"};

   TH1F* hEnTrkInter[netabins];
   const char* namesEnTrkInter[netabins] = {"hEnTrkInter1",
					    "hEnTrkInter2",
					    "hEnTrkInter3",
					    "hEnTrkInter4",
					    "hEnTrkInter5",
					    "hEnTrkInter6",
					    "hEnTrkInter7",
					    "hEnTrkInter8",
					    "hEnTrkInter9",
					    "hEnTrkInter10",
					    "hEnTrkInter11",
					    "hEnTrkInter12"};

   const char* titleEnTrkInter[netabins] = {"EnTrkInter1",
					    "EnTrkInter2",
					    "EnTrkInter3",
					    "EnTrkInter4",
					    "EnTrkInter5",
					    "EnTrkInter6",
					    "EnTrkInter7",
					    "EnTrkInter8",
					    "EnTrkInter9",
					    "EnTrkInter10",
					    "EnTrkInter11",
					    "EnTrkInter12"};

   for(Int_t ih=0; ih < netabins; ih++) { 
     hEnTrkNotInter[ih] = new TH1F(namesEnTrkNotInter[ih], titleEnTrkNotInter[ih], 40, -1., 3.);
     hEnTrkInter[ih] = new TH1F(namesEnTrkInter[ih], titleEnTrkInter[ih], 40, -1., 3.);
   }
   //
 
   // ===> for pi- and pi+
   // N total and N reco tracks
   // find how to write output in file
   //~/scratch0/CMSSW_TEST/cmssw134gammajet/src/JetMETCorrections/GammaJet/src/GammaJetAnalysis.cc

   // total number of analized MC tracks
   Int_t ntrk[netabins][nptbins] =   {  
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
   };
   // number of found tracks
   Int_t ntrkreco[netabins][nptbins] = {
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
   };
   // number of lost tracks
   Int_t ntrklost[netabins][nptbins] = {
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
   };
   // number of tracks with impact point on calo
   Int_t ntrkrecor[netabins][nptbins] = {
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
   };

   // trk efficiency and error
   Float_t trkeff[netabins][nptbins], etrkeff[netabins][nptbins];
   // response in 11x11 crystals + 5x5 HCAL around IP in ECAL
   Double_t response[netabins][nptbins] = {
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
   };

   // response for found tracks in cone 0.5 around MC track direction at vertex
   Double_t responseFoundTrk[netabins][nptbins] = {
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
   };

   // response for lost tracks in cone 0.5 around MC track direction at vertex
   Double_t responseLostTrk[netabins][nptbins] = {
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
     {0,0,0,0,0,0,0,0,0,0,0,0},
   };

   // average response in 11x11 crystals + 5x5 HCAL around IP in ECAL
   Double_t responseF[netabins][nptbins];
   // average response for found tracks in cone 0.5 around MC track direction at vertex
   Double_t responseFoundTrkF[netabins][nptbins];
   // average response for lost tracks in cone 0.5 around MC track direction at vertex
   Double_t responseLostTrkF[netabins][nptbins];
   // ratio of responses of lost and found tracks
   Double_t leak[netabins][nptbins];

   // 
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;
      // counting events for efficiency calculation
      for(Int_t ieta = 0; ieta < netabins; ieta++) {
	for(Int_t ipt = 0; ipt < nptbins; ipt++) {
	  // ==> pi+
	  if(fabs(etaSim1) >= eta[ieta] && fabs(etaSim1) < eta[ieta+1]) {
	    // checking leakage
	    if(ptSim1 > 10. && ptSim1 < 10000.)
	      {
		if(drTrk1 < 0.01 && purityTrk1 == 1) 
		  {
		    hEnTrkNotInter[ieta]->Fill(etCalo1/ptSim1);
		  } else {
		    hEnTrkInter[ieta]->Fill(etCalo1/ptSim1);
		  }
	      }
	    // end of checking leakage
	    if(ptSim1 >= pt[ipt] && ptSim1 < pt[ipt+1]) {
	      ntrk[ieta][ipt] = ntrk[ieta][ipt]+1;
	      // number of reco tracks
	      if(drTrk1 < 0.01 && purityTrk1 == 1) {
		ntrkreco[ieta][ipt] = ntrkreco[ieta][ipt]+1;
		// response for found tracks
		responseFoundTrk[ieta][ipt] = responseFoundTrk[ieta][ipt] + etCalo1/ptSim1;  
		//
		Double_t theta = 2.*atan(exp(-etaSim1));
		Double_t eSim1 = ptSim1/sin(theta);
		if(e1ECAL11x11 > -1000. && e1HCAL5x5 > -1000. && fabs(etaSim1) < 1000.) { 
		  ntrkrecor[ieta][ipt] = ntrkrecor[ieta][ipt]+1;
		  Double_t e_ecal11 = e1ECAL11x11/eSim1;
		  Double_t e_hcal5  = e1HCAL5x5/eSim1;
		  Double_t e_ecalhcal11x5 = e_ecal11 + e_hcal5;
		  response[ieta][ipt] = response[ieta][ipt] + e_ecalhcal11x5;
		}
	      } else {
		// response for lost tracks
		ntrklost[ieta][ipt] = ntrklost[ieta][ipt]+1;
		responseLostTrk[ieta][ipt] = responseLostTrk[ieta][ipt] + etCalo1/ptSim1;  
		//
	      }
	    }
	  }
	  // ==> pi-
	  if(fabs(etaSim2) >= eta[ieta] && fabs(etaSim2) < eta[ieta+1]) {
	    // checking leakage
	    if(ptSim2 > 10. && ptSim2 < 1000.)
	      {
		if(drTrk2 < 0.01 && purityTrk2 == 1) 
		  {
		    hEnTrkNotInter[ieta]->Fill(etCalo2/ptSim2);
		  } else {
		    hEnTrkInter[ieta]->Fill(etCalo2/ptSim2);
		  }
	      }
	    // end of checking leakage
	    if(ptSim2 >= pt[ipt] && ptSim2 < pt[ipt+1]) {
	      ntrk[ieta][ipt] = ntrk[ieta][ipt]+1;
	      // number of reco tracks
	      if(drTrk2 < 0.01 && purityTrk2 == 1) {
		ntrkreco[ieta][ipt] = ntrkreco[ieta][ipt]+1;
		// response for found tracks
		responseFoundTrk[ieta][ipt] = responseFoundTrk[ieta][ipt] + etCalo2/ptSim2;  
		//
		Double_t theta = 2.*atan(exp(-etaSim2));
		Double_t eSim2 = ptSim2/sin(theta);
		if(e2ECAL11x11 > -1000. && e2HCAL5x5 > -1000. && fabs(etaSim2) < 1000.) { 
		  ntrkrecor[ieta][ipt] = ntrkrecor[ieta][ipt]+1;
		  Double_t e_ecal11 = e2ECAL11x11/eSim2;
		  Double_t e_hcal5  = e2HCAL5x5/eSim2;
		  Double_t e_ecalhcal11x5 = e_ecal11 + e_hcal5;
		  response[ieta][ipt] = response[ieta][ipt] + e_ecalhcal11x5;
		}
	      } else {
		// response for lost tracks
		ntrklost[ieta][ipt] = ntrklost[ieta][ipt]+1;
		responseLostTrk[ieta][ipt] = responseLostTrk[ieta][ipt] + etCalo2/ptSim2;  
		//
	      }
	    }
	  }
	}
      }
   }

   // calculate efficiencfy, full graph and write output stream
   ofstream myoutput_eff("CMSSW_167_TrackNonEff.txt");
   ofstream myoutput_eff1("CMSSW_167_TrackNonEff_one.txt");
   ofstream myoutput_leak("CMSSW_167_TrackLeakage.txt");
   ofstream myoutput_leak1("CMSSW_167_TrackLeakage_one.txt");
   ofstream myoutput_resp("CMSSW_167_response.txt");

   // calculate map of leackage
   for(Int_t ieta = 0; ieta < netabins; ieta++) {
     for(Int_t ipt = 0; ipt < nptbins; ipt++) {
       
	 // found tracks
       if(ntrkreco[ieta][ipt] != 0) {
	 
	 responseFoundTrkF[ieta][ipt] = responseFoundTrk[ieta][ipt]/ntrkreco[ieta][ipt]; 
	 
       } else {
	 
	 responseFoundTrkF[ieta][ipt] = responseFoundTrkF[ieta][ipt-1];
       }
       // lost tracks
       if(ntrklost[ieta][ipt] != 0) {
	 
	 responseLostTrkF[ieta][ipt] = responseLostTrk[ieta][ipt]/ntrklost[ieta][ipt]; 
	 
       } else {
	 
	 responseLostTrkF[ieta][ipt] = responseLostTrkF[ieta][ipt-1];

       }
     }
   } 

   for(Int_t ieta = 0; ieta <= netabins; ieta++) {
     for(Int_t ipt = 0; ipt <= nptbins; ipt++) {

       if(ieta < netabins && ipt < nptbins) {

	 leak[ieta][ipt] = responseLostTrkF[ieta][ipt]/responseFoundTrkF[ieta][ipt];
	 
	 cout <<" ieta = " << ieta
	      <<" eta = " << eta[ieta]
	      <<" ipt = " << ipt
	      <<" pt = " << pt[ipt]
	      <<" ntrkreco = " << ntrkreco[ieta][ipt] 
	      <<" ntrklost = " << ntrklost[ieta][ipt] 
	      <<" responseFoundTrk = " << responseFoundTrkF[ieta][ipt] 
	      <<" responseLostTrk = " << responseLostTrkF[ieta][ipt]
	      <<" leak = " << leak[ieta][ipt] <<  endl;

	 myoutput_leak << ieta << " " 
		       << ipt  << " " 
		       << eta[ieta] << " " 
		       << pt[ipt] << " "
		       << leak[ieta][ipt] << endl;
	 myoutput_leak1 << ieta << " " 
			<< ipt  << " " 
			<< eta[ieta] << " " 
			<< pt[ipt] << " "
			<< " 1. " << endl;
       }
       if(ipt == nptbins && ieta < netabins) {
	 
	 myoutput_leak << ieta << " " 
		       << ipt  << " " 
		       << eta[ieta] << " " 
		       << pt[ipt] << " "
		       << leak[ieta][ipt-1] << endl;
	 myoutput_leak1 << ieta << " " 
			<< ipt  << " " 
			<< eta[ieta] << " " 
			<< pt[ipt] << " "
			<< " 1. " << endl;
       }
       if(ipt < nptbins && ieta == netabins) {

	 myoutput_leak << ieta << " " 
		       << ipt  << " " 
		       << eta[ieta] << " " 
		       << pt[ipt] << " "
		       << leak[ieta-1][ipt] << endl;
	 myoutput_leak1 << ieta << " " 
			<< ipt  << " " 
			<< eta[ieta] << " " 
			<< pt[ipt] << " "
			<< " 1. " << endl;

       }
       if(ipt == nptbins && ieta == netabins) {

	 myoutput_leak << ieta << " " 
		       << ipt  << " " 
		       << eta[ieta] << " " 
		       << pt[ipt] << " "
		       << leak[ieta-1][ipt-1] << endl;
	 myoutput_leak1 << ieta << " " 
			<< ipt  << " " 
			<< eta[ieta] << " " 
			<< pt[ipt] << " "
			<< " 1. " << endl;

       }
     }
   }

   // calculate map of response and tracker efficiency

   cout <<" " << endl;
   cout <<" " << endl;
   cout <<" " << endl;

   for(Int_t ieta = 0; ieta <= netabins; ieta++) {
     for(Int_t ipt = 0; ipt <= nptbins; ipt++) {

       if(ieta < netabins && ipt < nptbins) {

	 if(ntrk[ieta][ipt] != 0 && ntrkrecor[ieta][ipt] != 0) {
	   trkeff[ieta][ipt] = 1.*ntrkreco[ieta][ipt]/ntrk[ieta][ipt];
	   etrkeff[ieta][ipt] = sqrt( trkeff[ieta][ipt]*(1.-trkeff[ieta][ipt])/ntrk[ieta][ipt] );
	   responseF[ieta][ipt] = response[ieta][ipt]/ntrkrecor[ieta][ipt];
	   
	   cout <<" ieta = " << ieta
		<<" eta = " << eta[ieta]
		<<" ipt = " << ipt
		<<" pt = " << pt[ipt]
		<<" ntrkreco = " << ntrkreco[ieta][ipt] 
		<<" ntrkrecor = " << ntrkrecor[ieta][ipt] 
		<<" ntrk = " << ntrk[ieta][ipt]
		<<" eff = " << trkeff[ieta][ipt] 
		<<" +/- " << etrkeff[ieta][ipt] 
		<<" response = " << responseF[ieta][ipt] << endl;
	   hResp->Fill(pt[ipt],eta[ieta],responseF[ieta][ipt]);
	   
	   myoutput_eff << ieta << " " 
			<< ipt  << " " 
			<< eta[ieta] << " " 
			<< pt[ipt] << " "
			<< trkeff[ieta][ipt] << endl;
	   
	   myoutput_resp << ieta << " " 
			 << ipt  << " " 
			 << eta[ieta] << " " 
			 << pt[ipt] << " "
			 << responseF[ieta][ipt] << endl;
	   
	   myoutput_eff1 << ieta << " " 
			 << ipt  << " " 
			 << eta[ieta] << " " 
			 << pt[ipt] << " "
			 << " 1." << endl;
	 } else {

	   trkeff[ieta][ipt] = trkeff[ieta][ipt-1];
	   responseF[ieta][ipt] = responseF[ieta][ipt-1];
	   hResp->Fill(pt[ipt],eta[ieta],responseF[ieta][ipt]);
	   myoutput_eff << ieta << " " 
			<< ipt  << " " 
			<< eta[ieta] << " " 
			<< pt[ipt] << " "
			<< trkeff[ieta][ipt] << endl;
	   
	   myoutput_resp << ieta << " " 
			 << ipt  << " " 
			 << eta[ieta] << " " 
			 << pt[ipt] << " "
			 << responseF[ieta][ipt] << endl;
	   
	   myoutput_eff1 << ieta << " " 
			 << ipt  << " " 
			 << eta[ieta] << " " 
			 << pt[ipt] << " "
			 << " 1." << endl;
	 }
       }

       if(ipt == nptbins && ieta < netabins) {
	 myoutput_eff << ieta << " " 
		      << ipt  << " " 
		      << eta[ieta] << " " 
		      << pt[ipt] << " "
		      << trkeff[ieta][ipt-1] << endl;
	 
	 myoutput_resp << ieta << " " 
		       << ipt  << " " 
		       << eta[ieta] << " " 
		       << pt[ipt] << " "
		       << responseF[ieta][ipt-1] << endl;
	 
	 myoutput_eff1 << ieta << " " 
		       << ipt  << " " 
		       << eta[ieta] << " " 
		       << pt[ipt] << " "
		       << " 1." << endl;
       }
       if(ipt < nptbins && ieta == netabins) {
	 myoutput_eff << ieta << " " 
		      << ipt  << " " 
		      << eta[ieta] << " " 
		      << pt[ipt] << " "
		      << trkeff[ieta-1][ipt] << endl;
	 
	 myoutput_resp << ieta << " " 
		       << ipt  << " " 
		       << eta[ieta] << " " 
		       << pt[ipt] << " "
		       << responseF[ieta-1][ipt] << endl;
	 
	 myoutput_eff1 << ieta << " " 
		       << ipt  << " " 
		       << eta[ieta] << " " 
		       << pt[ipt] << " "
		       << " 1." << endl;
       }
       if(ipt == nptbins && ieta == netabins) {
	 myoutput_eff << ieta << " " 
		      << ipt  << " " 
		      << eta[ieta] << " " 
		      << pt[ipt] << " "
		      << trkeff[ieta-1][ipt-1] << endl;
	 
	 myoutput_resp << ieta << " " 
		       << ipt  << " " 
		       << eta[ieta] << " " 
		       << pt[ipt] << " "
		       << responseF[ieta-1][ipt-1] << endl;
	 
	 myoutput_eff1 << ieta << " " 
		       << ipt  << " " 
		       << eta[ieta] << " " 
		       << pt[ipt] << " "
		       << " 1." << endl;
       }
     }
   }
   
   //
   // plot leakage graph
   Float_t leakage[netabins], eleakage[netabins];
   Float_t MeanNotInter[netabins], MeanErrorNotInter[netabins];
   Float_t MeanInter[netabins], MeanErrorInter[netabins];

   for(Int_t ih=0; ih < netabins; ih++) { 
     //     hEnTrkNotInter[ih]->Write();
     //     hEnTrkInter[ih]->Write();

     MeanNotInter[ih]      = hEnTrkNotInter[ih]->GetMean();
     MeanErrorNotInter[ih] = hEnTrkNotInter[ih]->GetMeanError();
     MeanInter[ih]         = hEnTrkInter[ih]->GetMean();
     MeanErrorInter[ih]    = hEnTrkInter[ih]->GetMeanError();
     leakage[ih] = MeanInter[ih]/MeanNotInter[ih];
     eleakage[ih] = leakage[ih] *
       sqrt( (MeanErrorNotInter[ih]/MeanNotInter[ih])*(MeanErrorNotInter[ih]/MeanNotInter[ih]) +
             (MeanErrorInter[ih]/MeanInter[ih])*(MeanErrorInter[ih]/MeanInter[ih]));

     cout <<" ieta = " << ih <<" MeanNotInter = " << MeanNotInter[ih] <<" +/- " << MeanErrorNotInter[ih]
	  <<" MeanInter = " << MeanInter[ih] <<" +/- " << MeanErrorInter[ih]  
	  <<" leakage = " << leakage[ih] <<" +/- " << eleakage[ih] << endl; 
   }

   TGraph *gleak = new TGraphErrors(netabins,etagr,leakage, eetagr,eleakage);
   TGraph *eMeanNotInter = new TGraphErrors(netabins,etagr,MeanNotInter, eetagr,MeanErrorNotInter);
   TGraph *eMeanInter = new TGraphErrors(netabins,etagr,MeanInter, eetagr,MeanErrorInter);
   //  vs Eta
   setTDRStyle(0,0);
   TCanvas* c3 = new TCanvas("X","Y",1);
   c3->Divide(1,2);
   c3->cd(1);
   TAxis* xaxis = eMeanNotInter->GetXaxis();
   eMeanNotInter->GetXaxis()->SetTitle("#eta");
   eMeanNotInter->GetYaxis()->SetTitle("calo E_{T}^{raw reco} in cone 0.5/p_{T}^{MC}");
   xaxis->SetLimits(0.0,2.4);
   eMeanNotInter->SetMarkerStyle(21);
   //   eMeanNotInter->SetMaximum(0.95);
   //   eMeanNotInter->SetMinimum(0.55);
   eMeanNotInter->SetMaximum(1.);
   eMeanNotInter->SetMinimum(0.);
   eMeanNotInter->Draw("AP");
   eMeanInter->SetMarkerStyle(24);
   eMeanInter->Draw("P");
   TLatex *t = new TLatex();
   t->SetTextSize(0.08);
   t->DrawLatex(0.2,0.9,"CMSSW_1_6_9");
   TLegend *leg = new TLegend(0.2,0.2,0.8,0.4,NULL,"brNDC");
   leg->SetFillColor(10);
   leg->AddEntry(eMeanNotInter,"recontructed tracks 2 < p_{T}^{#pi^{#pm}} < 10 GeV","P");
   leg->AddEntry(eMeanInter,"not reconstructed tracks 2 < p_{T}^{#pi^{#pm}} < 10 GeV","P");
   leg->Draw();

   c3->cd(2);
   TAxis* xaxis = gleak->GetXaxis();
   gleak->GetXaxis()->SetTitle("#eta");
   gleak->GetYaxis()->SetTitle("ratio = <E_{T for lost tracks}^{raw reco} / E_{T for found tracks}^{raw reco}>");
   xaxis->SetLimits(0.0,2.4);
   gleak->SetMarkerStyle(21);
   gleak->SetMaximum(1.0);
   //  leak->SetMinimum(0.7);
   gleak->SetMinimum(0.5);
   gleak->Draw("AP");
   TLatex *t = new TLatex();
   t->SetTextSize(0.08);
   //   t->DrawLatex(0.1,0.75,"<E_{T for lost tracks}^{raw reco} / E_{T for found tracks}^{raw reco}> for p_{T}^{#pi^{#pm}} >2 GeV");
   c3->SaveAs("eMean_vs_eta_pt2-10.gif");
   c3->SaveAs("eMean_vs_eta_pt2-10.eps");

   //  original distribtions
   setTDRStyle(0,0);
   gStyle->SetOptFit(0);
   TCanvas* c4 = new TCanvas("X","Y",1);
   hEnTrkNotInter[0]->GetYaxis()->SetTitle("");
   hEnTrkNotInter[0]->GetXaxis()->SetTitle("calo E_{T}^{raw reco} in cone 0.5/p_{T}^{MC}");
   Double_t scale = 1./hEnTrkNotInter[0]->Integral();
   hEnTrkNotInter[0]->Scale(scale);
   hEnTrkNotInter[0]->SetLineWidth(4);
   hEnTrkNotInter[0]->SetMaximum(0.14);
   //   hEnTrkNotInter[0]->SetMinimum(0.55);
   // Fitting
   Int_t binMax = hEnTrkNotInter[0]->GetMaximumBin();
   TAxis* xaxis = hEnTrkNotInter[0]->GetXaxis();
   Double_t binCenter = xaxis->GetBinCenter(binMax);
   Double_t rms = hEnTrkNotInter[0]->GetRMS();
   Double_t rFitMin = binCenter - 1.5 * rms; 
   Double_t rFitMax = binCenter + 1.5 * rms;
   hEnTrkNotInter[0]->Fit("gaus","","",rFitMin,rFitMax);
   TF1 *fit = hEnTrkNotInter[0]->GetFunction("gaus"); 
   fit->SetLineWidth(4);
   fit->SetLineStyle(2);
   fit->Draw("same");
   scale = 1./hEnTrkInter[0]->Integral();
   hEnTrkInter[0]->Scale(scale);
   hEnTrkInter[0]->SetLineWidth(2);
  // Fitting
   Int_t binMax = hEnTrkInter[0]->GetMaximumBin();
   TAxis* xaxis = hEnTrkInter[0]->GetXaxis();
   Double_t binCenter = xaxis->GetBinCenter(binMax);
   Double_t rms = hEnTrkNotInter[0]->GetRMS();
   Double_t rFitMin = binCenter - 1.5 * rms; 
   Double_t rFitMax = binCenter + 1.5 * rms;
   hEnTrkInter[0]->Fit("gaus","","same",rFitMin,rFitMax);
   TF1 *fit = hEnTrkInter[0]->GetFunction("gaus"); 
   fit->SetLineWidth(2);
   fit->SetLineStyle(2);
   fit->Draw("same");
   TLatex *t = new TLatex();
   t->SetTextSize(0.08);
   t->DrawLatex(0.2,0.9,"CMSSW_1_6_9");
   TLegend *leg = new TLegend(0.15,0.7,0.9,0.9,NULL,"brNDC");
   leg->SetFillColor(10);
   leg->AddEntry(hEnTrkNotInter[0],"recontructed tracks 2< p_{T}^{#pi^{#pm}} < 10 GeV, 0<#eta<0.2","L");
   leg->AddEntry(hEnTrkInter[0],"not reconstructed tracks 2< p_{T}^{#pi^{#pm}} < 10 GeV, 0<#eta<0.2","L");
   leg->Draw();

   c4->SaveAs("eMean_at_eta0.gif");
   c4->SaveAs("eMean_at_eta0.eps");

   //  original distribtions
   setTDRStyle(0,0);
   gStyle->SetOptFit(0);
   TCanvas* c5 = new TCanvas("X","Y",1);
   hEnTrkNotInter[9]->GetYaxis()->SetTitle("");
   hEnTrkNotInter[9]->GetXaxis()->SetTitle("calo E_{T}^{raw reco} in cone 0.5/p_{T}^{MC}");
   Double_t scale = 1./hEnTrkNotInter[9]->Integral();
   hEnTrkNotInter[9]->Scale(scale);
   hEnTrkNotInter[9]->SetLineWidth(4);
   hEnTrkNotInter[9]->SetMaximum(0.17);
   //   hEnTrkNotInter[9]->SetMinimum(0.55);
   // Fitting
   Int_t binMax = hEnTrkNotInter[9]->GetMaximumBin();
   TAxis* xaxis = hEnTrkNotInter[9]->GetXaxis();
   Double_t binCenter = xaxis->GetBinCenter(binMax);
   Double_t rms = hEnTrkNotInter[9]->GetRMS();
   Double_t rFitMin = binCenter - 1.5 * rms; 
   Double_t rFitMax = binCenter + 1.5 * rms;
   hEnTrkNotInter[9]->Fit("gaus","","",rFitMin,rFitMax);
   TF1 *fit = hEnTrkNotInter[9]->GetFunction("gaus"); 
   fit->SetLineWidth(4);
   fit->SetLineStyle(2);
   fit->Draw("same");
   scale = 1./hEnTrkInter[9]->Integral();
   hEnTrkInter[9]->Scale(scale);
   hEnTrkInter[9]->SetLineWidth(2);
  // Fitting
   Int_t binMax = hEnTrkInter[9]->GetMaximumBin();
   TAxis* xaxis = hEnTrkInter[9]->GetXaxis();
   Double_t binCenter = xaxis->GetBinCenter(binMax);
   Double_t rms = hEnTrkNotInter[9]->GetRMS();
   Double_t rFitMin = binCenter - 1.2 * rms; 
   Double_t rFitMax = binCenter + 1.5 * rms;
   hEnTrkInter[9]->Fit("gaus","","same",rFitMin,rFitMax);
   TF1 *fit = hEnTrkInter[9]->GetFunction("gaus"); 
   fit->SetLineWidth(2);
   fit->SetLineStyle(2);
   fit->Draw("same");
   TLatex *t = new TLatex();
   t->SetTextSize(0.08);
   t->DrawLatex(0.2,0.9,"CMSSW_1_6_9");
   TLegend *leg = new TLegend(0.15,0.7,0.9,0.9,NULL,"brNDC");
   leg->SetFillColor(10);
   leg->AddEntry(hEnTrkNotInter[9],"recontructed tracks p_{T}^{#pi^{#pm}} > 2 GeV, 1.8<#eta<2.0","L");
   leg->AddEntry(hEnTrkInter[9],"not reconstructed tracks p_{T}^{#pi^{#pm}} > 2 GeV, 1.8<#eta<2.0","L");
   leg->Draw();

   c5->SaveAs("eMean_at_eta1.9.gif");
   c5->SaveAs("eMean_at_eta1.9.eps");

   TFile efile("response.root","recreate");
   hResp->Write();
   efile.Close();
}
