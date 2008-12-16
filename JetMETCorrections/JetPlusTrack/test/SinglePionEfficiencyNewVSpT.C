#define SinglePionEfficiencyNewVSpT_cxx
#include "SinglePionEfficiencyNewVSpT.h"
#include <TStyle.h>
#include <TCanvas.h>

void SinglePionEfficiencyNewVSpT::setTDRStyle(Int_t xlog, Int_t ylog) {

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
  tdrStyle->SetTitleYOffset(1.1);
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


void SinglePionEfficiencyNewVSpT::Loop()
{
//   In a ROOT session, you can do:
//      Root > .L SinglePionEfficiencyNewVSpT.C
//      Root > SinglePionEfficiencyNewVSpT t
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
   const Int_t nptbins = 17;
   const Int_t nptcuts = nptbins+1;
   //   const Double_t pt[nptcuts]={0.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40., 50.0};
   const Double_t pt[nptcuts]={1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40., 50.0, 60.0};
   //                              0    1    2    3    4    5    6    7    8    9
   // energy ecal+hcal in different matrices
   TProfile* hprEH11x5  = new  TProfile("hprEH11x5","prEH11x5",nptbins, pt, -10., 10.);
   TProfile* hprEH11x3  = new  TProfile("hprEH11x3","prEH11x3",nptbins, pt, -10., 10.);
   TProfile* hprEH7x5   = new  TProfile("hprEH7x5","prEH7x5",nptbins, pt, -10., 10.);
   TProfile* hprEH7x3   = new  TProfile("hprEH7x3","prEH7x3",nptbins, pt, -10., 10.);
   // energy in hcal matrix
   TProfile* hprH5  = new  TProfile("hprH5","prH5",nptbins, pt, -10., 10.);
   TProfile* hprH3  = new  TProfile("hprH3","prH3",nptbins, pt, -10., 10.);

   // 
   TH1F * hH5_1_2GeV    = new TH1F( "hH5_1_2GeV", "H5_1_2GeV", 40, -2., 2.);
   TH1F * hH3_1_2GeV    = new TH1F( "hH3_1_2GeV", "H3_1_2GeV", 40, -2., 2.);
   TH1F * hH5_3_4GeV    = new TH1F( "hH5_3_4GeV", "H5_3_4GeV", 40, -2., 2.);
   TH1F * hH3_3_4GeV    = new TH1F( "hH3_3_4GeV", "H3_3_4GeV", 40, -2., 2.);
   TH1F * hH5_5_10GeV   = new TH1F( "hH5_5_10GeV", "H5_5_10GeV", 40, -2., 2.);
   TH1F * hH3_5_10GeV   = new TH1F( "hH3_5_10GeV", "H3_5_10GeV", 40, -2., 2.);

   //
   TH1F * hE11_1_2GeV   = new TH1F( "hE11_1_2GeV", "E11_1_2GeV", 100, -2., 2.);
   TH1F * hE11_3_4GeV   = new TH1F( "hE11_3_4GeV", "E11_3_4GeV", 100, -2., 2.);
   TH1F * hE11_5_10GeV  = new TH1F( "hE11_5_10GeV", "E11_5_10GeV", 100, -2., 2.);

   //
   TH1F * hE11H5_1_2GeV   = new TH1F( "hE11H5_1_2GeV", "E11H5_1_2GeV", 40, -2., 2.);
   TH1F * hE11H5_3_4GeV   = new TH1F( "hE11H5_3_4GeV", "E11H5_3_4GeV", 40, -2., 2.);
   TH1F * hE11H5_5_10GeV  = new TH1F( "hE11H5_5_10GeV", "E11H5_5_10GeV", 40, -2., 2.);

   //   TH1F * hEH11x5 = new TH1F( "hEH11x5", "EH11x5", 200, -2., 2.);
   //   TH2F * hresp2 = new TH2F( "hresp2", "resp2", 200, -2., 2.,200, -2., 2.);

   // prepare for graph
   Float_t ptgr[nptbins], eptgr[nptbins];
   for(Int_t i = 0; i < nptbins; i++) {
     ptgr[i]  = 0.5*(pt[i]+pt[i+1]);
     eptgr[i] = 0.5*(pt[i+1]-pt[i]);
     cout <<" i = " << i <<" pT = " << ptgr[i] <<" err pT = " << eptgr[i] << endl;
   } 

   // number of eta bins and intervals
   const Int_t netabins = 12;
   const Int_t netacuts = netabins+1;
   Float_t eta[netacuts]={0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4};
   cout <<"  Eta bins " << endl;
   // prepare eta points for graph
   Float_t etagr[netabins], eetagr[netabins];
   for(Int_t i = 0; i < netabins; i++) {
     etagr[i]  = 0.5*(eta[i]+eta[i+1]);
     eetagr[i] = 0.5*(eta[i+1]-eta[i]);
     cout <<" i = " << i <<" Eta = " << etagr[i] <<" err Eta = " << eetagr[i] << endl;
   } 

   // ===> for pi- and pi+
   // N total and N reco tracks 
   // efficiency and error as a function of pT 
   Int_t ntrk[nptbins]     = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
   Int_t ntrkreco[nptbins] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
   Int_t ntrkrecor[nptbins] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
   Float_t trkeff[nptbins], etrkeff[nptbins];
   Double_t responceVSpt[nptbins]  = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
   Double_t responceVSptF[nptbins];
   Double_t eresponceVSpt[nptbins] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

   // N total and N reco tracks
   // efficiency and error as a function of eta 
   Int_t ntrketa[netabins]     = {0,0,0,0,0,0,0,0,0,0,0,0};
   Int_t ntrketareco[netabins] = {0,0,0,0,0,0,0,0,0,0,0,0};
   Int_t ntrketarecor[netabins] = {0,0,0,0,0,0,0,0,0,0,0,0};
   Float_t trketaeff[netabins], etrketaeff[netabins];
   Double_t responceVSeta[nptbins]  = {0,0,0,0,0,0,0,0,0,0,0,0};
   Double_t responceVSetaF[nptbins];
   Double_t eresponceVSeta[nptbins] = {0,0,0,0,0,0,0,0,0,0,0,0};

   // 
   Int_t Ntot = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      Ntot = Ntot + 1;
      // if (Cut(ientry) < 0) continue;
      // evaluate as a function of pT
      for(Int_t i = 0; i < nptbins; i++) {
	// ==> pi+
	if(ptSim1 >= pt[i] && ptSim1 < pt[i+1]) {
	  ntrk[i] = ntrk[i]+1;
	  // number of reco tracks
	  if(drTrk1 < 0.01 && purityTrk1 == 1) {
	    ntrkreco[i] = ntrkreco[i]+1;
	    Double_t theta = 2.*atan(exp(-etaSim1));
	    Double_t eSim1 = ptSim1/sin(theta);
	    if(e1ECAL11x11 > -1000. && e1HCAL5x5 > -1000. && fabs(etaSim1) < 1.0) { 
	    //	    if(e1ECAL11x11 > -1000. && e1HCAL5x5 > -1000. && fabs(etaSim1) < 1000) { 
	      ntrkrecor[i] = ntrkrecor[i]+1;

	      Double_t e_ecal11 = e1ECAL11x11/eSim1;
	      Double_t e_ecal7  = e1ECAL7x7/eSim1;
	      Double_t e_hcal5  = e1HCAL5x5/eSim1;
	      Double_t e_hcal3  = e1HCAL3x3/eSim1;

	      Double_t e_ecalhcal11x5 = e_ecal11 + e_hcal5;
	      Double_t e_ecalhcal11x3 = e_ecal11 + e_hcal3;
	      Double_t e_ecalhcal7x5  = e_ecal7 + e_hcal5;
	      Double_t e_ecalhcal7x3  = e_ecal7 + e_hcal3;
 
	      responceVSpt[i] = responceVSpt[i] + e_ecalhcal11x5;

	      hprEH11x5->Fill(ptSim1,e_ecalhcal11x5,1.);
	      hprEH11x3->Fill(ptSim1,e_ecalhcal11x3,1.);
	      hprEH7x5->Fill(ptSim1,e_ecalhcal7x5,1.);
	      hprEH7x3->Fill(ptSim1,e_ecalhcal7x3,1.);

	      hprH5->Fill(ptSim1,e_hcal5,1.);
	      hprH3->Fill(ptSim1,e_hcal3,1.);

	      //	      if(i == nptbins-1) {
	      if(i < 5) {
		hH5_1_2GeV->Fill(e_hcal5,1.);
		hH3_1_2GeV->Fill(e_hcal3,1.);
		hE11_1_2GeV->Fill(e_ecal11,1.);	      
		hE11H5_1_2GeV->Fill(e_ecalhcal11x5,1.);	      
	      }
	      if(i == 7) {
		hH5_3_4GeV->Fill(e_hcal5,1.);
		hH3_3_4GeV->Fill(e_hcal3,1.);
		hE11_3_4GeV->Fill(e_ecal11,1.);	      
		hE11H5_3_4GeV->Fill(e_ecalhcal11x5,1.);	      
	      }
	      if(i == 9) {
		hH5_5_10GeV->Fill(e_hcal5,1.);
		hH3_5_10GeV->Fill(e_hcal3,1.);
		hE11_5_10GeV->Fill(e_ecal11,1.);	      
		hE11H5_5_10GeV->Fill(e_ecalhcal11x5,1.);	      
	      }
	    }
	  }
	}
	// ==> pi-
	if(ptSim2 >= pt[i] && ptSim2 < pt[i+1]) {
	  ntrk[i] = ntrk[i]+1;
	  // number of reco tracks
	  if(drTrk2 < 0.01 && purityTrk2 == 1) {
	    ntrkreco[i] = ntrkreco[i]+1;
	    Double_t theta = 2.*atan(exp(-etaSim2));
	    Double_t eSim2 = ptSim2/sin(theta);
	    if(e2ECAL11x11 > -1000. && e2HCAL5x5 > -1000. && fabs(etaSim2) < 1.0) { 
	    //	    if(e2ECAL11x11 > -1000. && e2HCAL5x5 > -1000. && fabs(etaSim2) < 1000.) { 
	      ntrkrecor[i] = ntrkrecor[i]+1;

	      Double_t e_ecal11 = e2ECAL11x11/eSim2;
	      Double_t e_ecal7  = e2ECAL7x7/eSim2;
	      Double_t e_hcal5  = e2HCAL5x5/eSim2;
	      Double_t e_hcal3  = e2HCAL3x3/eSim2;

	      Double_t e_ecalhcal11x5 = e_ecal11 + e_hcal5;
	      Double_t e_ecalhcal11x3 = e_ecal11 + e_hcal3;
	      Double_t e_ecalhcal7x5  = e_ecal7 + e_hcal5;
	      Double_t e_ecalhcal7x3  = e_ecal7 + e_hcal3;
 
	      responceVSpt[i] = responceVSpt[i] + e_ecalhcal11x5;

	      hprEH11x5->Fill(ptSim2,e_ecalhcal11x5,1.);
	      hprEH11x3->Fill(ptSim2,e_ecalhcal11x3,1.);
	      hprEH7x5->Fill(ptSim2,e_ecalhcal7x5,1.);
	      hprEH7x3->Fill(ptSim2,e_ecalhcal7x3,1.);

	      hprH5->Fill(ptSim2,e_hcal5,1.);
	      hprH3->Fill(ptSim2,e_hcal3,1.);

	      //	      if(i == nptbins-1) {
	      if(i < 5) {
		hH5_1_2GeV->Fill(e_hcal5,1.);
		hH3_1_2GeV->Fill(e_hcal3,1.);
		hE11_1_2GeV->Fill(e_ecal11,1.);	      
		hE11H5_1_2GeV->Fill(e_ecalhcal11x5,1.);	      
	      }
	      if(i == 7) {
		hH5_3_4GeV->Fill(e_hcal5,1.);
		hH3_3_4GeV->Fill(e_hcal3,1.);
		hE11_3_4GeV->Fill(e_ecal11,1.);	      
		hE11H5_3_4GeV->Fill(e_ecalhcal11x5,1.);	      
	      }
	      if(i == 9) {
		hH5_5_10GeV->Fill(e_hcal5,1.);
		hH3_5_10GeV->Fill(e_hcal3,1.);
		hE11_5_10GeV->Fill(e_ecal11,1.);	      
		hE11H5_5_10GeV->Fill(e_ecalhcal11x5,1.);	      
	      }
	    }
	  }
	}
      }
      // Nick
   // evaluate efficiency as a function on eta
      for(Int_t i = 0; i < netabins; i++) {
	// ==> pi+
	if(fabs(etaSim1) >= eta[i] && fabs(etaSim1) < eta[i+1]) {
	  // number of sim tracks in pt interval
	  ntrketa[i] = ntrketa[i]+1;
	  // number of reco tracks
	  if(drTrk1 < 0.04 && purityTrk1 >= 0.7) {
	    Double_t theta = 2.*atan(exp(-etaSim1));
	    Double_t eSim1 = ptSim1/sin(theta);
	    ntrketareco[i] = ntrketareco[i]+1;
	    if(e1ECAL11x11 > -1000. && e1HCAL5x5 > -1000.) { 
	      ntrketarecor[i] = ntrketarecor[i]+1;
	      responceVSeta[i] = responceVSeta[i] + (e1ECAL7x7 + e1HCAL3x3)/eSim1;
	    }
	  }
	}
	// ==> pi-
	if(fabs(etaSim2) >= eta[i] && fabs(etaSim2) < eta[i+1]) {
	  // number of sim tracks in pt interval
	  ntrketa[i] = ntrketa[i]+1;
	  // number of reco tracks
	  if(drTrk2 < 0.04 && purityTrk2 >= 0.7) {
	    ntrketareco[i] = ntrketareco[i]+1;
	    Double_t theta = 2.*atan(exp(-etaSim2));
	    Double_t eSim2 = ptSim2/sin(theta);
	    if(e2ECAL11x11 > -1000. && e2HCAL5x5 > -1000.) { 
	      ntrketarecor[i] = ntrketarecor[i]+1;
	      responceVSeta[i] = responceVSeta[i] + (e2ECAL7x7 + e2HCAL3x3)/eSim2;
	    }
	  }
	}
      }
   }

   // calculate efficiency and full graph
   for(Int_t i = 0; i < nptbins; i++) {
     if(ntrk[i] > 0) {
       trkeff[i] = 1.*ntrkreco[i]/ntrk[i];
       etrkeff[i] = sqrt( trkeff[i]*(1.-trkeff[i])/ntrk[i] ); 
       //       responceVSptF[i] = responceVSpt[i]/ntrkrecor[i];
       cout <<" i = " << i 
	    <<" pt interval = " << pt[i] <<" - " << pt[i+1]
	    <<" ntrkreco[i] = " << ntrkreco[i] 
	    <<" ntrkrecor[i] = " << ntrkrecor[i] 
	    <<" ntrk[i] = " << ntrk[i]
	    <<" eff = " << trkeff[i] << endl;
	 //       	    <<" responce = " << responceVSptF[i] << endl; 
     }
   }
   // calculate efficiency vs Eta and full graph
   cout <<" Efficiency vs Eta " << endl;
   for(Int_t i = 0; i < netabins; i++) {
     if(ntrketa[i] > 0) {
       trketaeff[i] = 1.*ntrketareco[i]/ntrketa[i];
       etrketaeff[i] = sqrt( trketaeff[i]*(1.-trketaeff[i])/ntrketa[i] );
       responceVSetaF[i] = responceVSeta[i]/ntrketarecor[i];
       cout <<" i = " << i 
	    <<" eta interval = " << eta[i] <<" - " << eta[i+1]
	    <<" ntrketareco[i] = " << ntrketareco[i] 
	    <<" ntrketa[i] = " << ntrketa[i]
	    <<" eff = " << trketaeff[i] 
	    <<" responce = " << responceVSetaF[i] << endl; 
     }
   }

   // create graph
   //  vs pT
   setTDRStyle(1,0);
   TCanvas* c1 = new TCanvas("X","Y",1);
   TGraph *grpt  = new TGraphErrors(nptbins,ptgr,trkeff,eptgr,etrkeff);
   TAxis* xaxis = grpt->GetXaxis();
   grpt->GetXaxis()->SetTitle("p_{T}, GeV");
   grpt->GetYaxis()->SetTitle("track finding efficiency");
   xaxis->SetLimits(0.8,50.);
   grpt->SetMarkerStyle(21);
   grpt->SetMaximum(0.9);
   grpt->SetMinimum(0.6);
   grpt->Draw("AP");
   TLatex *t = new TLatex();
   t->SetTextSize(0.042);
   //   TLegend *leg = new TLegend(0.5,0.2,0.7,0.35,NULL,"brNDC");
   //   leg->SetFillColor(10);
   //   leg->AddEntry(grpt,"#pi ^{+} and #pi ^{-}","P");
   //   leg->Draw();  
   t->DrawLatex(1.,0.85,"CMSSW169, single #pi ^{+} and #pi ^{-}. |#eta ^{#pi}|< 2.5");
   c1->SaveAs("trkeff_vs_pt.gif");
   c1->SaveAs("trkeff_vs_pt.eps");

   setTDRStyle(0,0);
   //  vs Eta
   TCanvas* c2 = new TCanvas("X","Y",1);
   TGraph *greta = new TGraphErrors(netabins,etagr,trketaeff,eetagr,etrketaeff);
   TAxis* xaxis = greta->GetXaxis();
   greta->GetXaxis()->SetTitle("#eta");
   greta->GetYaxis()->SetTitle("track finding efficiency");
   xaxis->SetLimits(0.0,2.4);
   greta->SetMarkerStyle(21);
   greta->SetMaximum(1.0);
   greta->SetMinimum(0.50);
   greta->Draw("AP");
   TLatex *t = new TLatex();
   t->SetTextSize(0.042);
   //   TLegend *leg = new TLegend(0.5,0.2,0.7,0.35,NULL,"brNDC");
   //   leg->SetFillColor(10);
   //   leg->AddEntry(greta,"#pi ^{+} and #pi ^{-}","P");
   //   leg->Draw();  
   t->DrawLatex(0.3,0.87,"CMSSW217, single #pi ^{+} and #pi ^{-}");
   t->DrawLatex(0.8,0.85,"1 < p^{#pi^{#pm}} < 50 GeV");
   c2->SaveAs("trkeff_vs_eta.gif");
   c2->SaveAs("trkeff_vs_eta.eps");
   
   cout <<" Ntot = " << Ntot << endl;

   /*
   setTDRStyle(1,0);
   //
   TCanvas* c3 = new TCanvas("X","Y",1);
   TAxis* xaxis = hEH->GetXaxis();
   hEH->GetXaxis()->SetTitle("p_{T} of #pi ^{+} and #pi ^{-}, GeV");
   hEH->GetYaxis()->SetTitle("(ECAL11x11+HCAL5x5)/E^{true}");
   //   xaxis->SetLimits(1.2,50.);
   hEH->SetMarkerStyle(21);
   hEH->SetMaximum(0.9);
   hEH->SetMinimum(0.2);
   hEH->Draw();
   c3->SaveAs("EmatrixtWithZSP11x11.5x5.gif");

   setTDRStyle(1,0);
   //
   TCanvas* c4 = new TCanvas("X","Y",1);
   TAxis* xaxis = hEmatrix->GetXaxis();
   hHmatrix->GetXaxis()->SetTitle("p_{T} of #pi ^{+} and #pi ^{-}, GeV");
   hHmatrix->GetYaxis()->SetTitle("HCAL3x3/E^{true}");
   //   xaxis->SetLimits(1.2,50.);
   hHmatrix->SetMarkerStyle(21);
   hHmatrix->SetMaximum(0.9);
   hHmatrix->SetMinimum(-0.2);
   hHmatrix->Draw();
   c4->SaveAs("HmatrixWithNoZSP3x3.gif");

   setTDRStyle(0,0);
   TCanvas* c5 = new TCanvas("X","Y",1);
   hresp2->Draw("hist");

   setTDRStyle(0,0);
   TCanvas* c6 = new TCanvas("X","Y",1);
   hhcal->Draw("hist");
   */

   TFile efile("sr_barrel.root","recreate");

   hprEH11x5->Write();
   hprEH11x3->Write();
   hprEH7x5->Write();
   hprEH7x3->Write();
   hprH5->Write();
   hprH3->Write();
   hH5_1_2GeV->Write();
   hH3_1_2GeV->Write();
   hH5_3_4GeV->Write();
   hH3_3_4GeV->Write();
   hH5_5_10GeV->Write();
   hH3_5_10GeV->Write();
   hE11_1_2GeV->Write();
   hE11_3_4GeV->Write();
   hE11_5_10GeV->Write();
   hE11H5_1_2GeV->Write();
   hE11H5_3_4GeV->Write();
   hE11H5_5_10GeV->Write();

   efile.Close();
}
