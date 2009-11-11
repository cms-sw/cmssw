#define JPTPhiRootAnalysis_cxx
#include "JPTPhiRootAnalysis.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

Float_t JPTPhiRootAnalysis::deltaPhi(Float_t phi1, Float_t phi2)
{
  Float_t pi = 3.1415927;
  Float_t dphi = fabs(phi1 - phi2);
  if(dphi >= pi) dphi = 2. * pi - dphi; 
  return dphi;
}

Float_t JPTPhiRootAnalysis::dPhi(Float_t phi1, Float_t phi2) {
  Float_t pi = 3.141592693;
  Float_t result = phi1 - phi2;
  while (result > pi) result -= 2*pi;
  while (result <= -pi) result += 2*pi;
  return result;
}

Float_t JPTPhiRootAnalysis::deltaEta(Float_t eta1, Float_t eta2)
{
  Float_t deta = fabs(eta1-eta2);
  return deta;
}

Float_t JPTPhiRootAnalysis::deltaR(Float_t eta1, Float_t eta2,
		                Float_t phi1, Float_t phi2)
{
  Float_t dr = sqrt( deltaEta(eta1, eta2) * deltaEta(eta1, eta2) +
		     deltaPhi(phi1, phi2) * deltaPhi(phi1, phi2) );
  return dr;
}

void JPTPhiRootAnalysis::setTDRStyle(Int_t ylog) {

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
  tdrStyle->SetStatFontSize(0.03);
  tdrStyle->SetStatTextColor(1);
  tdrStyle->SetStatFormat("6.4g");
  tdrStyle->SetStatBorderSize(1);
  tdrStyle->SetStatH(0.3);
  tdrStyle->SetStatW(0.25);
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
  tdrStyle->SetTitleYOffset(2.05);
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
  tdrStyle->SetOptLogx(0);
  tdrStyle->SetOptLogy(0);
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

void JPTPhiRootAnalysis::Loop()
{
//   In a ROOT session, you can do:
//      Root > .L JPTPhiRootAnalysis.C
//      Root > JPTPhiRootAnalysis t
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

   // resolution plot
   const Int_t nbins = 10;
   const Int_t nh = 10;

   const Int_t nbx = nbins+1;
   const Float_t xbins[nbx]={20.,30.,40.,50.,60.,70.,80.,90.,100.,120.,150.};
   TH1F* hResRaw = new TH1F("hResRaw", "ResRaw", nbins, xbins);
   TH1F* hResJPTInCone = new TH1F("hResJPTInCone", "ResJPTInCone", nbins, xbins);
   TH1F* hResZSP = new TH1F("hResZSP", "ResZSP", nbins, xbins);
   TH1F* hResJPT = new TH1F("hResJPT", "ResJPT", nbins, xbins);
   // scale plot
   TH1F* hScaleRaw = new TH1F("hScaleRaw", "ScaleRaw", nbins, xbins);
   TH1F* hScaleJPTInCone = new TH1F("hScaleJPTInCone", "ScaleJPTInCone", nbins, xbins);
   TH1F* hScaleZSP = new TH1F("hScaleZSP", "ScaleZSP", nbins, xbins);
   TH1F* hScaleJPT = new TH1F("hScaleJPT", "ScaleJPT", nbins, xbins);

   // histogramms

   TH1F* hPhiRaw[nh];
   TH1F* hPhiJPTInCone[nh]; 
   TH1F* hPhiZSP[nh];
   TH1F* hPhiJPT[nh];

   const char* namesPhiRaw[nh] = {"hPhiRaw1","hPhiRaw2","hPhiRaw3","hPhiRaw4","hPhiRaw5","hPhiRaw6","hPhiRaw7","hPhiRaw8","hPhiRaw9","hPhiRaw10"};
   const char* titlePhiRaw[nh] = {"PhiRaw1","PhiRaw2","PhiRaw2","PhiRaw4","PhiRaw5","PhiRaw6","PhiRaw7","PhiRaw8","PhiRaw9","PhiRaw10"};

   const char* namesPhiJPTInCone[nh] = {"hPhiJPTInCone1","hPhiJPTInCone2","hPhiJPTInCone3","hPhiJPTInCone4","hPhiJPTInCone5","hPhiJPTInCone6","hPhiJPTInCone7","hPhiJPTInCone8","hPhiJPTInCone9","hPhiJPTInCone10"};
   const char* titlePhiJPTInCone[nh] = {"PhiJPTInCone1","PhiJPTInCone2","PhiJPTInCone3","PhiJPTInCone4","PhiJPTInCone5","PhiJPTInCone6","PhiJPTInCone7","PhiJPTInCone8","PhiJPTInCone9","PhiJPTInCone10"};

   const char* namesPhiZSP[nh] = {"hPhiZSP1","hPhiZSP2","hPhiZSP3","hPhiZSP4","hPhiZSP5","hPhiZSP6","hPhiZSP7","hPhiZSP8","hPhiZSP9","hPhiZSP10"};
   const char* titlePhiZSP[nh] = {"PhiZSP1","PhiZSP2","PhiZSP3","PhiZSP4","PhiZSP5","PhiZSP6","PhiZSP7","PhiZSP8","PhiZSP9","PhiZSP10"};

   const char* namesPhiJPT[nh] = {"hPhiJPT1","hPhiJPT2","hPhiJPT3","hPhiJPT4","hPhiJPT5","hPhiJPT6","hPhiJPT7","hPhiJPT8","hPhiJPT9","hPhiJPT10"};
   const char* titlePhiJPT[nh] = {"PhiJPT1","PhiJPT2","PhiJPT3","PhiJPT4","PhiJPT5","PhiJPT6","PhiJPT7","PhiJPT8","PhiJPT9","PhiJPT10"};

   for(Int_t ih=0; ih < nh; ih++) { 
     hPhiRaw[ih]  = new TH1F(namesPhiRaw[ih], titlePhiRaw[ih], 101, -0.505, 0.505);
     hPhiJPTInCone[ih]  = new TH1F(namesPhiJPTInCone[ih], titlePhiJPTInCone[ih], 101, -0.505, 0.505);
     hPhiZSP[ih]  = new TH1F(namesPhiZSP[ih], titlePhiZSP[ih], 101, -0.505, 0.505);
     hPhiJPT[ih]  = new TH1F(namesPhiJPT[ih], titlePhiJPT[ih], 101, -0.505, 0.505);
   }

   TH1F * hEtGen  = new TH1F( "hEtGen", "EtGen", 20, 0., 200.);
   TH1F * hEtaGen = new TH1F( "hEtaGen", "EtaGen", 16, 0., 2.1);
   TH1F * hDR     = new TH1F( "hDR", "DR", 100, 0., 10.);
   TH1F * hDeltaR = new TH1F( "hDeltaR", "DeltaR", 50, 0., 0.5);
   hDeltaR->GetXaxis()->SetTitle("#DeltaR(reco,gen) [rads]");
   
   // separate jets by DR
   Float_t DR;
   Float_t DRcut = 2.0;
   Float_t etaMin = 0.0;
   Float_t etaMax = 1.0;

   Long64_t nentries = fChain->GetEntriesFast();

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;

//       bool print = false;
//       if ( fabs( dPhi(PhiJPT1,PhiGen1) ) > 3.1 || 
// 	   fabs( dPhi(PhiJPT2,PhiGen2) ) > 3.1 ) {
// 	print = true;
// 	cout << " *BAD ";
//       } else if ( fabs( dPhi(PhiJPT1,PhiGen1) ) < 0.1 && 
// 		  fabs( dPhi(PhiJPT2,PhiGen2) ) < 0.1 ) {
// 	print = true;
// 	cout << " GOOD ";
//       }
//       if ( print ) {
// 	cout << "Gen" << " "
// 	     << "Raw" << " "
// 	     << "ZSP" << " "
// 	     << "JPT" << " "
// 	     << "diff(JPT,Gen)" << " "
// 	     << "deltaPhi(JPT,Gen)" << " "
// 	     << "dPhi(JPT,Gen)" << " "
// 	     << endl;
// 	cout << " PHI1 "
// 	     << PhiGen1 << " "
// 	     << PhiRaw1 << " "
// 	     << PhiZSP1 << " "
// 	     << PhiJPT1 << " "
// 	     << ( PhiJPT1 - PhiGen1 ) << " "
// 	     << deltaPhi(PhiJPT1,PhiGen1) << " "
// 	     << dPhi(PhiJPT1,PhiGen1) << " "
// 	     << endl;
// 	cout << " PHI2 "
// 	     << PhiGen2 << " "
// 	     << PhiRaw2 << " "
// 	     << PhiZSP2 << " "
// 	     << PhiJPT2 << " "
// 	     << ( PhiJPT2 - PhiGen2 ) << " "
// 	     << deltaPhi(PhiJPT2,PhiGen2) << " "
// 	     << dPhi(PhiJPT2,PhiGen2) << " "
// 	     << endl;
// 	cout << " ET1  "
// 	     << EtGen1 << " "
// 	     << EtRaw1 << " "
// 	     << EtZSP1 << " "
// 	     << EtJPT1 << " "
// 	     << endl;
// 	cout << " ET2  "
// 	     << EtGen2 << " "
// 	     << EtRaw2 << " "
// 	     << EtZSP2 << " "
// 	     << EtJPT2 << " "
// 	     << endl;
//       }

      hDeltaR->Fill( deltaR(EtaJPT1,EtaGen1,PhiJPT1,PhiGen1) );
      hDeltaR->Fill( deltaR(EtaJPT2,EtaGen2,PhiJPT2,PhiGen2) );
      
      // if (Cut(ientry) < 0) continue;
      for(Int_t ih = 0; ih < nh; ih++) {
	if(EtGen1 >= xbins[ih] && EtGen1 < xbins[ih+1] 
	   && DRMAXgjet1 < 0.3
	   && fabs(EtaGen1) > etaMin && fabs(EtaGen1) <= etaMax) {
	  if(EtRaw1/EtGen1 > 0.1) {

 	    if(EtGen2 < 20.) {
	      hPhiRaw[ih]->Fill( dPhi(PhiRaw1,PhiGen1) );
	      hPhiZSP[ih]->Fill( dPhi(PhiZSP1,PhiGen1) );
	      hPhiJPT[ih]->Fill( dPhi(PhiJPT1,PhiGen1) );
 	    } 

 	    if(EtGen2 > 20.) {
	      DR = deltaR(EtaGen1, EtaGen2, PhiGen1, PhiGen2);

// 	      if ( deltaPhi(PhiGen1,PhiGen2) - 
// 		   dPhi(PhiGen1,PhiGen2) > 1.e-3 ) {
// 		cout << "TEST " 
// 		     << PhiGen1 << " " 
// 		     << PhiGen2 << " "
// 		     << deltaPhi(PhiGen1,PhiGen2) << " " 
// 		     << dPhi(PhiGen1,PhiGen2)
// 		     << endl;
// 	      }

	      if(DR > DRcut) {
		hDR->Fill(DR);
		hPhiRaw[ih]->Fill( dPhi(PhiRaw1,PhiGen1) );
		hPhiZSP[ih]->Fill( dPhi(PhiZSP1,PhiGen1) );
		hPhiJPT[ih]->Fill( dPhi(PhiJPT1,PhiGen1) );
		hEtGen->Fill(EtGen1);
		hEtaGen->Fill(EtaGen1);
	      }
 	    }
	  }
	}
	if(EtGen2 >= xbins[ih] && EtGen2 < xbins[ih+1] 
	   && DRMAXgjet2 < 0.3
	   && fabs(EtaGen2) > etaMin && fabs(EtaGen2) <= etaMax) {
	  if(EtRaw2/EtGen2 > 0.1) {

	    DR = deltaR(EtaGen1, EtaGen2, PhiGen1, PhiGen2);
	    if(DR > DRcut) {
	      hDR->Fill(DR);
	      hPhiRaw[ih]->Fill( dPhi(PhiRaw2,PhiGen2) );
	      hPhiZSP[ih]->Fill( dPhi(PhiZSP2,PhiGen2) );
	      hPhiJPT[ih]->Fill( dPhi(PhiJPT2,PhiGen2) );
	      hEtGen->Fill(EtGen2);
	      hEtaGen->Fill(EtaGen2);
	    }
	  }
	}
	
// 	if ( hPhiJPT[ih]->GetEntries() != hPhiRaw[ih]->GetEntries() ) {
// 	  cout << "TESTa "
// 	       << ih << " " 
// 	       << hPhiRaw[ih]->GetEntries() << " "
// 	       << hPhiZSP[ih]->GetEntries() << " "
// 	       << hPhiJPT[ih]->GetEntries() << " "
// 	       << endl;
// 	}

      }
   }

   setTDRStyle(0);
   gStyle->SetOptFit();

   // plot histos
   /*
   TCanvas* cScape = new TCanvas("X","Y",1);
   // raw
   hEtRaw[0]->GetXaxis()->SetTitle("Et reco / Et gen");
   hEtRaw[0]->GetYaxis()->SetTitle("Nev");
   hEtRaw[0]->SetMaximum(100.);
   hEtRaw[0]->Draw("hist");
   //
   // MC
   hEtJPTInCone[0]->SetLineStyle(2);
   hEtJPTInCone[0]->SetLineWidth(1);
   hEtJPTInCone[0]->Draw("same");
   // ZSP
   hEtZSP[0]->SetLineStyle(3);
   hEtZSP[0]->SetLineWidth(4);
   hEtZSP[0]->Draw("same");
   // JPT
   hEtJPT[0]->SetLineStyle(4);
   hEtJPT[0]->SetLineWidth(4);
   hEtJPT[0]->Draw("same");
  // legend
   TLatex *t = new TLatex();
   t->SetTextSize(0.042);
   TLegend *leg = new TLegend(0.45,0.4,0.85,0.8,NULL,"brNDC");
   leg->SetFillColor(10);
   leg->AddEntry(hEtRaw[0],"Raw jets; #mu = 0.60, #sigma = 0.16","L");
   leg->AddEntry(hEtJPTInCone[0],"MC corr; #mu = 1.05, #sigma = 0.17","L");
   leg->AddEntry(hEtZSP[0],"ZSP corr; #mu = 0.73, #sigma = 0.15","L");
   leg->AddEntry(hEtJPT[0],"JPT corr; #mu = 0.89, #sigma = 0.11","L");
   leg->Draw();  
   t->DrawLatex(0.1,450.,"CMSSW160, Z+jets. |#eta ^{jet}|< 1.4, E_{T}^{gen.jet}>20 GeV");
   */

  tdrStyle->SetOptLogy(1);

   //fit JPT
   TCanvas* c1 = new TCanvas("X","Y",1);
   c1->Divide(2,2);
   char name[50];
   for(Int_t ih = 0; ih < nh; ++ih) {
   //   for(Int_t ih = 7; ih < 8; ++ih) {


     // gen energy bin center
     TAxis* xaxisRes = hResJPT->GetXaxis();
     Double_t EbinCenter = xaxisRes->GetBinCenter(ih+1);
     cout <<" bin center = " << EbinCenter << endl;

     Double_t mean = 1000.;
     Double_t meanErr = 1000.;
     Double_t sigma = 1000.;
     Double_t sigmaErr = 1000.;
     Float_t resolution = 1000.;
     Float_t resolutionErr = 1000.;
     
     c1->cd(1);
     // JPT
     // get bin with max content
     Int_t binMax = hPhiJPT[ih]->GetMaximumBin();
     TAxis* xaxis = hPhiJPT[ih]->GetXaxis();
     Double_t binCenter = xaxis->GetBinCenter(binMax);
     Double_t rms = hPhiJPT[ih]->GetRMS();
     Double_t rFitMin = binCenter - 2.0 * rms; 
     Double_t rFitMax = binCenter + 2.0 * rms;
     hPhiJPT[ih]->Fit("gaus","","",rFitMin,rFitMax);
     TF1 *fit = hPhiJPT[ih]->GetFunction("gaus"); 
     gStyle->SetOptFit();
     mean = 1000.;
     meanErr = 1000.;
     sigma = 1000.;
     sigmaErr = 1000.;
     resolution = 1000.;
     resolutionErr = 1000.;
     if ( fit ) {
       mean  = fit->GetParameter(1);
       meanErr  = fit->GetParError(1);
       sigma = fit->GetParameter(2);
       sigmaErr = fit->GetParError(2);
       if ( mean > 0. ) { resolution = sigma/mean; }
       if ( mean > 0. && sigma > 0. ) {
	 resolutionErr = resolution * sqrt((meanErr/mean)*(meanErr/mean) + (sigmaErr/sigma)*(sigmaErr/sigma));
       }
     }
     if ( sigma < 999. ) hResJPT->Fill(EbinCenter,sigma);
     if ( sigmaErr < 999. ) hResJPT->SetBinError(ih+1,sigmaErr);    
     if ( mean < 999. ) hScaleJPT->Fill(EbinCenter,mean);
     if ( meanErr < 999. ) hScaleJPT->SetBinError(ih+1,meanErr);    

     c1->cd(2);
     // ZSP
     // get bin with max content
     binMax = hPhiZSP[ih]->GetMaximumBin();
     xaxis = hPhiZSP[ih]->GetXaxis();
     binCenter = xaxis->GetBinCenter(binMax);
     rms = hPhiZSP[ih]->GetRMS();
     rFitMin = binCenter - 2.0 * rms; 
     rFitMax = binCenter + 2.0 * rms;
     hPhiZSP[ih]->Fit("gaus","","",rFitMin,rFitMax);
     fit = hPhiZSP[ih]->GetFunction("gaus"); 
     mean = 1000.;
     meanErr = 1000.;
     sigma = 1000.;
     sigmaErr = 1000.;
     resolution = 1000.;
     resolutionErr = 1000.;
     if ( fit ) {
       mean  = fit->GetParameter(1);
       meanErr  = fit->GetParError(1);
       sigma = fit->GetParameter(2);
       sigmaErr = fit->GetParError(2);
       if ( mean > 0. ) { resolution = sigma/mean; }
       if ( mean > 0. && sigma > 0. ) {
	 resolutionErr = resolution * sqrt((meanErr/mean)*(meanErr/mean) + (sigmaErr/sigma)*(sigmaErr/sigma));
       }
     }
     if ( sigma < 999. ) hResZSP->Fill(EbinCenter,sigma);
     if ( sigmaErr < 999. ) hResZSP->SetBinError(ih+1,sigmaErr);    
     if ( mean < 999. ) hScaleZSP->Fill(EbinCenter,mean);
     if ( meanErr < 999. ) hScaleZSP->SetBinError(ih+1,meanErr);    

     /*
     c1->cd(3);
     // JPTInCone
     // get bin with max content
     binMax = hPhiJPTInCone[ih]->GetMaximumBin();
     xaxis = hPhiJPTInCone[ih]->GetXaxis();
     binCenter = xaxis->GetBinCenter(binMax);
     rms = hPhiJPTInCone[ih]->GetRMS();
     rFitMin = binCenter - 2.0 * rms; 
     rFitMax = binCenter + 2.0 * rms;
     hPhiJPTInCone[ih]->Fit("gaus","","",rFitMin,rFitMax);
     fit = hPhiJPTInCone[ih]->GetFunction("gaus"); 
     mean  = fit->GetParameter(1);
     meanErr  = fit->GetParError(1);
     sigma = fit->GetParameter(2);
     sigmaErr = fit->GetParError(2);
     resolution = sigma/mean;
     resolutionErr = resolution * sqrt((meanErr/mean)*(meanErr/mean) + (sigmaErr/sigma)*(sigmaErr/sigma));
     hResJPTInCone->Fill(EbinCenter,resolution);
     hResJPTInCone->SetBinError(ih+1, resolutionErr);   
     hScaleJPTInCone->Fill(EbinCenter,mean);
     hScaleJPTInCone->SetBinError(ih+1,meanErr);    
     */

     c1->cd(4);
     // RAW
     // get bin with max content
     binMax = hPhiRaw[ih]->GetMaximumBin();
     xaxis = hPhiRaw[ih]->GetXaxis();
     binCenter = xaxis->GetBinCenter(binMax);
     rms = hPhiRaw[ih]->GetRMS();
     rFitMin = binCenter - 2.0 * rms; 
     rFitMax = binCenter + 2.0 * rms;
     hPhiRaw[ih]->Fit("gaus","","",rFitMin,rFitMax);
     fit = hPhiRaw[ih]->GetFunction("gaus"); 
     mean = 1000.;
     meanErr = 1000.;
     sigma = 1000.;
     sigmaErr = 1000.;
     resolution = 1000.;
     resolutionErr = 1000.;
     if ( fit ) {
       mean  = fit->GetParameter(1);
       meanErr  = fit->GetParError(1);
       sigma = fit->GetParameter(2);
       sigmaErr = fit->GetParError(2);
       if ( mean > 0. ) { resolution = sigma/mean; }
       if ( mean > 0. && sigma > 0. ) {
	 resolutionErr = resolution * sqrt((meanErr/mean)*(meanErr/mean) + (sigmaErr/sigma)*(sigmaErr/sigma));
       }
     }
     if ( sigma < 999. ) hResRaw->Fill(EbinCenter,sigma);
     if ( sigmaErr < 999. ) hResRaw->SetBinError(ih+1, sigmaErr);    
     if ( mean < 999. ) hScaleRaw->Fill(EbinCenter,mean);
     if ( meanErr < 999. ) hScaleRaw->SetBinError(ih+1,meanErr);
     sprintf(name,"hCalo1_%d.eps",ih);
     c1->SaveAs(name);
   }

  tdrStyle->SetOptLogy(0);

   /*
   TCanvas* c3 = new TCanvas("X","Y",1);

   hResJPT->GetXaxis()->SetTitle("E_{T} Gen, GeV");
   hResJPT->GetYaxis()->SetTitle("Energy resolution, % ");

   hResJPT->SetMaximum(0.45);
   hResJPT->SetMinimum(0.08);
   hResJPT->SetMarkerStyle(21);
   hResJPT->SetMarkerSize(1.2);
   hResJPT->Draw("histPE1");

   hResZSP->SetMarkerSize(1.0);
   hResZSP->SetMarkerStyle(24);
   hResZSP->Draw("samePE1");

   hResRaw->SetMarkerSize(1.5);
   hResRaw->SetMarkerStyle(22);
   hResRaw->Draw("samePE1");

   TLatex *t = new TLatex();
   t->SetTextSize(0.042);
   TLegend *leg = new TLegend(0.45,0.5,0.85,0.8,NULL,"brNDC");
   leg->SetFillColor(10);
   leg->AddEntry(hResJPT,"with ZSP+JPT corr","P");
   leg->AddEntry(hResZSP,"with ZSP corr","P");
   leg->AddEntry(hResRaw,"Raw calo jets","P");
   leg->Draw();  
   t->DrawLatex(40,0.42,"CMSSW160, Z+jets. |#eta ^{jet}|< 1.4");

   c3->SaveAs("resRawZSPJPT.eps");
   c3->SaveAs("resRawZSPJPT.eps");
   */

   // save histo on disk

   //   TFile efile("CTF.root","recreate");

   TFile efile("test.root","recreate");
   hResRaw->Write();
   hResJPTInCone->Write();
   hResJPT->Write();
   hScaleRaw->Write();
   hScaleJPTInCone->Write();
   hScaleJPT->Write();
   efile.Close();

   TCanvas* c40 = new TCanvas("X","Y",1);

   hResJPT->GetXaxis()->SetTitle("E_{T}^{gen} [GeV]");
   hResJPT->GetYaxis()->SetTitle("#phi resolution [rads]");
   hResJPT->GetYaxis()->SetTitleOffset(1.7);

   hResJPT->SetMaximum(0.25);
   hResJPT->SetMinimum(0.00);
   hResJPT->SetMarkerStyle(21);
   hResJPT->SetMarkerSize(1.2);
   hResJPT->Draw("histPE1");

   //   hResJPTInCone->SetMarkerSize(1.0);
   //   hResJPTInCone->SetMarkerStyle(24);
   //   hResJPTInCone->Draw("samePE1");
   hResZSP->SetMarkerSize(1.0);
   hResZSP->SetMarkerStyle(24);
   hResZSP->Draw("samePE1");

   hResRaw->SetMarkerSize(1.5);
   hResRaw->SetMarkerStyle(26);
   hResRaw->Draw("samePE1");
   
   TLatex *t = new TLatex();
   TLegend *leg = new TLegend(0.6,0.45,0.9,0.6,NULL,"brNDC");
   leg->SetFillColor(10);
   leg->AddEntry(hResRaw,"RAW CaloJets","P");
   leg->AddEntry(hResZSP,"ZSP corrected","P");
   leg->AddEntry(hResJPT,"ZSP+JPT corrected","P");
   leg->Draw();  
   t->SetTextSize(0.04);
   t->DrawLatex(50,0.23,"CMSSW_3_4_0");
   t->DrawLatex(50,0.21,"RelVal QCD 80-120 GeV, |#eta^{jet}| < 1.0");

   c40->SaveAs("resJPT219.eps");

   /*
   TCanvas* c1 = new TCanvas("X","Y",1);

   hScaleJPT->GetXaxis()->SetTitle("E_{T} Gen, GeV");
   hScaleJPT->GetYaxis()->SetTitle("E_{T}^{reco}/E_{T}^{gen}");

   hScaleJPT->SetMaximum(1.2);
   hScaleJPT->SetMinimum(0.3);
   hScaleJPT->SetMarkerStyle(21);
   hScaleJPT->SetMarkerSize(1.2);
   hScaleJPT->Draw("histPE1");

   hScaleZSP->SetMarkerSize(1.0);
   hScaleZSP->SetMarkerStyle(24);
   hScaleZSP->Draw("samePE1");

   hScaleRaw->SetMarkerSize(1.5);
   hScaleRaw->SetMarkerStyle(22);
   hScaleRaw->Draw("samePE1");

   // PF jets
   //   TFile* file = new TFile("Jetsresolution_curve_short.root");
   //   gResp_iterativeCone5PFJets->SetMarkerSize(1.5);
   //   gResp_iterativeCone5PFJets->SetMarkerStyle(22);
   //   gResp_iterativeCone5PFJets->Draw("P");
   //

   TLatex *t = new TLatex();
   t->SetTextSize(0.042);
   TLegend *leg = new TLegend(0.5,0.15,0.9,0.35,NULL,"brNDC");
   leg->SetFillColor(10);
   leg->AddEntry(hScaleJPT,"with ZSP+JPT corr","P");
   leg->AddEntry(hScaleZSP,"with ZSP corr","P");
   leg->AddEntry(hScaleRaw,"Raw calo jets","P");
   leg->Draw();  
   t->DrawLatex(40,1.12,"CMSSW160, Z+jets. |#eta ^{jet}|< 1.4");

   c1->SaveAs("ScaleRawZSPJPT.eps");
   c1->SaveAs("ScaleRawZSPJPT.eps");
   */

   TCanvas* c20 = new TCanvas("X","Y",1);

   hScaleJPT->GetXaxis()->SetTitle("E_{T}^{gen} [GeV]");
   hScaleJPT->GetYaxis()->SetTitle("#phi bias [rads]");
   hScaleJPT->GetYaxis()->SetTitleOffset(1.7);

   hScaleJPT->SetMaximum(0.01);
   hScaleJPT->SetMinimum(-0.01);
   hScaleJPT->SetMarkerStyle(21);
   hScaleJPT->SetMarkerSize(1.2);
   hScaleJPT->Draw("histPE1");

   //   hScaleJPTInCone->SetMarkerSize(1.0);
   //   hScaleJPTInCone->SetMarkerStyle(24);
   //   hScaleJPTInCone->Draw("samePE1");

   hScaleZSP->SetMarkerSize(1.0);
   hScaleZSP->SetMarkerStyle(24);
   hScaleZSP->Draw("samePE1");
   
   hScaleRaw->SetMarkerSize(1.5);
   hScaleRaw->SetMarkerStyle(26);
   hScaleRaw->Draw("samePE1");

   TLatex *t = new TLatex();
   TLegend *leg = new TLegend(0.6,0.2,0.9,0.35,NULL,"brNDC");
   leg->SetFillColor(10);
   leg->AddEntry(hScaleRaw,"RAW CaloJets","P");
   leg->AddEntry(hScaleZSP,"ZSP corrected","P");
   leg->AddEntry(hScaleJPT,"ZSP+JPT corrected","P");
   leg->Draw();  
   t->SetTextSize(0.04);
   t->DrawLatex(50,0.008,"CMSSW_3_4_0");
   t->DrawLatex(50,0.0065,"RelVal QCD 80-120 GeV, |#eta^{jet}| < 1.0");

   c20->SaveAs("ScaleJPT219.eps");

   /*
   TCanvas* c10 = new TCanvas("X","Y",1);
   c10->Divide(1,2);
   // EtGen
   c10->cd(1);
   hEtGen->GetXaxis()->SetTitle("E_{T} gen, GeV");
   hEtGen->GetYaxis()->SetTitle("Nev");
   hEtGen->Draw("hist");

   c10->cd(2);
   hEtaGen->GetXaxis()->SetTitle("| #eta | gen");
   hEtaGen->GetYaxis()->SetTitle("Nev");
   hEtaGen->SetMinimum(0.);
   hEtaGen->Draw("hist");

   //   c10->cd(3);
   //   hDR->GetXaxis()->SetTitle("#Delta R");
   //   hDR->GetYaxis()->SetTitle("Nev");
   //   hDR->Draw("hist");
   c10->SaveAs("JetEtEta.eps");
   */

   TCanvas* tmp = new TCanvas("X","Y",1);
   tdrStyle->SetOptStat(1111);
   //tdrStyle->SetOptStat("mr");
   hDeltaR->Draw();
   tmp->SaveAs("DeltaR.eps");
   
}
