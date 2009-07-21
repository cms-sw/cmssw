// Original Author:  Viola Sordini &b Joanna Weng July  2009

#define JPTBjetRootAnalysis_cxx
#include "JPTBjetRootAnalysis.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>



void JPTBjetRootAnalysis::Loop()
{
  //   In a ROOT session, you can do:
  //      Root > .L JPTBjetRootAnalysis.C
  //      Root > JPTBjetRootAnalysis t
  //      Root > t.GetEntry(12); // Fill t data members with entry number 12
  //      Root > t.Show();       // Show values of entry 12
  //      Root > t.Show(16);     // Read and show values of entry 16
  //      Root > t.Loop();       // Loop on all entries
  //
 
  // Analysis Settings here  ----------------------------------- 
  // separate jets by DR
  Float_t drmax=0.3;
  Float_t DRcut = 1.;
  Float_t etaMin = 0.;
  Float_t etaMax = 1.;
  Float_t ptMin=0.;
  //----------------------

  if (fChain == 0) return;
  // resolution plot
  const Int_t nbins = 10;
  const Int_t nh = 10;
  int jet1=0;
  int jet2=0;
  // Init Histograms -----------------------------------
  const Int_t nbx = nbins+1;
  const Float_t xbins[nbx]={20.,30.,40.,50.,60.,70.,80.,90.,100.,120.,150.};
  TH1F* hResRaw = new TH1F("hResRaw", "ResRaw", nbins, xbins);
  TH1F* hResJPTInCone = new TH1F("hResJPTInCone", "ResJPTInCone", nbins, xbins);
  TH1F* hResJPT = new TH1F("hResJPT", "ResJPT", nbins, xbins); 
  TH1F* hResJPTelectrons = new TH1F("hResJPTelectrons", "ResJPTelectrons", nbins, xbins);
  TH1F* hResJPTmuons = new TH1F("hResJPTmuons", "ResJPTmuons", nbins, xbins);
  // scale plot
  TH1F* hScaleRaw = new TH1F("hScaleRaw", "ScaleRaw", nbins, xbins);
  TH1F* hScaleJPTInCone = new TH1F("hScaleJPTInCone", "ScaleJPTInCone", nbins, xbins);
  TH1F* hScaleJPT = new TH1F("hScaleJPT", "ScaleJPT", nbins, xbins);
  TH1F* hScaleJPTelectrons = new TH1F("hScaleJPTelectrons", "ScaleJPTelectrons", nbins, xbins);
  TH1F* hScaleJPTmuons = new TH1F("hScaleJPTmuons", "ScaleJPTmuons", nbins, xbins);
  // histogramms

  TH1F* hEtRaw[nh];
  TH1F* hEtJPTInCone[nh]; 
  TH1F* hEtZSP[nh];
  TH1F* hEtJPT[nh];
  TH1F* hEtJPTmuons[nh];
  TH1F* hEtJPTelectrons[nh];

  const char* namesEtJPT[nh] = {"hEtJPT1","hEtJPT2","hEtJPT3","hEtJPT4","hEtJPT5","hEtJPT6","hEtJPT7","hEtJPT8","hEtJPT9","hEtJPT10"};
  const char* titleEtJPT[nh] = {"EtJPT1","EtJPT2","EtJPT3","EtJPT4","EtJPT5","EtJPT6","EtJPT7","EtJPT8","EtJPT9","EtJPT10"};
  
  const char* namesEtElecJPT[nh] = {"hEtElecJPT1","hEtElecJPT2","hEtElecJPT3","hEtElecJPT4","hEtElecJPT5","hEtElecJPT6","hEtElecJPT7","hEtElecJPT8","hEtElecJPT9","hEtElecJPT10"};
  const char* titleEtElecJPT[nh] = {"EtElecJPT1","EtElecJPT2","EtElecJPT3","EtElecJPT4","EtElecJPT5","EtElecJPT6","EtElecJPT7","EtElecJPT8","EtElecJPT9","EtElecJPT10"};
  
  const char* namesEtMuJPT[nh] = {"hEtMuJPT1","hEtMuJPT2","hEtMuJPT3","hEtMuJPT4","hEtMuJPT5","hEtMuJPT6","hEtMuJPT7","hEtMuJPT8","hEtMuJPT9","hEtMuJPT10"};
  const char* titleEtMuJPT[nh] = {"EtMuJPT1","EtMuJPT2","EtMuJPT3","EtMuJPT4","EtMuJPT5","EtMuJPT6","EtMuJPT7","EtMuJPT8","EtMuJPT9","EtMuJPT10"};

  for(Int_t ih=0; ih < nh; ih++) { 
    hEtJPT[ih]  = new TH1F(namesEtJPT[ih], titleEtJPT[ih], 60, 0., 3.);
    hEtJPTmuons[ih]  = new TH1F(namesEtMuJPT[ih], titleEtMuJPT[ih], 60, 0., 3.);
    hEtJPTelectrons[ih]  = new TH1F(namesEtElecJPT[ih], titleEtElecJPT[ih], 60, 0., 3.);
  }

  TH1F * hEtGen  = new TH1F( "hEtGen", "EtGen", 20, 0., 200.);
  TH1F * hEtaGen = new TH1F( "hEtaGen", "EtaGen", 42, -2.4., 2.4);
  TH1F * hDR     = new TH1F( "hDR", "DR", 100, 0., 10.);
  TH1F * emul     = new TH1F( "hemul", "hemul", 10, 0., 10.);  
  TH1F * ept     = new TH1F( "hept", "hept", 100, 0., 100.);



  Long64_t nentries = fChain->GetEntriesFast();
 
  Long64_t nbytes = 0, nb = 0;  for (Long64_t jentry=0; jentry<nentries;jentry++) {
  
    nelecs=0;
    Reset();


    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // cout <<elecPt[0] << endl; 
   
    // For all histograms - Fill Electrons 
    if (elecPt[0]  >ptMin || nelecs==0) emul->Fill(nelecs); 
    for (int i=0;i<nelecs;i++){ if  
      (elecPt[i]  >ptMin ) ept->Fill(elecPt[i]);
    }

    for(Int_t ih = 0; ih < nh; ih++) {

      // if(abs(drElecFromZjet1)<0.8 ||abs(drMuonFromZjet1)<0.8 ||abs(drTauFromZjet1)<0.8  ) continue;
      //if(abs(drElecFromZjet2)<0.8 ||abs(drMuonFromZjet2)<0.8 ||abs(drTauFromZjet2)<0.8  ) continue;

      // in which pt, bin + dr Cut,eta region  calculated previously , matched with 0.4
      if(EtGen1 >= xbins[ih] && EtGen1 < xbins[ih+1] 
	 && DRMAXgjet1 <drmax 
	 && fabs(EtaGen1) > etaMin && fabs(EtaGen1) <= etaMax ) {	
	 
	// low e jet, fill only 1 jet
	if(EtRaw1/EtGen1 > 0.1) {
	  if(EtGen2 < 20.) {
	    jet1++;
	    //  cout <<" ElectronFlagGen1 " <<ElectronFlagGen1  << " MuonFlagGen1 " << MuonFlagGen1 <<endl;
	    if (ElectronFlagGen1NoLep==0 && MuonFlagGen1NoLep==0 )  hEtJPT[ih]->Fill(EtJPT1/EtGen1);
	    if (elecPt[0]>ptMin ) if (ElectronFlagGen1==1 && MuonFlagGen1==0 )  hEtJPTelectrons[ih]->Fill(EtJPT1/EtGen1);	  
	    if (ElectronFlagGen1==0 && MuonFlagGen1==1 ) hEtJPTmuons[ih]->Fill(EtJPT1/EtGen1);  
	    //  if (ElectronFlagGen1NoLep==1 )  hEtJPTelectrons[ih]->Fill(EtJPT1/EtGen1);
	    // if (ElectronFlagGen1NoLep==0 && MuonFlagGen1NoLep==1 )  hEtJPTmuons[ih]->Fill(EtJPT1/EtGen1);	 
	  } 

	  //2. jet
	  if(EtGen2 > 20.) {
	    // 2. jet hard, check if overlapping
	    Float_t DR = deltaR(EtaGen1, EtaGen2, PhiGen1, PhiGen2);
	    if(DR > DRcut) {	     
	      // hDR->Fill(DR);
	      jet1++;	  
	      if (ElectronFlagGen1NoLep==0 && MuonFlagGen1NoLep==0 )  hEtJPT[ih]->Fill(EtJPT1/EtGen1);
	      if (elecPt[0]>ptMin )     if (ElectronFlagGen1==1 && MuonFlagGen1==0 )  hEtJPTelectrons[ih]->Fill(EtJPT1/EtGen1);	   
	      if (ElectronFlagGen1==0 && MuonFlagGen1==1 ) hEtJPTmuons[ih]->Fill(EtJPT1/EtGen1); 
	      // if (ElectronFlagGen1NoLep==1)  hEtJPTelectrons[ih]->Fill(EtJPT1/EtGen1);
	      //     if (ElectronFlagGen1NoLep==0 && MuonFlagGen1NoLep==1 )  hEtJPTmuons[ih]->Fill(EtJPT1/EtGen1);
	      hEtGen->Fill(EtGen1);
	      hEtaGen->Fill(EtaGen1);
	    }
	  } //   if(EtGen2 > 20.)
	}//  if(EtRaw1/EtGen1 > 0.1) {
      }

      if(EtGen2 >= xbins[ih] && EtGen2 < xbins[ih+1] 
	 && DRMAXgjet2 < drmax
	 && fabs(EtaGen2) > etaMin && fabs(EtaGen2) <= etaMax) {	  

	if(EtRaw2/EtGen2 > 0.1) {
	  DR = deltaR(EtaGen1, EtaGen2, PhiGen1, PhiGen2);
	  if(DR > DRcut) {
	    hDR->Fill(DR);
	    jet2++;	   
	    if (ElectronFlagGen2NoLep==0 && MuonFlagGen2NoLep==0 )	    hEtJPT[ih]->Fill(EtJPT2/EtGen2);
	    if (elecPt[0]>ptMin )      if (ElectronFlagGen2==1 && MuonFlagGen2==0 )            hEtJPTelectrons[ih]->Fill(EtJPT2/EtGen2);
	    // if (ElectronFlagGen2NoLep==1 )            hEtJPTelectrons[ih]->Fill(EtJPT2/EtGen2);
	    //if (ElectronFlagGen2NoLep==0 && MuonFlagGen2NoLep==1 )            hEtJPTmuons[ih]->Fill(EtJPT2/EtGen2); 
	    if (ElectronFlagGen2==0 && MuonFlagGen2==1 ) hEtJPTmuons[ih]->Fill(EtJPT2/EtGen2);
	    hEtGen->Fill(EtGen2);
	    hEtaGen->Fill(EtaGen2);
	  }// 	  if(DR > DRcut) { 
	}
      } // eta max/pt min
    }  //end  For all histograms  for(Int_t ih = 0; ih < nh; ih++) 
  } // end  for (Long64_t jentry=0; jentry<nentries;jentry
 

// Fitting ----------------------------------- 
  setTDRStyle(0);
  gStyle->SetOptFit();  
 //  c2->cd(1);
//   emul->Draw(); 
//   c2->cd(2);
//   ept->Draw(); 
//   c2->SaveAs("ElectronMul.gif");
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

    c1->cd(1);
    // JPT
    // get bin with max content relative resolution
    Int_t binMax = hEtJPT[ih]->GetMaximumBin();
    TAxis* xaxis = hEtJPT[ih]->GetXaxis();
    Double_t binCenter = xaxis->GetBinCenter(binMax);
    Double_t rms = hEtJPT[ih]->GetRMS();
    Double_t rFitMin = binCenter - 2.0 * rms; 
    Double_t rFitMax = binCenter + 2.0 * rms;
    hEtJPT[ih]->Fit("gaus","","",rFitMin,rFitMax);
    TF1 *fit = hEtJPT[ih]->GetFunction("gaus"); 
    gStyle->SetOptFit();
    Double_t mean  = fit->GetParameter(1);
    Double_t meanErr  = fit->GetParError(1);
    Double_t sigma = fit->GetParameter(2);
    Double_t sigmaErr = fit->GetParError(2);
    Float_t resolution = sigma/mean;
    //Both errors are propageted
    Float_t resolutionErr = resolution * sqrt((meanErr/mean)*(meanErr/mean) + (sigmaErr/sigma)*(sigmaErr/sigma));
    hResJPT->Fill(EbinCenter,resolution);
    hResJPT->SetBinError(ih+1,resolutionErr);    
    hScaleJPT->Fill(EbinCenter,mean);
    hScaleJPT->SetBinError(ih+1,meanErr);
    
    c1->cd(2);
    // JPT with electrons
    // get bin with max content relative resolution
    Int_t binMax = hEtJPTelectrons[ih]->GetMaximumBin();
    TAxis* xaxis = hEtJPTelectrons[ih]->GetXaxis();
    Double_t binCenter = xaxis->GetBinCenter(binMax);
    Double_t rms = hEtJPTelectrons[ih]->GetRMS();
    Double_t rFitMin = binCenter - 2.0 * rms; 
    Double_t rFitMax = binCenter + 2.0 * rms;
    hEtJPTelectrons[ih]->Fit("gaus","","",rFitMin,rFitMax);
    TF1 *fit = hEtJPTelectrons[ih]->GetFunction("gaus"); 
    gStyle->SetOptFit();
    Double_t mean  = fit->GetParameter(1);
    Double_t meanErr  = fit->GetParError(1);
    Double_t sigma = fit->GetParameter(2);
    Double_t sigmaErr = fit->GetParError(2);
    Float_t resolution = sigma/mean;
    //Both errors are propageted
    Float_t resolutionErr = resolution * sqrt((meanErr/mean)*(meanErr/mean) + (sigmaErr/sigma)*(sigmaErr/sigma));
    hResJPTelectrons->Fill(EbinCenter,resolution);
    hResJPTelectrons->SetBinError(ih+1,resolutionErr);    
    hScaleJPTelectrons->Fill(EbinCenter,mean);
    hScaleJPTelectrons->SetBinError(ih+1,meanErr);    

    c1->cd(3);
    // JPT with muons
    // get bin with max content relative resolution
    Int_t binMax = hEtJPTmuons[ih]->GetMaximumBin();
    TAxis* xaxis = hEtJPTmuons[ih]->GetXaxis();
    Double_t binCenter = xaxis->GetBinCenter(binMax);
    Double_t rms = hEtJPTmuons[ih]->GetRMS();
    Double_t rFitMin = binCenter - 2.0 * rms; 
    Double_t rFitMax = binCenter + 2.0 * rms;
    hEtJPTmuons[ih]->Fit("gaus","","",rFitMin,rFitMax);
    TF1 *fit = hEtJPTmuons[ih]->GetFunction("gaus"); 
    gStyle->SetOptFit();
    Double_t mean  = fit->GetParameter(1);
    Double_t meanErr  = fit->GetParError(1);
    Double_t sigma = fit->GetParameter(2);
    Double_t sigmaErr = fit->GetParError(2);
    Float_t resolution = sigma/mean;
    //Both errors are propageted
    Float_t resolutionErr = resolution * sqrt((meanErr/mean)*(meanErr/mean) + (sigmaErr/sigma)*(sigmaErr/sigma));
    hResJPTmuons->Fill(EbinCenter,resolution);
    hResJPTmuons->SetBinError(ih+1,resolutionErr);    
    hScaleJPTmuons->Fill(EbinCenter,mean);
    hScaleJPTmuons->SetBinError(ih+1,meanErr);    

    sprintf(name,"hCalo1_%d.gif",ih);
    c1->SaveAs(name);

  } // end for each histogramm ... 

// Saving as root file  ----------------------------------- 
  // save histo on disk
  // Resolution .-----------------------
  TFile efile("test.root","recreate");
  hResRaw->Write();
  hResJPTInCone->Write();
  hResJPT->Write();
  hScaleRaw->Write();
  hScaleJPTInCone->Write();
  hScaleJPT->Write();
  hScaleJPTelectrons->Write(); 
  hScaleJPTmuons->Write(); 
  hResJPTelectrons->Write(); 
  hResJPTmuons->Write();
  emul->Write();
  ept->Write(); hEtaGen->Write();
  efile.Close();

  TCanvas* c40 = new TCanvas("X","Y",1);

  hResJPT->GetXaxis()->SetTitle("E_{T} Gen, GeV");
  hResJPT->GetYaxis()->SetTitle("Energy resolution, % ");
  hResJPT->SetMaximum(0.45);
  hResJPT->SetMinimum(0.05);
  hResJPT->SetMarkerStyle(21);
  hResJPT->SetMarkerSize(1.2);
  hResJPT->Draw("histPE1");
  //   hResJPTInCone->SetMarkerSize(1.0);
  //   hResJPTInCone->SetMarkerStyle(24);
  //   hResJPTInCone->Draw("samePE1");
  hResJPTmuons->SetMarkerSize(1.5);
  hResJPTmuons->SetMarkerStyle(24);
  hResJPTmuons->Draw("samePE1");

  hResJPTelectrons->SetMarkerSize(1.5);
  hResJPTelectrons->SetMarkerStyle(22);
  hResJPTelectrons->Draw("samePE1");

  TLatex *t = new TLatex();
  t->SetTextSize(0.042);
  TLegend *leg = new TLegend(0.45,0.65,0.85,0.8,NULL,"brNDC");
  leg->SetFillColor(10);
  leg->AddEntry(hResJPTelectrons,"Electrons JPT","P");
  leg->AddEntry(hResJPTmuons,"Muons JPT","P");
  leg->AddEntry(hResJPT,"Hadronic JPT","P");
  // t->DrawLatex(25,0.42,"CMSSW219");
  leg->Draw(); 
  t->DrawLatex(25,0.40,"MadGraph VQQ (Zbb), |#eta ^{jet}|< 1.0");
  c40->SaveAs("resComponentsJPT219.gif");

  // Scale ------------------------

  TCanvas* c20 = new TCanvas("X","Y",1);
  hScaleJPT->GetXaxis()->SetTitle("E_{T} Gen, GeV");
  hScaleJPT->GetYaxis()->SetTitle("E_{T}^{reco}/E_{T}^{gen}");
  hScaleJPT->SetMaximum(1.5);
  hScaleJPT->SetMinimum(0.8);
  hScaleJPT->SetMarkerStyle(21);
  hScaleJPT->SetMarkerSize(1.2);
  hScaleJPT->Draw("histPE1");

  hScaleJPTelectrons->SetMarkerSize(1.6);
  hScaleJPTelectrons->SetMarkerStyle(22);
  hScaleJPTelectrons->Draw("samePE1");
  
  hScaleJPTmuons->SetMarkerSize(1.5);
  hScaleJPTmuons->SetMarkerStyle(24);
  hScaleJPTmuons->Draw("samePE1");
  
  TLatex *t = new TLatex();
  t->SetTextSize(0.042);
  //  TLegend *leg = new TLegend(0.5,0.15,0.9,0.35,NULL,"brNDC");
  TLegend *leg = new TLegend(0.45,0.65,0.85,0.8,NULL,"brNDC");
  leg->SetFillColor(10);
  leg->AddEntry(hScaleJPTelectrons,"Electrons JPT","P");
  leg->AddEntry(hScaleJPTmuons,"Muons JPT","P");
  leg->AddEntry(hScaleJPT,"Hadronic JPT","P");
  leg->Draw();  
  //  t->DrawLatex(25,1.12,"CMSSW219");
  //  t->DrawLatex(25,1.06,"RelVal QCD 80-120 GeV, |#eta ^{jet}|< 1.0");
  t->DrawLatex(25,0.40,"MadGraph VQQ (Zbb), |#eta ^{jet}|< 1.0");
  c20->SaveAs("ScaleComponentsJPT219.gif");
  cout<<jet1 <<" "  <<jet2<< endl;
  
}









// ----------------Cosmetics & Helper



Float_t JPTBjetRootAnalysis::deltaPhi(Float_t phi1, Float_t phi2)
{
  Float_t pi = 3.1415927;
  Float_t dphi = fabs(phi1 - phi2);
  if(dphi >= pi) dphi = 2. * pi - dphi; 
  return dphi;
}

Float_t JPTBjetRootAnalysis::deltaEta(Float_t eta1, Float_t eta2)
{
  Float_t deta = fabs(eta1-eta2);
  return deta;
}

Float_t JPTBjetRootAnalysis::deltaR(Float_t eta1, Float_t eta2,
				    Float_t phi1, Float_t phi2)
{
  Float_t dr = sqrt( deltaEta(eta1, eta2) * deltaEta(eta1, eta2) +
		     deltaPhi(phi1, phi2) * deltaPhi(phi1, phi2) );
  return dr;
}

void JPTBjetRootAnalysis::setTDRStyle(Int_t ylog) {

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

  tdrStyle->SetHistLineColor(1);
  tdrStyle->SetHistLineStyle(0);
  tdrStyle->SetHistLineWidth(2);
  tdrStyle->SetEndErrorSize(4);
  tdrStyle->SetMarkerStyle(20);

  //For the fit/function:
  tdrStyle->SetOptFit(1);
  tdrStyle->SetFitFormat("5.4g");
  tdrStyle->SetFuncColor(1);
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
  //   tdrStyle->SetTitleXSize(Float_t size = 0.02); // Another way to set the size?
  //   tdrStyle->SetTitleYSize(Float_t size = 0.02);
  tdrStyle->SetTitleXOffset(0.9);
  tdrStyle->SetTitleYOffset(1.05);
  // tdrStyle->SetTitleOffset(1.1, "Y"); // Another way to set the Offset

  // For the axis labels:
  tdrStyle->SetLabelColor(1, "XYZ");
  tdrStyle->SetLabelFont(36, "XYZ");
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
  tdrStyle->SetOptLogy(ylog);
  tdrStyle->SetOptLogz(0);

  // Postscript options:

  //  tdrStyle->SetPaperSize(7.5,7.5);

  tdrStyle->SetPaperSize(15.,15.);

  tdrStyle->cd();
}
