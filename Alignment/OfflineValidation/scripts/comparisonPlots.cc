#include "comparisonPlots.h"
#include <string>
#include <sstream>

#include "TProfile.h"
#include "TList.h"
#include "TNtuple.h"
#include "TString.h"
#include <iostream>
#include "TStyle.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "THStack.h"


comparisonPlots::comparisonPlots(std::string filename, std::string outputDir, std::string outputFilename)
{
	
	_outputDir = outputDir;
	
	fin = new TFile(filename.c_str());
	fin->cd();
	
	output = new TFile((outputDir+outputFilename).c_str(),"recreate");
	output->cd();
	
	readTree();
	
}

void comparisonPlots::readTree(){
	
	data = (TTree*)fin->Get("alignTree");
	data->SetBranchAddress("id",&id_);
	data->SetBranchAddress("mid",&mid_);
	data->SetBranchAddress("level",&level_);
	data->SetBranchAddress("mlevel",&mlevel_);
	data->SetBranchAddress("sublevel",&sublevel_);
	data->SetBranchAddress("x",&x_);
	data->SetBranchAddress("y",&y_);
	data->SetBranchAddress("z",&z_);
	data->SetBranchAddress("alpha",&alpha_);
	data->SetBranchAddress("beta",&beta_);
	data->SetBranchAddress("gamma",&gamma_);
	data->SetBranchAddress("phi",&phi_);
	data->SetBranchAddress("eta",&eta_);
	data->SetBranchAddress("r",&r_);
	data->SetBranchAddress("dx",&dx_);
	data->SetBranchAddress("dy",&dy_);
	data->SetBranchAddress("dz",&dz_);
	data->SetBranchAddress("dphi",&dphi_);
	data->SetBranchAddress("dr",&dr_);	
	data->SetBranchAddress("dalpha",&dalpha_);
	data->SetBranchAddress("dbeta",&dbeta_);
	data->SetBranchAddress("dgamma",&dgamma_);
	data->SetBranchAddress("useDetId",&useDetId_);
	data->SetBranchAddress("detDim",&detDim_);
}

void comparisonPlots::plotTwist(TCut Cut, char* dirName, bool savePlot, std::string plotName, bool autolimits,int ColorCode)
{
  gStyle->SetTitleAlign(22);
  gStyle->SetTitleX(0.5);
  gStyle->SetTitleY(0.97);
  gStyle->SetTitleFont(62); 
  
  TDirectory* plotDir = output->mkdir( dirName );
  
  double minimum, maximum;
  
  if (autolimits)
    {
      TH1F* phdphi = new TH1F("phdphi", "phdphi", 200, -10, 10);
      data->Project("phdphi","dphi",Cut);
      getHistMaxMin(phdphi, maximum, minimum, 1);
      delete phdphi;
    }
  else
    {
      minimum = -0.0004, maximum = 0.0004;
    }


  TH2F* h_dphiVz = new TH2F("h_dphiVz","#Delta #phi vs. z", 200,minimum,maximum, 200,minimum,maximum);
  h_dphiVz->SetMarkerStyle(6);

  int j = 2;
  TCut zCut[6];
  zCut[0] = "z < 0";
  zCut[1] = "z >= 0";
  
  
  if (ColorCode == 1) {
    j = 6;
    //    zCut[6];
    zCut[0] = "sublevel==1";
    zCut[1] = "sublevel==2";
    zCut[2] = "sublevel==3";
    zCut[3] = "sublevel==4";
    zCut[4] = "sublevel==5";
    zCut[5] = "sublevel==6";
  } 
  
  std::stringstream st;
  st << "dphi > " << minimum << " && dphi < " << maximum;
  
  TCut minmaxCut(st.str().c_str());

  data->Project("h_dphiVz", "dphi:z",Cut+minmaxCut);
  TGraph* gr_dphiVz_Array[j];
  TMultiGraph* mgr_dphiVz=new TMultiGraph();
  for ( int i = 0; i < j; i++) {
    data->Draw("dphi:z",Cut+zCut[i]+minmaxCut);	  
    gr_dphiVz_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
    gr_dphiVz_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
    gr_dphiVz_Array[i]->SetMarkerStyle(6);
    gr_dphiVz_Array[i]->SetMaximum(maximum);
    gr_dphiVz_Array[i]->SetMinimum(minimum);
    mgr_dphiVz->Add(gr_dphiVz_Array[i],"p");
  }


  
  // ---------  draw histograms ---------
  TCanvas* c = new TCanvas("c_twist", "c_twist", 200, 10, 600, 600);
  c->SetFillColor(0);
  data->SetMarkerSize(0.5);
  data->SetMarkerStyle(6);
  c->Divide(1,1);

  mgr_dphiVz->SetTitle("#Delta #phi vs. z");
  // if (!autolimits)
  //   h_dphiVz->Draw();
  // else
  mgr_dphiVz->SetMaximum(maximum);
  mgr_dphiVz->SetMinimum(minimum);
  mgr_dphiVz->Draw("a");
  	
  c->Update();
  plotDir->cd();

  //h_dphiVz->Write();
  if (savePlot) c->Print((_outputDir+"plotTwist_"+plotName).c_str());
	
  delete c;
  delete h_dphiVz;
}

void comparisonPlots::plot3x3Rot(TCut Cut, char* dirName, bool savePlot, std::string plotName, bool autolimits, int ColorCode ){
	
  gStyle->SetTitleFont(62);
  gStyle->SetOptTitle(0);
   
  gStyle->SetFrameBorderMode(0);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetPadColor(0);
  gStyle->SetCanvasColor(0);
  gStyle->SetTitleColor(1);
  gStyle->SetStatColor(0);
  gStyle->SetStatBorderSize(1);
  gStyle->SetFrameFillColor(0);

  gStyle->SetTextFont(132);
  gStyle->SetTextSize(0.08);
  gStyle->SetLabelFont(132,"x");
  gStyle->SetLabelFont(132,"y");
  gStyle->SetLabelFont(132,"z");
  gStyle->SetTitleSize(0.08,"x");
  gStyle->SetTitleSize(0.08,"y");
  gStyle->SetTitleSize(0.08,"z");
  gStyle->SetLabelSize(0.08,"x");
  gStyle->SetLabelSize(0.08,"y");
  gStyle->SetLabelSize(0.08,"z");

  gStyle->SetMarkerStyle(8);
  gStyle->SetHistLineWidth(1.85);
  gStyle->SetLineStyleString(2,"[12 12]"); // postscript dashes

  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  
	TDirectory* plotDir = output->mkdir( dirName );
	
	double minimumR, maximumR;
	double minimumZ, maximumZ;
	double minimumPhi, maximumPhi;
	double minimumDalpha, maximumDalpha;
	double minimumDbeta, maximumDbeta;
	double minimumDgamma, maximumDgamma;

	if (autolimits){
		// ---------  get right limits for histogram ---------
		TH1F* phr = new TH1F("phr", "phr", 200, 0, 200);
		TH1F* phz = new TH1F("phz", "phz", 400, -300, 300);
		TH1F* phphi = new TH1F("phphi", "phphi", 200, -3.15, 3.15);
		TH1F* phdalpha = new TH1F("phdalpha", "phdalpha", 2000, -10, 10);
		TH1F* phdbeta = new TH1F("phdbeta", "phdbeta", 2000, -10, 10);
		TH1F* phdgamma = new TH1F("phdgamma", "phdgamma", 200, -10, 10);
		data->Project("phr","r",Cut);
		data->Project("phz","z",Cut);
		data->Project("phphi","phi",Cut);
		data->Project("phdalpha","dalpha",Cut);
		data->Project("phdbeta","dbeta",Cut);
		data->Project("phdgamma","dgamma",Cut);
		getHistMaxMin(phr, maximumR, minimumR, 0);
		getHistMaxMin(phz, maximumZ, minimumZ, 0);
		getHistMaxMin(phphi, maximumPhi, minimumPhi, 0);
		getHistMaxMin(phdalpha, maximumDalpha, minimumDalpha, 1);
		getHistMaxMin(phdbeta, maximumDbeta, minimumDbeta, 1);
		getHistMaxMin(phdgamma, maximumDgamma, minimumDgamma, 1);
	}
	else{
		minimumR = 0., maximumR = 200.; 
		minimumZ = -300., maximumZ = 300.; 
		minimumPhi = -3.15, maximumPhi = 3.15;
		minimumDalpha = -1, maximumDalpha = 1;
		minimumDbeta = -1, maximumDbeta = 1;
		minimumDgamma = -1, maximumDgamma = 1;
	}
	
	// ---------  declare histograms ---------
	TH2F* h_dalphaVr   = new TH2F("h_dalphaVr","#Delta #alpha vs. r",200,minimumR,maximumR,200,minimumDalpha,maximumDalpha);
	TH2F* h_dbetaVr    = new TH2F("h_dbetaVr", "#Delta #beta vs. r", 200,minimumR,maximumR,200,minimumDbeta, maximumDbeta);
	TH2F* h_dgammaVr   = new TH2F("h_dgammaVr","#Delta #gamma vs. r",200,minimumR,maximumR,200,minimumDgamma,maximumDgamma);

	TH2F* h_dalphaVz   = new TH2F("h_dalphaVz","#Delta #alpha vs. z", 200,minimumZ,maximumZ, 200,minimumDalpha,maximumDalpha);
	TH2F* h_dbetaVz    = new TH2F("h_dbetaVz", "#Delta #beta vs. z",  200,minimumZ,maximumZ, 200,minimumDbeta, maximumDbeta);
	TH2F* h_dgammaVz   = new TH2F("h_dgammaVz","#Delta #gama vs. z",  200,minimumZ,maximumZ, 200,minimumDgamma,maximumDgamma);

	TH2F* h_dalphaVphi = new TH2F("h_dalphaVphi","#Delta #alpha vs. #phi", 200,minimumPhi,maximumPhi,200,minimumDalpha,maximumDalpha);
	TH2F* h_dbetaVphi  = new TH2F("h_dbetaVphi", "#Delta #beta vs. #phi",  200,minimumPhi,maximumPhi,200,minimumDbeta, maximumDbeta);
	TH2F* h_dgammaVphi = new TH2F("h_dgammaVphi","#Delta #gamma vs. #phi", 200,minimumPhi,maximumPhi,200,minimumDgamma,maximumDgamma);

	h_dalphaVr   ->SetMarkerStyle(6) ; 
	h_dbetaVr    ->SetMarkerStyle(6) ; 
	h_dgammaVr   ->SetMarkerStyle(6) ; 

	h_dalphaVz   ->SetMarkerStyle(6) ; 
	h_dbetaVz    ->SetMarkerStyle(6) ; 
	h_dgammaVz   ->SetMarkerStyle(6) ; 

	h_dalphaVphi ->SetMarkerStyle(6) ; 
	h_dbetaVphi  ->SetMarkerStyle(6) ; 
	h_dgammaVphi ->SetMarkerStyle(6) ; 

	int j = 2;
	TCut zCut[6];
	zCut[0] = "z < 0";
	zCut[1] = "z >= 0";

	if (ColorCode == 1) {
	  j = 6;
          //	  zCut[6];
	  zCut[0] = "sublevel==1";
	  zCut[1] = "sublevel==2";
	  zCut[2] = "sublevel==3";
	  zCut[3] = "sublevel==4";
	  zCut[4] = "sublevel==5";
	  zCut[5] = "sublevel==6";
	} 
	 
	data->Project("h_dalphaVr", "dalpha*1000:r",Cut);
	TGraph* gr_dalphaVr_Array[j];
	TMultiGraph* mgr_dalphaVr=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dalpha*1000:r",Cut+zCut[i]);	  
		gr_dalphaVr_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_dalphaVr_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dalphaVr_Array[i]->SetMarkerStyle(6);	  
		mgr_dalphaVr->Add(gr_dalphaVr_Array[i],"p");
	}	
	
	data->Project("h_dbetaVr", "dbeta*1000:r",Cut);
	TGraph* gr_dbetaVr_Array[j];
	TMultiGraph* mgr_dbetaVr=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dbeta*1000:r",Cut+zCut[i]);	  
		gr_dbetaVr_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_dbetaVr_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dbetaVr_Array[i]->SetMarkerStyle(6);	  
		mgr_dbetaVr->Add(gr_dbetaVr_Array[i],"p");
	}	
	
	data->Project("h_dgammaVr", "dgamma*1000:r",Cut);
	TGraph* gr_dgammaVr_Array[j];
	TMultiGraph* mgr_dgammaVr=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dgamma*1000:r",Cut+zCut[i]);	  
		gr_dgammaVr_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_dgammaVr_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dgammaVr_Array[i]->SetMarkerStyle(6);	  
		mgr_dgammaVr->Add(gr_dgammaVr_Array[i],"p");
	}	
	
	data->Project("h_dalphaVz", "dalpha*1000:z",Cut);
	TGraph* gr_dalphaVz_Array[j];
	TMultiGraph* mgr_dalphaVz=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dalpha*1000:z",Cut+zCut[i]);	  
		gr_dalphaVz_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_dalphaVz_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dalphaVz_Array[i]->SetMarkerStyle(6);	  
		mgr_dalphaVz->Add(gr_dalphaVz_Array[i],"p");
	}	
	
	data->Project("h_dbetaVz", "dbeta*1000:z",Cut);
	TGraph* gr_dbetaVz_Array[j];
	TMultiGraph* mgr_dbetaVz=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dbeta*1000:z",Cut+zCut[i]);	  
		gr_dbetaVz_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_dbetaVz_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dbetaVz_Array[i]->SetMarkerStyle(6);	  
		mgr_dbetaVz->Add(gr_dbetaVz_Array[i],"p");
	}	
	
	data->Project("h_dgammaVz", "dgamma*1000:z",Cut);
	TGraph* gr_dgammaVz_Array[j];
	TMultiGraph* mgr_dgammaVz=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dgamma*1000:z",Cut+zCut[i]);	  
		gr_dgammaVz_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_dgammaVz_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dgammaVz_Array[i]->SetMarkerStyle(6);	  
		mgr_dgammaVz->Add(gr_dgammaVz_Array[i],"p");
	}	
	
	data->Project("h_dalphaVphi", "dalpha*1000:phi",Cut);
	TGraph* gr_dalphaVphi_Array[j];
	TMultiGraph* mgr_dalphaVphi=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dalpha*1000:phi",Cut+zCut[i]);	  
		gr_dalphaVphi_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_dalphaVphi_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dalphaVphi_Array[i]->SetMarkerStyle(6);	  
		mgr_dalphaVphi->Add(gr_dalphaVphi_Array[i],"p");
	}	
	
	data->Project("h_dbetaVphi", "dbeta*1000:phi",Cut);
	TGraph* gr_dbetaVphi_Array[j];
	TMultiGraph* mgr_dbetaVphi=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dbeta*1000:phi",Cut+zCut[i]);	  
		gr_dbetaVphi_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_dbetaVphi_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dbetaVphi_Array[i]->SetMarkerStyle(6);	  
		mgr_dbetaVphi->Add(gr_dbetaVphi_Array[i],"p");
	}	
	
	data->Project("h_dgammaVphi", "dgamma*1000:phi",Cut);
	TGraph* gr_dgammaVphi_Array[j];
	TMultiGraph* mgr_dgammaVphi=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dgamma*1000:phi",Cut+zCut[i]);	  
		gr_dgammaVphi_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_dgammaVphi_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dgammaVphi_Array[i]->SetMarkerStyle(6);	  
		mgr_dgammaVphi->Add(gr_dgammaVphi_Array[i],"p");
	}	
	
	// ---------  draw histograms ---------
	TCanvas* c = new TCanvas("c", "c", 200, 10, 1200, 700);
	c->SetFillColor(0);
	data->SetMarkerSize(0.5);
	data->SetMarkerStyle(6);
	c->Divide(3,3);
	c->cd(1);
	if (!autolimits) h_dalphaVr->Draw();
	if (autolimits) mgr_dalphaVr->Draw("a");
	c->Update();
	c->cd(2);
	if (!autolimits) h_dbetaVr->Draw();
	if (autolimits) mgr_dbetaVr->Draw("a");
	c->Update();
	c->cd(3);
	if (!autolimits) h_dgammaVr->Draw();
	if (autolimits) mgr_dgammaVr->Draw("a");
	c->cd(4);
	if (!autolimits) h_dalphaVz->Draw();
	if (autolimits) mgr_dalphaVz->Draw("a");
	c->cd(5);
	if (!autolimits) h_dbetaVz->Draw();
	if (autolimits) mgr_dbetaVz->Draw("a");
	c->cd(6);
	if (!autolimits) h_dgammaVz->Draw();
	if (autolimits) mgr_dgammaVz->Draw("a");
	c->cd(7);
	if (!autolimits) h_dalphaVphi->Draw();
	if (autolimits) mgr_dalphaVphi->Draw("a");
	c->cd(8);
	if (!autolimits) h_dbetaVphi->Draw();
	if (autolimits) mgr_dbetaVphi->Draw("a");
	c->cd(9);
	if (!autolimits) h_dgammaVphi->Draw();
	if (autolimits) mgr_dgammaVphi->Draw("a");
	
	mgr_dalphaVr->GetXaxis()  ->SetTitle("r (cm)");
	mgr_dbetaVr->GetXaxis()   ->SetTitle("r (cm)");           
	mgr_dgammaVr->GetXaxis()  ->SetTitle("r (cm)");    
	mgr_dalphaVz->GetXaxis()  ->SetTitle("z (cm)");           
	mgr_dbetaVz->GetXaxis()   ->SetTitle("z (cm)");           
	mgr_dgammaVz->GetXaxis()  ->SetTitle("z (cm)");    
	mgr_dalphaVphi->GetXaxis()->SetTitle("#phi");      
	mgr_dbetaVphi->GetXaxis() ->SetTitle("#phi");      
	mgr_dgammaVphi->GetXaxis()->SetTitle("#phi");
	
	mgr_dalphaVr->GetYaxis()  ->SetTitle("#Delta #alpha (mrad)");
	mgr_dbetaVr->GetYaxis()   ->SetTitle("#Delta #beta (mrad)");           
	mgr_dgammaVr->GetYaxis()  ->SetTitle("#Delta #gamma (mrad)");    
	mgr_dalphaVz->GetYaxis()  ->SetTitle("#Delta #alpha (mrad)");           
	mgr_dbetaVz->GetYaxis()   ->SetTitle("#Delta #beta (mrad)");           
	mgr_dgammaVz->GetYaxis()  ->SetTitle("#Delta #gamma (mrad)");    
	mgr_dalphaVphi->GetYaxis()->SetTitle("#Delta #alpha (mrad)");      
	mgr_dbetaVphi->GetYaxis() ->SetTitle("#Delta #beta (mrad)");      
	mgr_dgammaVphi->GetYaxis()->SetTitle("#Delta #gamma (mrad)");
	
	c->Update();
	
	// // ---------  set output directory for histograms ---------
	// plotDir->cd();
	// h_dalphaVr->Write(); h_dbetaVr->Write(); h_dgammaVr->Write(); 
	// h_dalphaVz->Write(); h_dbetaVz->Write(); h_dgammaVz->Write(); 
	// h_dalphaVphi->Write(); h_dbetaVphi->Write(); h_dgammaVphi->Write(); 
	 
	if (savePlot) c->Print((_outputDir+"plot3x3Rot_"+plotName).c_str());

}

void comparisonPlots::plot3x5(TCut Cut, char* dirName, bool savePlot, std::string plotName, bool autolimits, int ColorCode ){
	
  //int ColorCodeHalfBarrels = 1; 
  //color seperation of moodules corresponding to the different subdets
  //if you want to seperate corresponding to the halfbarrels (+z or -z) set colorcode !=1
  
  gStyle->SetTitleFont(62);
  gStyle->SetOptTitle(0);
   
  gStyle->SetFrameBorderMode(0);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetPadColor(0);
  gStyle->SetCanvasColor(0);
  gStyle->SetTitleColor(1);
  gStyle->SetStatColor(0);
  gStyle->SetStatBorderSize(1);
  gStyle->SetFrameFillColor(0);

  gStyle->SetTextFont(132);
  gStyle->SetTextSize(0.08);
  gStyle->SetLabelFont(132,"x");
  gStyle->SetLabelFont(132,"y");
  gStyle->SetLabelFont(132,"z");
  gStyle->SetTitleSize(0.08,"x");
  gStyle->SetTitleSize(0.08,"y");
  gStyle->SetTitleSize(0.08,"z");
  gStyle->SetLabelSize(0.08,"x");
  gStyle->SetLabelSize(0.08,"y");
  gStyle->SetLabelSize(0.08,"z");

  gStyle->SetMarkerStyle(8);
  gStyle->SetHistLineWidth(1.85);
  gStyle->SetLineStyleString(2,"[12 12]"); // postscript dashes

  gStyle->SetTitleAlign(22);
  gStyle->SetTitleX(0.5);
  gStyle->SetTitleY(0.97);
  gStyle->SetTitleFont(62);

  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);

  gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadRightMargin(0.05);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.18);
  
  // ---------  create directory for histograms ---------
  //const char* dirName = Cut;
	TDirectory* plotDir = output->mkdir( dirName );
	
	// ---------  get right limits for histogram ---------
	double minimumR, maximumR;
	double minimumZ, maximumZ;
	double minimumPhi, maximumPhi;
	double minimumDR, maximumDR;
	double minimumDZ, maximumDZ;
	double minimumRDPhi, maximumRDPhi;
	double minimumDX, maximumDX;
	double minimumDY, maximumDY;
	if (autolimits){
		// ---------  get right limits for histogram ---------
		TH1F* phr     = new TH1F("phr",     "phr",     200,  0,     200 );
		TH1F* phz     = new TH1F("phz",     "phz",     400,  -300,  300 );
		TH1F* phphi   = new TH1F("phphi",   "phphi",   200,  -3.15, 3.15);
		TH1F* phdr    = new TH1F("phdr",    "phdr",    20000,-1000,  1000 );
		TH1F* phdz    = new TH1F("phdz",    "phdz",    20000,-1000,  1000 );
		TH1F* phrdphi = new TH1F("phrdphi", "phrdphi", 20000,-1000,  1000 );
		TH1F* phdx    = new TH1F("phdx",    "phy",     20000,-1000,  1000 );
		TH1F* phdy    = new TH1F("phdy",    "phy",     20000,-1000,  1000 );
		data->Project("phr","r",Cut);
		data->Project("phz","z",Cut);
		data->Project("phphi","phi",Cut);
		data->Project("phdr","dr*10000",Cut);
		data->Project("phdz","dz*10000",Cut);
		data->Project("phrdphi","r*dphi*10000",Cut);
		data->Project("phdx","dx*10000",Cut);
		data->Project("phdy","dy*10000",Cut);
		getHistMaxMin(phr, maximumR, minimumR, 0);
		getHistMaxMin(phz, maximumZ, minimumZ, 0);
		getHistMaxMin(phphi, maximumPhi, minimumPhi, 0);
		getHistMaxMin(phdr, maximumDR, minimumDR, 1);
		getHistMaxMin(phdz, maximumDZ, minimumDZ, 1);
		getHistMaxMin(phrdphi, maximumRDPhi, minimumRDPhi, 1);
		getHistMaxMin(phdx, maximumDX, minimumDX, 1);
		getHistMaxMin(phdy, maximumDY, minimumDY, 1);
	}
	else{
		minimumR = 0., maximumR = 200.; 
		minimumZ = -300., maximumZ = 300.; 
		minimumPhi = -3.15, maximumPhi = 3.15;
		minimumDR = -1, maximumDR = 1;
		minimumDZ = -1, maximumDZ = 1;
		minimumRDPhi = -1, maximumRDPhi = 1;
		minimumDX = -1, maximumDX = 1;
		minimumDY = -1, maximumDY = 1;
	}
	
	
	// ---------  declare histograms ---------
	TH1F* h_dr = new TH1F("h_dr", "#Delta r", 2000, minimumDR, maximumDR);
	TH1F* h_dz = new TH1F("h_dz", "#Delta z", 2000, minimumDZ, maximumDZ);
	TH1F* h_rdphi = new TH1F("h_rdphi", "r* #Delta #phi", 2000, minimumRDPhi, maximumRDPhi);
	TH2F* h_drVr = new TH2F("h_drVr","#Delta r vs. r",200,minimumR,maximumR,200,minimumDR,maximumDR);
	TH2F* h_dzVr = new TH2F("h_dzVr","#Delta z vs. r",200,minimumR,maximumR,200,minimumDZ,maximumDZ);
	TH2F* h_rdphiVr = new TH2F("h_rdphiVr","r#Delta #phi vs. r",200,minimumR,maximumR,200,minimumRDPhi,maximumRDPhi);
	TH2F* h_dxVr = new TH2F("h_dxVr","#Delta x vs. r", 200,minimumR,maximumR, 200,minimumDX,maximumDX);
	TH2F* h_dyVr = new TH2F("h_dyVr","#Delta y vs. r", 200,minimumR,maximumR, 200,minimumDY,maximumDY);
	TH2F* h_drVz = new TH2F("h_drVz","#Delta r vs. z", 200,minimumZ,maximumZ, 200,minimumDR,maximumDR);
	TH2F* h_dzVz = new TH2F("h_dzVz","#Delta z vs. z", 200,minimumZ,maximumZ, 200,minimumDZ,maximumDZ);
	TH2F* h_rdphiVz = new TH2F("h_rdphiVz","r#Delta #phi vs. z", 200,minimumZ,maximumZ, 200,minimumRDPhi,maximumRDPhi);
	TH2F* h_dxVz = new TH2F("h_dxVz","#Delta x vs. z", 200,minimumZ,maximumZ, 200,minimumDX,maximumDX);
	TH2F* h_dyVz = new TH2F("h_dyVz","#Delta y vs. z", 200,minimumZ,maximumZ, 200,minimumDY,maximumDY);
	TH2F* h_drVphi = new TH2F("h_drVphi","#Delta r vs. #phi", 200,minimumPhi,maximumPhi,200,minimumDR,maximumDR);
	TH2F* h_dzVphi = new TH2F("h_dzVphi","#Delta z vs. #phi", 200,minimumPhi,maximumPhi, 200,minimumDZ,maximumDZ);
	TH2F* h_rdphiVphi = new TH2F("h_rdphiVphi","r#Delta #phi vs. #phi", 200,minimumPhi,maximumPhi,200,minimumRDPhi,maximumRDPhi);
	TH2F* h_dxVphi = new TH2F("h_dxVphi","#Delta x vs. #phi", 200,minimumPhi,maximumPhi, 200,minimumDX,maximumDX);
	TH2F* h_dyVphi = new TH2F("h_dyVphi","#Delta y vs. #phi", 200,minimumPhi,maximumPhi, 200,minimumDY,maximumDY);
	
	h_drVr->SetMarkerStyle(6);
	h_dzVr->SetMarkerStyle(6);
	h_rdphiVr->SetMarkerStyle(6);
	h_dxVr->SetMarkerStyle(6);
	h_dyVr->SetMarkerStyle(6);
	h_drVz->SetMarkerStyle(6);
	h_dzVz->SetMarkerStyle(6);
	h_rdphiVz->SetMarkerStyle(6);
	h_dxVz->SetMarkerStyle(6);
	h_dyVz->SetMarkerStyle(6);
	h_drVphi->SetMarkerStyle(6);
	h_dzVphi->SetMarkerStyle(6);
	h_rdphiVphi->SetMarkerStyle(6);
	h_dxVphi->SetMarkerStyle(6);
	h_dyVphi->SetMarkerStyle(6);
	
	
	int j = 2;
	TCut zCut[6];
	zCut[0] = "z < 0";
	zCut[1] = "z >= 0";


	if (ColorCode == 1) {
	  j = 6;
          //	  zCut[6];
	  zCut[0] = "sublevel==1";
	  zCut[1] = "sublevel==2";
	  zCut[2] = "sublevel==3";
	  zCut[3] = "sublevel==4";
	  zCut[4] = "sublevel==5";
	  zCut[5] = "sublevel==6";
	} 
	 
	// ---------  project tree onto histograms ---------
	data->Project("h_dr","dr*10000",Cut);
	data->Project("h_dz","dz*10000",Cut);
	data->Project("h_rdphi","r*dphi*10000",Cut);
	

	data->Project("h_drVr", "dr*10000:r",Cut);
	TGraph* gr_drVr_Array[j];
	TMultiGraph* mgr_drVr=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dr*10000:r",Cut+zCut[i]);	  
		gr_drVr_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_drVr_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_drVr_Array[i]->SetMarkerStyle(6);	  
		mgr_drVr->Add(gr_drVr_Array[i],"p");
	}	
	
	data->Project("h_dzVr", "dz*10000:r",Cut);
	TGraph* gr_dzVr_Array[j];
	TMultiGraph* mgr_dzVr=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dz*10000:r",Cut+zCut[i]);	  
		gr_dzVr_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_dzVr_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dzVr_Array[i]->SetMarkerStyle(6);	  
		mgr_dzVr->Add(gr_dzVr_Array[i],"p");
	}
	
	data->Project("h_rdphiVr", "r*dphi*10000:r",Cut);
	TGraph* gr_rdphiVr_Array[j];
	TMultiGraph* mgr_rdphiVr=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("r*dphi*10000:r",Cut+zCut[i]);	  
		gr_rdphiVr_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_rdphiVr_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_rdphiVr_Array[i]->SetMarkerStyle(6);	  
		mgr_rdphiVr->Add(gr_rdphiVr_Array[i],"p");
	}
	
	data->Project("h_dxVr", "dx*10000:r",Cut);
	TGraph* gr_dxVr_Array[j];
	TMultiGraph* mgr_dxVr=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dx*10000:r",Cut+zCut[i]);	  
		gr_dxVr_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_dxVr_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dxVr_Array[i]->SetMarkerStyle(6);	  
		mgr_dxVr->Add(gr_dxVr_Array[i],"p");
	}
	
	data->Project("h_dyVr", "dy*10000:r",Cut);
	TGraph* gr_dyVr_Array[j];
	TMultiGraph* mgr_dyVr=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dy*10000:r",Cut+zCut[i]);	  
		gr_dyVr_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_dyVr_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dyVr_Array[i]->SetMarkerStyle(6);	  
		mgr_dyVr->Add(gr_dyVr_Array[i],"p");
	}
	
	data->Project("h_drVz", "dr*10000:z",Cut);
	TGraph* gr_drVz_Array[j];
	TMultiGraph* mgr_drVz=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dr*10000:z",Cut+zCut[i]);	  
		gr_drVz_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_drVz_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_drVz_Array[i]->SetMarkerStyle(6);	  
		mgr_drVz->Add(gr_drVz_Array[i],"p");
	}
	
	data->Project("h_dzVz", "dz*10000:z",Cut);
	TGraph* gr_dzVz_Array[j];
	TMultiGraph* mgr_dzVz=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dz*10000:z",Cut+zCut[i]);	  
		gr_dzVz_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_dzVz_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dzVz_Array[i]->SetMarkerStyle(6);	  
		mgr_dzVz->Add(gr_dzVz_Array[i],"p");
	}
	
	data->Project("h_rdphiVz", "r*dphi*10000:z",Cut);
	TGraph* gr_rdphiVz_Array[j];
	TMultiGraph* mgr_rdphiVz=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("r*dphi*10000:z",Cut+zCut[i]);	  
		gr_rdphiVz_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_rdphiVz_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_rdphiVz_Array[i]->SetMarkerStyle(6);	  
		mgr_rdphiVz->Add(gr_rdphiVz_Array[i],"p");
	}
	
	data->Project("h_dxVz", "dx*10000:z",Cut);
	TGraph* gr_dxVz_Array[j];
	TMultiGraph* mgr_dxVz=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dx*10000:z",Cut+zCut[i]);	  
		gr_dxVz_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_dxVz_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dxVz_Array[i]->SetMarkerStyle(6);	  
		mgr_dxVz->Add(gr_dxVz_Array[i],"p");
	}
	
	data->Project("h_dyVz", "dy*10000:z",Cut);
	TGraph* gr_dyVz_Array[j];
	TMultiGraph* mgr_dyVz=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dy*10000:z",Cut+zCut[i]);	  
		gr_dyVz_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_dyVz_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dyVz_Array[i]->SetMarkerStyle(6);	  
		mgr_dyVz->Add(gr_dyVz_Array[i],"p");
	}
	
	data->Project("h_drVphi", "dr*10000:phi",Cut);
	TGraph* gr_drVphi_Array[j];
	TMultiGraph* mgr_drVphi=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dr*10000:phi",Cut+zCut[i]);	  
		gr_drVphi_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_drVphi_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_drVphi_Array[i]->SetMarkerStyle(6);	  
		mgr_drVphi->Add(gr_drVphi_Array[i],"p");
	}
	
	data->Project("h_dzVphi", "dz*10000:phi",Cut);
	TGraph* gr_dzVphi_Array[j];
	TMultiGraph* mgr_dzVphi=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dz*10000:phi",Cut+zCut[i]);	  
		gr_dzVphi_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_dzVphi_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dzVphi_Array[i]->SetMarkerStyle(6);	  
		mgr_dzVphi->Add(gr_dzVphi_Array[i],"p");
	}
	
	data->Project("h_rdphiVphi", "r*dphi*10000:phi",Cut);
	TGraph* gr_rdphiVphi_Array[j];
	TMultiGraph* mgr_rdphiVphi=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("r*dphi*10000:phi",Cut+zCut[i]);	  
		gr_rdphiVphi_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_rdphiVphi_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_rdphiVphi_Array[i]->SetMarkerStyle(6);	  
		mgr_rdphiVphi->Add(gr_rdphiVphi_Array[i],"p");
	}

	data->Project("h_dxVphi", "dx*10000:phi",Cut);
	TGraph* gr_dxVphi_Array[j];
	TMultiGraph* mgr_dxVphi=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dx*10000:phi",Cut+zCut[i]);	  
		gr_dxVphi_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_dxVphi_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dxVphi_Array[i]->SetMarkerStyle(6);	  
		mgr_dxVphi->Add(gr_dxVphi_Array[i],"p");
	}
	
	data->Project("h_dyVphi", "dy*10000:phi",Cut);
	TGraph* gr_dyVphi_Array[j];
	TMultiGraph* mgr_dyVphi=new TMultiGraph();
	for ( int i = 0; i < j; i++) {
		data->Draw("dy*10000:phi",Cut+zCut[i]);	  
		gr_dyVphi_Array[i] = new TGraph(data->GetSelectedRows(),data->GetV2(),data->GetV1());
		gr_dyVphi_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dyVphi_Array[i]->SetMarkerStyle(6);	  
		mgr_dyVphi->Add(gr_dyVphi_Array[i],"p");
	}
	
	
	
	// ---------  draw histograms ---------
	TCanvas* c0 = new TCanvas("c0", "c0", 200, 10, 900, 300);
	c0->SetFillColor(0);
	c0->Divide(3,1);
	c0->cd(1);
	h_dr->Draw();
	c0->cd(2);
	h_dz->Draw();
	c0->cd(3);
	h_rdphi->Draw();
	if (savePlot) c0->Print((_outputDir+"plot3x1_"+plotName).c_str());
	
	//TStyle::SetTitleAlign(23)
	mgr_drVr->SetTitle("#Delta r vs. r");
	mgr_dzVr->SetTitle("#Delta z vs. r");           
	mgr_rdphiVr->SetTitle("r#Delta #phi vs. r");    
	mgr_dxVr->SetTitle("#Delta x vs. r");           
	mgr_dyVr->SetTitle("#Delta y vs. r");           
	mgr_drVz->SetTitle("#Delta r vs. z");           
	mgr_dzVz->SetTitle("#Delta z vs. z");           
	mgr_rdphiVz->SetTitle("r#Delta #phi vs. z");    
	mgr_dxVz->SetTitle("#Delta x vs. z");           
	mgr_dyVz->SetTitle("#Delta y vs. z");           
	mgr_drVphi->SetTitle("#Delta r vs. #phi");      
	mgr_dzVphi->SetTitle("#Delta z vs. #phi");      
	mgr_rdphiVphi->SetTitle("r#Delta #phi vs. #phi");
	mgr_dxVphi->SetTitle("#Delta x vs. #phi");      
	mgr_dyVphi->SetTitle("#Delta y vs. #phi");   
	
	// ---------  draw histograms ---------
	TCanvas* c = new TCanvas("c", "c", 200, 10, 1200, 700);
	c->SetFillColor(0);
	data->SetMarkerSize(0.5);
	data->SetMarkerStyle(6);
	c->Divide(5,3);
	c->cd(1);
	if (!autolimits) h_drVr->Draw();
	if (autolimits) mgr_drVr->Draw("a");
	c->Update();
	c->cd(2);
	if (!autolimits) h_dzVr->Draw();
	if (autolimits) mgr_dzVr->Draw("a");
	c->Update();
	c->cd(3);
	if (!autolimits) h_rdphiVr->Draw();
	if (autolimits) mgr_rdphiVr->Draw("a");
	c->cd(4);
	if (!autolimits) h_dxVr->Draw();
	if (autolimits) mgr_dxVr->Draw("a");
	c->cd(5);
	if (!autolimits) h_dyVr->Draw();
	if (autolimits) mgr_dyVr->Draw("a");
	c->cd(6);
	if (!autolimits) h_drVz->Draw();
	if (autolimits) mgr_drVz->Draw("a");
	c->cd(7);
	if (!autolimits) h_dzVz->Draw();
	if (autolimits) mgr_dzVz->Draw("a");
	c->cd(8);
	if (!autolimits) h_rdphiVz->Draw();
	if (autolimits) mgr_rdphiVz->Draw("a");
	c->cd(9);
	if (!autolimits) h_dxVz->Draw();
	if (autolimits) mgr_dxVz->Draw("a");
	c->cd(10);
	if (!autolimits) h_dyVz->Draw();
	if (autolimits) mgr_dyVz->Draw("a");
	c->cd(11);
	if (!autolimits) h_drVphi->Draw();
	if (autolimits) mgr_drVphi->Draw("a");
	c->cd(12);
	if (!autolimits) h_dzVphi->Draw();
	if (autolimits) mgr_dzVphi->Draw("a");
	c->cd(13);
	if (!autolimits) h_rdphiVphi->Draw();
	if (autolimits) mgr_rdphiVphi->Draw("a");
	c->cd(14);
	if (!autolimits) h_dxVphi->Draw();
	if (autolimits) mgr_dxVphi->Draw("a");
	c->cd(15);
	if (!autolimits) h_dyVphi->Draw();
	if (autolimits) mgr_dyVphi->Draw("a");
	
	mgr_drVr->GetXaxis()->SetTitle("r (cm)");
	mgr_dzVr->GetXaxis()->SetTitle("r (cm)");           
	mgr_rdphiVr->GetXaxis()->SetTitle("r (cm)");    
	mgr_dxVr->GetXaxis()->SetTitle("r (cm)");           
	mgr_dyVr->GetXaxis()->SetTitle("r (cm)");           
	mgr_drVz->GetXaxis()->SetTitle("z (cm)");           
	mgr_dzVz->GetXaxis()->SetTitle("z (cm)");           
	mgr_rdphiVz->GetXaxis()->SetTitle("z (cm)");    
	mgr_dxVz->GetXaxis()->SetTitle("z (cm)");           
	mgr_dyVz->GetXaxis()->SetTitle("z (cm)");           
	mgr_drVphi->GetXaxis()->SetTitle("#phi");      
	mgr_dzVphi->GetXaxis()->SetTitle("#phi");      
	mgr_rdphiVphi->GetXaxis()->SetTitle("#phi");
	mgr_dxVphi->GetXaxis()->SetTitle("#phi");      
	mgr_dyVphi->GetXaxis()->SetTitle("#phi");   
	
	mgr_drVr->GetYaxis()->SetTitle("#Delta r (#mum)");
	mgr_dzVr->GetYaxis()->SetTitle("#Delta z (#mum)");           
	mgr_rdphiVr->GetYaxis()->SetTitle("r#Delta #phi (#mum)");    
	mgr_dxVr->GetYaxis()->SetTitle("#Delta x (#mum)");           
	mgr_dyVr->GetYaxis()->SetTitle("#Delta y (#mum)");           
	mgr_drVz->GetYaxis()->SetTitle("#Delta r (#mum)");           
	mgr_dzVz->GetYaxis()->SetTitle("#Delta z (#mum)");           
	mgr_rdphiVz->GetYaxis()->SetTitle("r#Delta #phi (#mum)");    
	mgr_dxVz->GetYaxis()->SetTitle("#Delta x (#mum)");           
	mgr_dyVz->GetYaxis()->SetTitle("#Delta y (#mum)");           
	mgr_drVphi->GetYaxis()->SetTitle("#Delta r (#mum)");      
	mgr_dzVphi->GetYaxis()->SetTitle("#Delta z (#mum)");      
	mgr_rdphiVphi->GetYaxis()->SetTitle("r#Delta #phi (#mum)");
	mgr_dxVphi->GetYaxis()->SetTitle("#Delta x (#mum)");      
	mgr_dyVphi->GetYaxis()->SetTitle("#Delta y (#mum)");   
	
	mgr_drVr->GetYaxis()->SetTitleOffset(1.15);
	mgr_dzVr->GetYaxis()->SetTitleOffset(1.15);           
	mgr_rdphiVr->GetYaxis()->SetTitleOffset(1.15);    
	mgr_dxVr->GetYaxis()->SetTitleOffset(1.15);           
	mgr_dyVr->GetYaxis()->SetTitleOffset(1.15);           
	mgr_drVz->GetYaxis()->SetTitleOffset(1.15);           
	mgr_dzVz->GetYaxis()->SetTitleOffset(1.15);           
	mgr_rdphiVz->GetYaxis()->SetTitleOffset(1.15);    
	mgr_dxVz->GetYaxis()->SetTitleOffset(1.15);           
	mgr_dyVz->GetYaxis()->SetTitleOffset(1.15);           
	mgr_drVphi->GetYaxis()->SetTitleOffset(1.15);      
	mgr_dzVphi->GetYaxis()->SetTitleOffset(1.15);      
	mgr_rdphiVphi->GetYaxis()->SetTitleOffset(1.15);
	mgr_dxVphi->GetYaxis()->SetTitleOffset(1.15);      
	mgr_dyVphi->GetYaxis()->SetTitleOffset(1.15);   
	
	c->Update();
	
	// ---------  set output directory for histograms ---------
	plotDir->cd();
	h_dr->Write(); h_dz->Write(); h_rdphi->Write();
	h_drVr->Write(); h_dzVr->Write(); h_rdphiVr->Write(); h_dxVr->Write(); h_dyVr->Write();
	h_drVz->Write(); h_dzVz->Write(); h_rdphiVz->Write(); h_dxVz->Write(); h_dyVz->Write();
	h_drVphi->Write(); h_dzVphi->Write(); h_rdphiVphi->Write(); h_dxVphi->Write(); h_dyVphi->Write();
	
	if (savePlot) c->Print((_outputDir+"plot3x5_"+plotName).c_str());
	
}

void comparisonPlots::plot3x5Profile(TCut Cut, char* dirName, int nBins, bool savePlot, std::string plotName, bool autolimits,int ColorCode ){

  //int ColorCodeHalfBarrels = 1; 
  //color seperation of moodules corresponding to the different subdets
  //if you want to seperate corresponding to the halfbarrels (+z or -z) set colorcode !=1

  //for all histograms set the title centered
  gStyle->SetTitleFont(62);
  gStyle->SetOptTitle(0);
   
  gStyle->SetFrameBorderMode(0);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetPadColor(0);
  gStyle->SetCanvasColor(0);
  gStyle->SetTitleColor(1);
  gStyle->SetStatColor(0);
  gStyle->SetStatBorderSize(1);
  gStyle->SetFrameFillColor(0);

  gStyle->SetTextFont(132);
  gStyle->SetTextSize(0.08);
  gStyle->SetLabelFont(132,"x");
  gStyle->SetLabelFont(132,"y");
  gStyle->SetLabelFont(132,"z");
  gStyle->SetTitleSize(0.08,"x");
  gStyle->SetTitleSize(0.08,"y");
  gStyle->SetTitleSize(0.08,"z");
  gStyle->SetLabelSize(0.08,"x");
  gStyle->SetLabelSize(0.08,"y");
  gStyle->SetLabelSize(0.08,"z");

  gStyle->SetTitleAlign(22);
  gStyle->SetTitleX(0.5);
  gStyle->SetTitleY(0.97);
  gStyle->SetTitleFont(62);

  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);

  gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadRightMargin(0.05);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.18);
  
	// ---------  create directory for histograms ---------
	//const char* dirName = Cut;
	string s;// = "profile";
	s = s + dirName;
	s.append("_profile");
	TDirectory* plotDir = output->mkdir( s.data() );
	

	int j = 2;
	TCut zCut[6];
	zCut[0] = "z < 0";
	zCut[1] = "z >= 0";


	if (ColorCode == 1) {
	  j = 6;
          //	  zCut[6];
	  zCut[0] = "sublevel==1";
	  zCut[1] = "sublevel==2";
	  zCut[2] = "sublevel==3";
	  zCut[3] = "sublevel==4";
	  zCut[4] = "sublevel==5";
	  zCut[5] = "sublevel==6";
	  
	} 
	
	double minimumR, maximumR;
	double minimumZ, maximumZ;
	double minimumPhi, maximumPhi;
	double minimumDR, maximumDR;
	double minimumDZ, maximumDZ;
	double minimumRDPhi, maximumRDPhi;
	double minimumDX, maximumDX;
	double minimumDY, maximumDY;
	if (autolimits){
		// ---------  get right limits for histogram ---------
		TH1F* phr     = new TH1F("phr",     "phr",     200,  0,     200 );
		TH1F* phz     = new TH1F("phz",     "phz",     400,  -300,  300 );
		TH1F* phphi   = new TH1F("phphi",   "phphi",   200,  -3.15, 3.15);
		TH1F* phdr    = new TH1F("phdr",    "phdr",    20000,-1000,  1000 );
		TH1F* phdz    = new TH1F("phdz",    "phdz",    20000,-1000,  1000 );
		TH1F* phrdphi = new TH1F("phrdphi", "phrdphi", 20000,-1000,  1000 );
		TH1F* phdx    = new TH1F("phdx",    "phy",     20000,-1000,  1000 );
		TH1F* phdy    = new TH1F("phdy",    "phy",     20000,-1000,  1000 );
		data->Project("phr","r",Cut);
		data->Project("phz","z",Cut);
		data->Project("phphi","phi",Cut);
		data->Project("phdr","dr*10000",Cut);
		data->Project("phdz","dz*10000",Cut);
		data->Project("phrdphi","r*dphi*10000",Cut );
		data->Project("phdx","dx*10000",Cut );
		data->Project("phdy","dy*10000",Cut );
		getHistMaxMin(phr, maximumR, minimumR, 0);
		getHistMaxMin(phz, maximumZ, minimumZ, 0);
		getHistMaxMin(phphi, maximumPhi, minimumPhi, 0);
		getHistMaxMin(phdr, maximumDR, minimumDR, 1);
		getHistMaxMin(phdz, maximumDZ, minimumDZ, 1);
		getHistMaxMin(phrdphi, maximumRDPhi, minimumRDPhi, 1);
		getHistMaxMin(phdx, maximumDX, minimumDX, 1);
		getHistMaxMin(phdy, maximumDY, minimumDY, 1);
	}
	else{
		minimumR     = 0.,    maximumR     = 200.; 
		minimumZ     = -300., maximumZ     = 300.; 
		minimumPhi   = -3.15, maximumPhi   = 3.15;
		minimumDR    = -1,    maximumDR    = 1;
		minimumDZ    = -1,    maximumDZ    = 1;
		minimumRDPhi = -1,    maximumRDPhi = 1;
		minimumDX    = -1,    maximumDX    = 1;
		minimumDY    = -1,    maximumDY    = 1;
	}
	
	// ---------  declare histograms ---------
	TProfile* hprof_drVr = new TProfile("hprof_drVr","#Delta r vs. r",nBins,minimumR,maximumR,minimumDR,maximumDR);
	TProfile* hprof_dzVr = new TProfile("hprof_dzVr","#Delta z vs. r",nBins,minimumR,maximumR,minimumDZ,maximumDZ);
	TProfile* hprof_rdphiVr = new TProfile("hprof_rdphiVr","r#Delta #phi vs. r",nBins,minimumR,maximumR,minimumRDPhi,maximumRDPhi);
	TProfile* hprof_dxVr = new TProfile("hprof_dxVr","#Delta x vs. r", nBins,minimumR,maximumR,minimumDX,maximumDX);
	TProfile* hprof_dyVr = new TProfile("hprof_dyVr","#Delta y vs. r", nBins,minimumR,maximumR,minimumDY,maximumDY);
	TProfile* hprof_drVz = new TProfile("hprof_drVz","#Delta r vs. z", nBins,minimumZ,maximumZ,minimumDR,maximumDR);
	TProfile* hprof_dzVz = new TProfile("hprof_dzVz","#Delta z vs. z", nBins,minimumZ,maximumZ,minimumDZ,maximumDZ);
	TProfile* hprof_rdphiVz = new TProfile("hprof_rdphiVz","r#Delta #phi vs. z", nBins,minimumZ,maximumZ,minimumRDPhi,maximumRDPhi);
	TProfile* hprof_dxVz = new TProfile("hprof_dxVz","#Delta x vs. z", nBins,minimumZ,maximumZ,minimumDX,maximumDX);
	TProfile* hprof_dyVz = new TProfile("hprof_dyVz","#Delta y vs. z", nBins,minimumZ,maximumZ,minimumDY,maximumDY);
	TProfile* hprof_drVphi = new TProfile("hprof_drVphi","#Delta r vs. #phi", nBins,minimumPhi,maximumPhi,minimumDR,maximumDR);
	TProfile* hprof_dzVphi = new TProfile("hprof_dzVphi","#Delta z vs. #phi", nBins,minimumPhi,maximumPhi,minimumDZ,maximumDZ);
	TProfile* hprof_rdphiVphi = new TProfile("hprof_rdphiVphi","r#Delta #phi vs. #phi", nBins,minimumPhi,maximumPhi,minimumRDPhi,maximumRDPhi);
	TProfile* hprof_dxVphi = new TProfile("hprof_dxVphi","#Delta x vs. #phi", nBins,minimumPhi,maximumPhi,minimumDX,maximumDX);
	TProfile* hprof_dyVphi = new TProfile("hprof_dyVphi","#Delta y vs. #phi", nBins,minimumPhi,maximumPhi,minimumDY,maximumDY);
	
	TProfile* gr_drVr_Array[j];
	THStack* mgr_drVr=new THStack("mgr_drVr","");
	
	for ( int i = 0; i < j; i++) {
	  data->Draw("dr*10000:r>>hprof_drVr",Cut+zCut[i],"prof");
	  if (hprof_drVr->GetEntries()>0){
	    gr_drVr_Array[i] =(TProfile*)hprof_drVr->Clone() ;//data->Draw("dr:r",Cut+zCut[i],"prof");
	    gr_drVr_Array[i]->SetMarkerColor(int(i/4)+i+1);
	    gr_drVr_Array[i]->SetLineColor(int(i/4)+i+1);
	    mgr_drVr->Add(gr_drVr_Array[i]);
	  }
	}
	//y-axis scaled 2 times the max value for values >5 micron
	//for max values<5 micron set fixed range to 10 micron
	if ( (mgr_drVr->GetMaximum("nostack")>0.0005*10000)||(mgr_drVr->GetMinimum("nostack")<-0.0005*10000) ){
	  mgr_drVr->SetMaximum( 2.*mgr_drVr->GetMaximum("nostack") );
	  mgr_drVr->SetMinimum( 2.*mgr_drVr->GetMinimum("nostack") );
	}
	else{
	  mgr_drVr->SetMaximum(10.);
	  mgr_drVr->SetMinimum( -10.);
	}
	
	//data->Project("hprof_dzVr", "dz:r",Cut,"prof");
	TProfile* gr_dzVr_Array[j];
	THStack* mgr_dzVr=new THStack("mgr_dzVr","");
	for ( int i = 0; i < j; i++) {
	  data->Draw("dz*10000:r>>hprof_dzVr",Cut+zCut[i],"prof");	  
	  if (hprof_dzVr->GetEntries()>0){
	    gr_dzVr_Array[i] =(TProfile*)hprof_dzVr->Clone() ; 
	    gr_dzVr_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
	    gr_dzVr_Array[i]->SetLineColor(int(i/4)+i+1);
	    mgr_dzVr->Add(gr_dzVr_Array[i]);
	  }
	}
	 if ( (mgr_dzVr->GetMaximum("nostack")>0.0005*10000)||(mgr_dzVr->GetMinimum("nostack")<-0.0005*10000) ){
	   mgr_dzVr->SetMaximum( 2.*mgr_dzVr->GetMaximum("nostack") );
	   mgr_dzVr->SetMinimum( 2.*mgr_dzVr->GetMinimum("nostack") );
	 }	
	 else{
	   mgr_dzVr->SetMaximum( 10.);
	   mgr_dzVr->SetMinimum( -10.);
	 }

	//data->Project("hprof_rdphiVr", "r*dphi:r",Cut,"prof");
	TProfile* gr_rdphiVr_Array[j];
	THStack* mgr_rdphiVr=new THStack();
	for ( int i = 0; i < j; i++) {
	  data->Draw("r*dphi*10000:r>>hprof_rdphiVr",Cut+zCut[i],"prof");	  
	  if (hprof_rdphiVr->GetEntries()>0){
	    gr_rdphiVr_Array[i] = (TProfile*)hprof_rdphiVr->Clone() ; 
	    gr_rdphiVr_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
	    gr_rdphiVr_Array[i]->SetLineColor(int(i/4)+i+1);	  
	    mgr_rdphiVr->Add(gr_rdphiVr_Array[i]);
	  }
	}
	if ( (mgr_rdphiVr->GetMaximum("nostack")>0.0005*10000)||(mgr_rdphiVr->GetMinimum("nostack")<-0.0005*10000) ){
	mgr_rdphiVr->SetMaximum( 2.*mgr_rdphiVr->GetMaximum("nostack") );
	mgr_rdphiVr->SetMinimum( 2.*mgr_rdphiVr->GetMinimum("nostack") );
	}
	else{
	mgr_rdphiVr->SetMaximum( 10.);
	mgr_rdphiVr->SetMinimum( -10.);
	}
	  
	//data->Project("hprof_dxVr", "dx:r",Cut,"prof");
	TProfile* gr_dxVr_Array[j];
	THStack* mgr_dxVr=new THStack();
	for ( int i = 0; i < j; i++) {
	  data->Draw("dx*10000:r>>hprof_dxVr",Cut+zCut[i],"prof");
	  if (hprof_dxVr->GetEntries()>0){
	  gr_dxVr_Array[i] =  (TProfile*)hprof_dxVr->Clone() ; 
	  gr_dxVr_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
	  gr_dxVr_Array[i]->SetLineColor(int(i/4)+i+1);	  
	  mgr_dxVr->Add(gr_dxVr_Array[i]);
	  }
	}
	if ( (mgr_dxVr->GetMaximum("nostack")>0.0005*10000)||(mgr_dxVr->GetMinimum("nostack")<-0.0005*10000) ){
	mgr_dxVr->SetMaximum( 2.*mgr_dxVr->GetMaximum("nostack") );
	mgr_dxVr->SetMinimum( 2.*mgr_dxVr->GetMinimum("nostack") );
	}
	else{
	  mgr_dxVr->SetMaximum( 10.);
	  mgr_dxVr->SetMinimum( -10.);
        }
	
	//data->Project("hprof_dyVr", "dy:r",Cut,"prof");
	TProfile* gr_dyVr_Array[j];
	THStack* mgr_dyVr=new THStack();
	for ( int i = 0; i < j; i++) {
	  data->Draw("dy*1000:r>>hprof_dyVr",Cut+zCut[i],"prof");	  
	  if (hprof_dyVr->GetEntries()>0){
	    gr_dyVr_Array[i] = (TProfile*)hprof_dyVr->Clone() ;
	    gr_dyVr_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
	    gr_dyVr_Array[i]->SetLineColor(int(i/4)+i+1);	  
	    mgr_dyVr->Add(gr_dyVr_Array[i]);
	  }
	}
	if ( (mgr_dyVr->GetMaximum("nostack")>0.0005*10000)||(mgr_dyVr->GetMinimum("nostack")<-0.0005*10000) ){
	  mgr_dyVr->SetMaximum( 2.*mgr_dyVr->GetMaximum("nostack") );
	  mgr_dyVr->SetMinimum( 2.*mgr_dyVr->GetMinimum("nostack") );
	}
	else{
	  mgr_dyVr->SetMaximum( 10.);
	  mgr_dyVr->SetMinimum( -10.);
	}

	//data->Project("hprof_drVz", "dr:z",Cut,"prof");
	TProfile* gr_drVz_Array[j];
	THStack* mgr_drVz=new THStack();
	for ( int i = 0; i < j; i++) {
	  data->Draw("dr*10000:z>>hprof_drVz",Cut+zCut[i],"prof");	  
	  if (hprof_drVz->GetEntries()>0){
	    gr_drVz_Array[i] = (TProfile*)hprof_drVz->Clone() ; 
	    gr_drVz_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
	    gr_drVz_Array[i]->SetLineColor(int(i/4)+i+1);	  
	    mgr_drVz->Add(gr_drVz_Array[i]);
	  }
	}
	if ( (mgr_drVz->GetMaximum("nostack")>0.0005*10000)||(mgr_drVz->GetMinimum("nostack")<-0.0005*10000) ){
	mgr_drVz->SetMaximum( 2.*mgr_drVz->GetMaximum("nostack") );
	mgr_drVz->SetMinimum( 2.*mgr_drVz->GetMinimum("nostack") );
	}
	else{
	mgr_drVz->SetMaximum( 10.);
	mgr_drVz->SetMinimum( -10.);
	}

	//data->Project("hprof_dzVz", "dz:z",Cut,"prof");
	TProfile* gr_dzVz_Array[j];
	THStack* mgr_dzVz=new THStack();
	for ( int i = 0; i < j; i++) {
		data->Draw("dz*10000:z>>hprof_dzVz",Cut+zCut[i],"prof");	  
		if (hprof_dzVz->GetEntries()>0){
		gr_dzVz_Array[i] =  (TProfile*)hprof_dzVz->Clone() ;
		gr_dzVz_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dzVz_Array[i]->SetLineColor(int(i/4)+i+1);	  
		mgr_dzVz->Add(gr_dzVz_Array[i]);
		}
	}
	if ( (mgr_dzVz->GetMaximum("nostack")>0.0005*10000)||(mgr_dzVz->GetMinimum("nostack")<-0.0005*10000) ){
	mgr_dzVz->SetMaximum( 2.*mgr_dzVz->GetMaximum("nostack") );
	mgr_dzVz->SetMinimum( 2.*mgr_dzVz->GetMinimum("nostack") );
	}
	else{
	mgr_dzVz->SetMaximum( 10.);
	mgr_dzVz->SetMinimum( -10.);
	}

	//data->Project("hprof_rdphiVz", "r*dphi:z",Cut,"prof");
	TProfile* gr_rdphiVz_Array[j];
	THStack* mgr_rdphiVz=new THStack();
	for ( int i = 0; i < j; i++) {
	  data->Draw("r*dphi*10000:z>>hprof_rdphiVz",Cut+zCut[i],"prof");	  
	  if (hprof_rdphiVz->GetEntries()>0){
	    gr_rdphiVz_Array[i] = (TProfile*)hprof_rdphiVz->Clone() ; 
	    gr_rdphiVz_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
	    gr_rdphiVz_Array[i]->SetLineColor(int(i/4)+i+1);
	    mgr_rdphiVz->Add(gr_rdphiVz_Array[i]);
	  }
	}
	if ( (mgr_rdphiVz->GetMaximum("nostack")>0.0005*10000)||(mgr_rdphiVz->GetMinimum("nostack")<-0.0005*10000) ){
	mgr_rdphiVz->SetMaximum( 2.*mgr_rdphiVz->GetMaximum("nostack") );
	mgr_rdphiVz->SetMinimum( 2.*mgr_rdphiVz->GetMinimum("nostack") );
	}
	else{
	mgr_rdphiVz->SetMaximum( 10.);
	mgr_rdphiVz->SetMinimum( -10.);
	}

	//data->Project("hprof_dxVz", "dx:z",Cut,"prof");
	TProfile* gr_dxVz_Array[j];
	THStack* mgr_dxVz=new THStack();
	for ( int i = 0; i < j; i++) {
	  data->Draw("dx*10000:z>>hprof_dxVz",Cut+zCut[i],"prof");	  
	  if (hprof_dxVz->GetEntries()>0){
	  gr_dxVz_Array[i] = (TProfile*)hprof_dxVz->Clone() ; 
		gr_dxVz_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dxVz_Array[i]->SetLineColor(int(i/4)+i+1);	  
		mgr_dxVz->Add(gr_dxVz_Array[i]);
	  }
	}
	if ( (mgr_dxVz->GetMaximum("nostack")>0.0005*10000)||(mgr_dxVz->GetMinimum("nostack")<-0.0005*10000) ){
	mgr_dxVz->SetMaximum( 2.*mgr_dxVz->GetMaximum("nostack") );
	mgr_dxVz->SetMinimum( 2.*mgr_dxVz->GetMinimum("nostack") );
	}
	else{
	  mgr_dxVz->SetMaximum( 10.);
	  mgr_dxVz->SetMinimum( -10.);
	}
	
	//data->Project("hprof_dyVz", "dy:z",Cut,"prof");
	TProfile* gr_dyVz_Array[j];
	THStack* mgr_dyVz=new THStack();
	for ( int i = 0; i < j; i++) {
		data->Draw("dy*10000:z>>hprof_dyVz",Cut+zCut[i],"prof");
		if (hprof_dyVz->GetEntries()>0){
		gr_dyVz_Array[i] = (TProfile*)hprof_dyVz->Clone() ; 
		gr_dyVz_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dyVz_Array[i]->SetLineColor(int(i/4)+i+1);	  
		mgr_dyVz->Add(gr_dyVz_Array[i],"p");
		}
	}
	if ( (mgr_dyVz->GetMaximum("nostack")>0.0005*10000)||(mgr_dyVz->GetMinimum("nostack")<-0.0005*10000) ){
	  mgr_dyVz->SetMaximum( 2.*mgr_dyVz->GetMaximum("nostack") );
	  mgr_dyVz->SetMinimum( 2.*mgr_dyVz->GetMinimum("nostack") );
	}
	else{
	  mgr_dyVz->SetMaximum( 10.);
	  mgr_dyVz->SetMinimum( -10.);
	}

	//data->Project("hprof_drVphi", "dr:phi",Cut,"prof");
	TProfile* gr_drVphi_Array[j];
	THStack* mgr_drVphi=new THStack();
	for ( int i = 0; i < j; i++) {
		data->Draw("dr*10000:phi>>hprof_drVphi",Cut+zCut[i],"prof");	  
		if (hprof_drVphi->GetEntries()>0){
		gr_drVphi_Array[i] = (TProfile*)hprof_drVphi->Clone() ; 
		gr_drVphi_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_drVphi_Array[i]->SetLineColor(int(i/4)+i+1);
		mgr_drVphi->Add(gr_drVphi_Array[i],"p");
	}
	}
	if ( (mgr_drVphi->GetMaximum("nostack")>0.0005*10000)||(mgr_drVphi->GetMinimum("nostack")<-0.0005*10000) ){
	mgr_drVphi->SetMaximum( 2.*mgr_drVphi->GetMaximum("nostack") );
	mgr_drVphi->SetMinimum( 2.*mgr_drVphi->GetMinimum("nostack") );
	}
	else{
	  mgr_drVphi->SetMaximum( 10.);
	  mgr_drVphi->SetMinimum( -10.);
	}

	//data->Project("hprof_dzVphi", "dz:phi",Cut,"prof");
	TProfile* gr_dzVphi_Array[j];
	THStack* mgr_dzVphi=new THStack();
	for ( int i = 0; i < j; i++) {
		data->Draw("dz*10000:phi>>hprof_dzVphi",Cut+zCut[i],"prof");	  
		if (hprof_dzVphi->GetEntries()>0){
		gr_dzVphi_Array[i] =  (TProfile*)hprof_dzVphi->Clone() ; 
		gr_dzVphi_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dzVphi_Array[i]->SetLineColor(int(i/4)+i+1);
		mgr_dzVphi->Add(gr_dzVphi_Array[i],"p");
	}
	}
	if ( (mgr_dzVphi->GetMaximum("nostack")>0.0005*10000)||(mgr_dzVphi->GetMinimum("nostack")<-0.0005*10000) ){
	mgr_dzVphi->SetMaximum( 2.*mgr_dzVphi->GetMaximum("nostack") );
	mgr_dzVphi->SetMinimum( 2.*mgr_dzVphi->GetMinimum("nostack") );
	}
	else{
	mgr_dzVphi->SetMaximum( 10.);
	mgr_dzVphi->SetMinimum( -10.);
	}

	//data->Project("hprof_rdphiVphi", "r*dphi:phi",Cut,"prof");
	TProfile* gr_rdphiVphi_Array[j];
	THStack* mgr_rdphiVphi=new THStack();
	for ( int i = 0; i < j; i++) {
		data->Draw("r*dphi*10000:phi>>hprof_rdphiVphi",Cut+zCut[i],"prof");	  
		if (hprof_rdphiVphi->GetEntries()>0){
		gr_rdphiVphi_Array[i] =  (TProfile*)hprof_rdphiVphi->Clone() ; 
		gr_rdphiVphi_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_rdphiVphi_Array[i]->SetLineColor(int(i/4)+i+1);	  
		mgr_rdphiVphi->Add(gr_rdphiVphi_Array[i],"p");
	}
	}
	if ( (mgr_rdphiVphi->GetMaximum("nostack")>0.0005*10000)||(mgr_rdphiVphi->GetMinimum("nostack")<-0.0005*10000) ){
	mgr_rdphiVphi->SetMaximum( 2.*mgr_rdphiVphi->GetMaximum("nostack") );
	mgr_rdphiVphi->SetMinimum( 2.*mgr_rdphiVphi->GetMinimum("nostack") );
	}
	else{
	  mgr_rdphiVphi->SetMaximum( 10.);
	  mgr_rdphiVphi->SetMinimum( -10.);
	}

	//data->Project("hprof_dxVphi", "dx:phi",Cut,"prof");
	TProfile* gr_dxVphi_Array[j];
	THStack* mgr_dxVphi=new THStack();
	for ( int i = 0; i < j; i++) {
		data->Draw("dx*10000:phi>>hprof_dxVphi",Cut+zCut[i],"prof");
		if (hprof_dxVphi->GetEntries()>0){
		gr_dxVphi_Array[i] =  (TProfile*)hprof_dxVphi->Clone() ; 
		gr_dxVphi_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dxVphi_Array[i]->SetLineColor(int(i/4)+i+1);
		mgr_dxVphi->Add(gr_dxVphi_Array[i],"p");
		}
	}
	if ( (mgr_dxVphi->GetMaximum("nostack")>0.0005*10000)||(mgr_dxVphi->GetMinimum("nostack")<-0.0005*10000) ){
	mgr_dxVphi->SetMaximum( 2.*mgr_dxVphi->GetMaximum("nostack") );
	mgr_dxVphi->SetMinimum( 2.*mgr_dxVphi->GetMinimum("nostack") );
	}
	else{
	mgr_dxVphi->SetMaximum( 10.);
	mgr_dxVphi->SetMinimum( -10.);
	}

	//data->Project("hprof_dyVphi", "dy:phi",Cut,"prof");
	TProfile* gr_dyVphi_Array[j];
	THStack* mgr_dyVphi=new THStack();
	for ( int i = 0; i < j; i++) {
		data->Draw("dy*10000:phi>>hprof_dyVphi",Cut+zCut[i],"prof");	  
		if (hprof_dyVphi->GetEntries()>0){
		gr_dyVphi_Array[i] =  (TProfile*)hprof_dyVphi->Clone() ; 
		gr_dyVphi_Array[i]->SetMarkerColor(int(i/4)+i+1);	  
		gr_dyVphi_Array[i]->SetLineColor(int(i/4)+i+1);
		mgr_dyVphi->Add(gr_dyVphi_Array[i],"p");
	}
	}
        if ( (mgr_dyVphi->GetMaximum("nostack")>0.0005*10000)||(mgr_dyVphi->GetMinimum("nostack")<-0.0005*10000) ){
        	mgr_dyVphi->SetMaximum( 2.*mgr_dyVphi->GetMaximum("nostack") );
        	mgr_dyVphi->SetMinimum( 2.*mgr_dyVphi->GetMinimum("nostack") );
        }
        else{
               mgr_dyVphi->SetMaximum( 10.);
               mgr_dyVphi->SetMinimum( -10.);
        }
	
	mgr_drVr->SetTitle("#Delta r vs. r");           
	mgr_dzVr->SetTitle("#Delta z vs. r");           
	mgr_rdphiVr->SetTitle("r#Delta #phi vs. r");    
	mgr_dxVr->SetTitle("#Delta x vs. r");           
	mgr_dyVr->SetTitle("#Delta y vs. r");           
	mgr_drVz->SetTitle("#Delta r vs. z");           
	mgr_dzVz->SetTitle("#Delta z vs. z");           
	mgr_rdphiVz->SetTitle("r#Delta #phi vs. z");    
	mgr_dxVz->SetTitle("#Delta x vs. z");           
	mgr_dyVz->SetTitle("#Delta y vs. z");           
	mgr_drVphi->SetTitle("#Delta r vs. #phi");      
	mgr_dzVphi->SetTitle("#Delta z vs. #phi");      
	mgr_rdphiVphi->SetTitle("r#Delta #phi vs. #phi");
	mgr_dxVphi->SetTitle("#Delta x vs. #phi");      
	mgr_dyVphi->SetTitle("#Delta y vs. #phi");   
	
	// ---------  draw histograms ---------
	TCanvas* c = new TCanvas("c", "c", 200, 10, 1200, 700);
	c->SetFillColor(0);
	data->SetMarkerSize(0.5);
	data->SetMarkerStyle(6);
	c->Divide(5,3);
	c->cd(1);
	if (!autolimits) mgr_drVr->Draw("nostack");
	if (autolimits) mgr_drVr->Draw("nostack");
	c->Update();
	c->cd(2);
	if (!autolimits) mgr_dzVr->Draw("nostack");
	if (autolimits) mgr_dzVr->Draw("nostack");
	c->Update();
	c->cd(3);
	if (!autolimits)  mgr_rdphiVr->Draw("nostack");
	if (autolimits) mgr_rdphiVr->Draw("nostack");
	c->cd(4);
	if (!autolimits) mgr_dxVr->Draw("nostack"); 
	if (autolimits) mgr_dxVr->Draw("nostack");
	c->cd(5);
	if (!autolimits) mgr_dyVr->Draw("nostack");
	if (autolimits) mgr_dyVr->Draw("nostack");
	c->cd(6);
	if (!autolimits) mgr_drVz->Draw("nostack");
	if (autolimits) mgr_drVz->Draw("nostack");
	c->cd(7);
	if (!autolimits)  mgr_dzVz->Draw("nostack");
	if (autolimits) mgr_dzVz->Draw("nostack");
	c->cd(8);
	if (!autolimits) mgr_rdphiVz->Draw("nostack");
	if (autolimits) mgr_rdphiVz->Draw("nostack");
	c->cd(9);
	if (!autolimits) mgr_dxVz->Draw("nostack");
	if (autolimits) mgr_dxVz->Draw("nostack");
	c->cd(10);
	if (!autolimits)  mgr_dyVz->Draw("nostack");
	if (autolimits) mgr_dyVz->Draw("nostack");
	c->cd(11);
	if (!autolimits) mgr_drVphi->Draw("nostack");
	if (autolimits) mgr_drVphi->Draw("nostack");
	c->cd(12);
	if (!autolimits) mgr_dzVphi->Draw("nostack");
	if (autolimits) mgr_dzVphi->Draw("nostack");
	c->cd(13);
	if (!autolimits) mgr_rdphiVphi->Draw("nostack");
	if (autolimits) mgr_rdphiVphi->Draw("nostack");
	c->cd(14);
	if (!autolimits) mgr_dxVphi->Draw("nostack");
	if (autolimits) mgr_dxVphi->Draw("nostack");
	c->cd(15);
	if (!autolimits)  mgr_dyVphi->Draw("nostack");
	if (autolimits) mgr_dyVphi->Draw("nostack");
	
	mgr_drVr->GetXaxis()->SetTitle("r (cm)");
	mgr_dzVr->GetXaxis()->SetTitle("r (cm)");           
	mgr_rdphiVr->GetXaxis()->SetTitle("r (cm)");    
	mgr_dxVr->GetXaxis()->SetTitle("r (cm)");           
	mgr_dyVr->GetXaxis()->SetTitle("r (cm)");           
	mgr_drVz->GetXaxis()->SetTitle("z (cm)");           
	mgr_dzVz->GetXaxis()->SetTitle("z (cm)");           
	mgr_rdphiVz->GetXaxis()->SetTitle("z (cm)");    
	mgr_dxVz->GetXaxis()->SetTitle("z (cm)");           
	mgr_dyVz->GetXaxis()->SetTitle("z (cm)");           
	mgr_drVphi->GetXaxis()->SetTitle("#phi");      
	mgr_dzVphi->GetXaxis()->SetTitle("#phi");      
	mgr_rdphiVphi->GetXaxis()->SetTitle("#phi");
	mgr_dxVphi->GetXaxis()->SetTitle("#phi");      
	mgr_dyVphi->GetXaxis()->SetTitle("#phi");   
	
	mgr_drVr->GetYaxis()->SetTitle("#Delta r (#mum)");
	mgr_dzVr->GetYaxis()->SetTitle("#Delta z (#mum)");           
	mgr_rdphiVr->GetYaxis()->SetTitle("r#Delta #phi (#mum)");    
	mgr_dxVr->GetYaxis()->SetTitle("#Delta x (#mum)");           
	mgr_dyVr->GetYaxis()->SetTitle("#Delta y (#mum)");           
	mgr_drVz->GetYaxis()->SetTitle("#Delta r (#mum)");           
	mgr_dzVz->GetYaxis()->SetTitle("#Delta z (#mum)");           
	mgr_rdphiVz->GetYaxis()->SetTitle("r#Delta #phi (#mum)");    
	mgr_dxVz->GetYaxis()->SetTitle("#Delta x (#mum)");           
	mgr_dyVz->GetYaxis()->SetTitle("#Delta y (#mum)");           
	mgr_drVphi->GetYaxis()->SetTitle("#Delta r (#mum)");      
	mgr_dzVphi->GetYaxis()->SetTitle("#Delta z (#mum)");      
	mgr_rdphiVphi->GetYaxis()->SetTitle("r#Delta #phi (#mum)");
	mgr_dxVphi->GetYaxis()->SetTitle("#Delta x (#mum)");      
	mgr_dyVphi->GetYaxis()->SetTitle("#Delta y (#mum)");   
	
	mgr_drVr->GetYaxis()->SetTitleOffset(1.15);
	mgr_dzVr->GetYaxis()->SetTitleOffset(1.15);           
	mgr_rdphiVr->GetYaxis()->SetTitleOffset(1.15);    
	mgr_dxVr->GetYaxis()->SetTitleOffset(1.15);           
	mgr_dyVr->GetYaxis()->SetTitleOffset(1.15);           
	mgr_drVz->GetYaxis()->SetTitleOffset(1.15);           
	mgr_dzVz->GetYaxis()->SetTitleOffset(1.15);           
	mgr_rdphiVz->GetYaxis()->SetTitleOffset(1.15);    
	mgr_dxVz->GetYaxis()->SetTitleOffset(1.15);           
	mgr_dyVz->GetYaxis()->SetTitleOffset(1.15);           
	mgr_drVphi->GetYaxis()->SetTitleOffset(1.15);      
	mgr_dzVphi->GetYaxis()->SetTitleOffset(1.15);      
	mgr_rdphiVphi->GetYaxis()->SetTitleOffset(1.15);
	mgr_dxVphi->GetYaxis()->SetTitleOffset(1.15);      
	mgr_dyVphi->GetYaxis()->SetTitleOffset(1.15);   
	
	c->Update();
	
	// ---------  set output directory for histograms ---------
	plotDir->cd();
	
	hprof_drVr->Write(); hprof_dzVr->Write(); hprof_rdphiVr->Write(); hprof_dxVr->Write(); hprof_dyVr->Write();
	hprof_drVz->Write(); hprof_dzVz->Write(); hprof_rdphiVz->Write(); hprof_dxVz->Write(); hprof_dyVz->Write();
	hprof_drVphi->Write(); hprof_dzVphi->Write(); hprof_rdphiVphi->Write(); hprof_dxVphi->Write(); hprof_dyVphi->Write();
	
	/*
	// ---------  project tree onto histograms ---------
	data->Project("hprof_drVr", "dr:r",Cut ,"prof");
	data->Project("hprof_dzVr", "dz:r",Cut ,"prof");
	data->Project("hprof_rdphiVr", "r*dphi:r",Cut ,"prof");
	data->Project("hprof_dxVr", "dx:r",Cut ,"prof");
	data->Project("hprof_dyVr", "dy:r",Cut ,"prof");
	data->Project("hprof_drVz", "dr:z",Cut ,"prof");
	data->Project("hprof_dzVz", "dz:z",Cut ,"prof");
	data->Project("hprof_rdphiVz", "r*dphi:z",Cut ,"prof");
	data->Project("hprof_dxVz", "dx:z",Cut ,"prof");
	data->Project("hprof_dyVz", "dy:z",Cut ,"prof");
	data->Project("hprof_drVphi", "dr:phi",Cut ,"prof");
	data->Project("hprof_dzVphi", "dz:phi",Cut ,"prof");
	data->Project("hprof_rdphiVphi", "r*dphi:phi",Cut ,"prof");
	data->Project("hprof_dxVphi", "dx:phi",Cut ,"prof");
	data->Project("hprof_dyVphi", "dy:phi",Cut ,"prof");
	
	// ---------  draw histograms ---------
	TCanvas* cp = new TCanvas("cp", "cp", 200, 10, 1200, 700);
	cp->SetFillColor(0);
	data->SetMarkerSize(0.5);
	data->SetMarkerStyle(6);
	cp->Divide(5,3);
	cp->cd(1);
	hprof_drVr->Draw();
	cp->cd(2);
	hprof_dzVr->Draw();
	cp->cd(3);
	hprof_rdphiVr->Draw();
	cp->cd(4);
	hprof_dxVr->Draw();
	cp->cd(5);
	hprof_dyVr->Draw();
	cp->cd(6);
	hprof_drVz->Draw();
	cp->cd(7);
	hprof_dzVz->Draw();
	cp->cd(8);
	hprof_rdphiVz->Draw();
	cp->cd(9);
	hprof_dxVz->Draw();
	cp->cd(10);
	hprof_dyVz->Draw();
	cp->cd(11);
	hprof_drVphi->Draw();
	cp->cd(12);
	hprof_dzVphi->Draw();
	cp->cd(13);
	hprof_rdphiVphi->Draw();
	cp->cd(14);
	hprof_dxVphi->Draw();
	cp->cd(15);
	hprof_dyVphi->Draw();
	
	// ---------  set output directory for histograms ---------
	plotDir->cd();
	hprof_drVr->Write(); hprof_dzVr->Write(); hprof_rdphiVr->Write(); hprof_dxVr->Write(); hprof_dyVr->Write();
	hprof_drVz->Write(); hprof_dzVz->Write(); hprof_rdphiVz->Write(); hprof_dxVz->Write(); hprof_dyVz->Write();
	hprof_drVphi->Write(); hprof_dzVphi->Write(); hprof_rdphiVphi->Write(); hprof_dxVphi->Write(); hprof_dyVphi->Write();
	*/
	if (savePlot) c->Print((_outputDir+"plot3x5Prof_"+plotName).c_str());
}


void comparisonPlots::getMaxMin(){
	
	data->GetEntry(0);
	minR = r_; maxR = r_;
	minZ = z_; maxZ = z_;
	minPhi = phi_; maxPhi = phi_;
	minDR = dr_; maxDR = dr_;
	minDZ = dz_; maxDZ = dz_;
	minRDPhi = r_*dphi_; maxRDPhi = r_*dphi_;
	minDX = dx_; maxDX = dx_;
	minDY = dy_; maxDY = dy_;
	
	int nEntries = data->GetEntries();
	for (int i = 1; i < nEntries; ++i){
		data->GetEntry(i);
		
		if (r_ < minR) minR = r_;
		if (r_ > maxR) maxR = r_;
		if (z_ < minZ) minZ = z_;
		if (z_ > maxZ) maxZ = z_;
		if (phi_ < minPhi) minPhi = phi_;
		if (phi_ > maxPhi) maxPhi = phi_;
		if (dr_ < minDR) minDR = dr_;
		if (dr_ > maxDR) maxDR = dr_;
		if (dz_ < minDZ) minDZ = dz_;
		if (dz_ > maxDZ) maxDZ = dz_;
		if (r_*dphi_ < minRDPhi) minRDPhi = r_*dphi_;
		if (r_*dphi_ > maxRDPhi) maxRDPhi = r_*dphi_;
		if (dx_ < minDX) minDX = dx_;
		if (dx_ > maxDX) maxDX = dx_;
		if (dy_ < minDY) minDY = dy_;
		if (dy_ > maxDY) maxDY = dy_;
	}
}


void comparisonPlots::getHistMaxMin(TH1* hist, double &max, double &min, int flag){
	
	int nBins = hist->GetNbinsX();
	for (int i = 0; i < nBins; ++i){
		double binContent = hist->GetBinContent(i);
		if (binContent > 0){
			//double binWidth = hist->GetBinLowEdge(i) - hist->GetBinLowEdge(i-1);
			//std::cout << "bin width1: " << hist->GetBinWidth(i) << ", bin width2: " << binWidth << std::endl;
			if (flag == 0) max = hist->GetBinLowEdge(i) + 2.*hist->GetBinWidth(i) + 0.1*hist->GetBinLowEdge(i);
			if (flag == 1) max = hist->GetBinLowEdge(i) + hist->GetBinWidth(i) + 0.1*hist->GetBinLowEdge(i);
		}
	}
	for (int i = (nBins-1); i >= 0; i--){
		double binContent = hist->GetBinContent(i);
		if (binContent > 0) min = hist->GetBinLowEdge(i) - 0.1*hist->GetBinLowEdge(i);
	}
	//std::cout << "max: " << max << ", min: " << min << std::endl;
}

void comparisonPlots::Write()
{
	output->Write();
}


