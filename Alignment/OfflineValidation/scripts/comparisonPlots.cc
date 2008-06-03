#include "comparisonPlots.h"
#include <string>
#include <sstream>

#include "TProfile.h"
#include "TList.h"
#include "TNtuple.h"
#include "TString.h"
#include <iostream>
#include "TStyle.h"

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

void comparisonPlots::plot3x5(TCut Cut, char* dirName, bool savePlot, std::string plotName){
	
	// ---------  create directory for histograms ---------
	//const char* dirName = Cut;
	TDirectory* plotDir = output->mkdir( dirName );
	
	// ---------  get right limits for histogram ---------
	/*
	TH1F* hr = new TH1F("hr", "hr", 200, 0, 200);
	TH1F* hz = new TH1F("hz", "hz", 400, -200, 200);
	TH1F* hphi = new TH1F("hphi", "hphi", 200, -3.15, 3.15);
	TH1F* hdr = new TH1F("hdr", "hdr", 2000, -10, 10);
	TH1F* hdz = new TH1F("hdz", "hdz", 2000, -10, 10);
	TH1F* hrdphi = new TH1F("hrdphi", "hrdphi", 2000, -10, 10);
	TH1F* hdx = new TH1F("hdx", "hy", 2000, -10, 10);
	TH1F* hdy = new TH1F("hdy", "hy", 2000, -10, 10);
	data->Project("hr","r",Cut);
	data->Project("hz","z",Cut);
	data->Project("hphi","phi",Cut);
	data->Project("hdr","dr",Cut);
	data->Project("hdz","dz",Cut);
	data->Project("hrdphi","r*dphi",Cut);
	data->Project("hdx","dx",Cut);
	data->Project("hdy","dy",Cut);
	double minimumR, maximumR; getHistMaxMin(hr, maximumR, minimumR, 0);
	double minimumZ, maximumZ; getHistMaxMin(hz, maximumZ, minimumZ, 0);
	double minimumPhi, maximumPhi; getHistMaxMin(hphi, maximumPhi, minimumPhi, 0);
	double minimumDR, maximumDR; getHistMaxMin(hdr, maximumDR, minimumDR, 1);
	double minimumDZ, maximumDZ; getHistMaxMin(hdz, maximumDZ, minimumDZ, 1);
	double minimumRDPhi, maximumRDPhi; getHistMaxMin(hrdphi, maximumRDPhi, minimumRDPhi, 1);
	double minimumDX, maximumDX; getHistMaxMin(hdx, maximumDX, minimumDX, 1);
	double minimumDY, maximumDY; getHistMaxMin(hdy, maximumDY, minimumDY, 1);
	*/
	double minimumR = 0., maximumR = 200.; 
	double minimumZ = -200., maximumZ = 200.; 
	double minimumPhi = -3.15, maximumPhi = 3.15;
	double minimumDR = -1, maximumDR = 1;
	double minimumDZ = -1, maximumDZ = 1;
	double minimumRDPhi = -1, maximumRDPhi = 1;
	double minimumDX = -1, maximumDX = 1;
	double minimumDY = -1, maximumDY = 1;
	
	
	
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
	
	// ---------  project tree onto histograms ---------
	data->Project("h_dr","dr",Cut);
	data->Project("h_dz","dz",Cut);
	data->Project("h_rdphi","r*dphi",Cut);
	data->Project("h_drVr", "dr:r",Cut);
	data->Project("h_dzVr", "dz:r",Cut);
	data->Project("h_rdphiVr", "r*dphi:r",Cut);
	data->Project("h_dxVr", "dx:r",Cut);
	data->Project("h_dyVr", "dy:r",Cut);
	data->Project("h_drVz", "dr:z",Cut);
	data->Project("h_dzVz", "dz:z",Cut);
	data->Project("h_rdphiVz", "r*dphi:z",Cut);
	data->Project("h_dxVz", "dx:z",Cut);
	data->Project("h_dyVz", "dy:z",Cut);
	data->Project("h_drVphi", "dr:phi",Cut);
	data->Project("h_dzVphi", "dz:phi",Cut);
	data->Project("h_rdphiVphi", "r*dphi:phi",Cut);
	data->Project("h_dxVphi", "dx:phi",Cut);
	data->Project("h_dyVphi", "dy:phi",Cut);
	 
	
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

	
	// ---------  draw histograms ---------
	TCanvas* c = new TCanvas("c", "c", 200, 10, 1200, 700);
	c->SetFillColor(0);
	data->SetMarkerSize(0.5);
	data->SetMarkerStyle(6);
	c->Divide(5,3);
	c->cd(1);
	h_drVr->Draw();
	c->cd(2);
	h_dzVr->Draw();
	c->cd(3);
	h_rdphiVr->Draw();
	c->cd(4);
	h_dxVr->Draw();
	c->cd(5);
	h_dyVr->Draw();
	c->cd(6);
	h_drVz->Draw();
	c->cd(7);
	h_dzVz->Draw();
	c->cd(8);
	h_rdphiVz->Draw();
	c->cd(9);
	h_dxVz->Draw();
	c->cd(10);
	h_dyVz->Draw();
	c->cd(11);
	h_drVphi->Draw();
	c->cd(12);
	h_dzVphi->Draw();
	c->cd(13);
	h_rdphiVphi->Draw();
	c->cd(14);
	h_dxVphi->Draw();
	c->cd(15);
	h_dyVphi->Draw();
	
	// ---------  set output directory for histograms ---------
	plotDir->cd();
	h_dr->Write(); h_dz->Write(); h_rdphi->Write();
	h_drVr->Write(); h_dzVr->Write(); h_rdphiVr->Write(); h_dxVr->Write(); h_dyVr->Write();
	h_drVz->Write(); h_dzVz->Write(); h_rdphiVz->Write(); h_dxVz->Write(); h_dyVz->Write();
	h_drVphi->Write(); h_dzVphi->Write(); h_rdphiVphi->Write(); h_dxVphi->Write(); h_dyVphi->Write();
	
	if (savePlot) c->Print((_outputDir+"plot3x5_"+plotName).c_str());
	
}

void comparisonPlots::plot3x5Profile(TCut Cut, char* dirName, int nBins, bool savePlot, std::string plotName){
	
	// ---------  create directory for histograms ---------
	//const char* dirName = Cut;
	string s;// = "profile";
	s = s + dirName;
	s.append("_profile");
	TDirectory* plotDir = output->mkdir( s.data() );

	/*
	// ---------  get right limits for histogram ---------
	TH1F* phr = new TH1F("phr", "phr", 200, 0, 200);
	TH1F* phz = new TH1F("phz", "phz", 400, -200, 200);
	TH1F* phphi = new TH1F("phphi", "phphi", 200, -3.15, 3.15);
	TH1F* phdr = new TH1F("phdr", "phdr", 2000, -10, 10);
	TH1F* phdz = new TH1F("phdz", "phdz", 2000, -10, 10);
	TH1F* phrdphi = new TH1F("phrdphi", "phrdphi", 200, -10, 10);
	TH1F* phdx = new TH1F("phdx", "phy", 2000, -10, 10);
	TH1F* phdy = new TH1F("phdy", "phy", 2000, -10, 10);
	data->Project("phr","r",Cut);
	data->Project("phz","z",Cut);
	data->Project("phphi","phi",Cut);
	data->Project("phdr","dr",Cut);
	data->Project("phdz","dz",Cut);
	data->Project("phrdphi","r*dphi",Cut);
	data->Project("phdx","dx",Cut);
	data->Project("phdy","dy",Cut);
	double minimumR, maximumR; getHistMaxMin(phr, maximumR, minimumR, 0);
	double minimumZ, maximumZ; getHistMaxMin(phz, maximumZ, minimumZ, 0);
	double minimumPhi, maximumPhi; getHistMaxMin(phphi, maximumPhi, minimumPhi, 0);
	double minimumDR, maximumDR; getHistMaxMin(phdr, maximumDR, minimumDR, 1);
	double minimumDZ, maximumDZ; getHistMaxMin(phdz, maximumDZ, minimumDZ, 1);
	double minimumRDPhi, maximumRDPhi; getHistMaxMin(phrdphi, maximumRDPhi, minimumRDPhi, 1);
	double minimumDX, maximumDX; getHistMaxMin(phdx, maximumDX, minimumDX, 1);
	double minimumDY, maximumDY; getHistMaxMin(phdy, maximumDY, minimumDY, 1);
	*/
	double minimumR = 0., maximumR = 200.; 
	double minimumZ = -200., maximumZ = 200.; 
	double minimumPhi = -3.15, maximumPhi = 3.15;
	double minimumDR = -1, maximumDR = 1;
	double minimumDZ = -1, maximumDZ = 1;
	double minimumRDPhi = -1, maximumRDPhi = 1;
	double minimumDX = -1, maximumDX = 1;
	double minimumDY = -1, maximumDY = 1;
	
	
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
	
	// ---------  project tree onto histograms ---------
	data->Project("hprof_drVr", "dr:r",Cut,"prof");
	data->Project("hprof_dzVr", "dz:r",Cut,"prof");
	data->Project("hprof_rdphiVr", "r*dphi:r",Cut,"prof");
	data->Project("hprof_dxVr", "dx:r",Cut,"prof");
	data->Project("hprof_dyVr", "dy:r",Cut,"prof");
	data->Project("hprof_drVz", "dr:z",Cut,"prof");
	data->Project("hprof_dzVz", "dz:z",Cut,"prof");
	data->Project("hprof_rdphiVz", "r*dphi:z",Cut,"prof");
	data->Project("hprof_dxVz", "dx:z",Cut,"prof");
	data->Project("hprof_dyVz", "dy:z",Cut,"prof");
	data->Project("hprof_drVphi", "dr:phi",Cut,"prof");
	data->Project("hprof_dzVphi", "dz:phi",Cut,"prof");
	data->Project("hprof_rdphiVphi", "r*dphi:phi",Cut,"prof");
	data->Project("hprof_dxVphi", "dx:phi",Cut,"prof");
	data->Project("hprof_dyVphi", "dy:phi",Cut,"prof");
	
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
	
	if (savePlot) cp->Print((_outputDir+"plot3x5Prof_"+plotName).c_str());
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
			if (flag == 0) max = hist->GetBinLowEdge(i) + 2.*hist->GetBinWidth(i);
			if (flag == 1) max = hist->GetBinLowEdge(i) + hist->GetBinWidth(i);
		}
	}
	for (int i = (nBins-1); i >= 0; i--){
		double binContent = hist->GetBinContent(i);
		if (binContent > 0) min = hist->GetBinLowEdge(i);
	}
	//std::cout << "max: " << max << ", min: " << min << std::endl;
}

void comparisonPlots::Write()
{
	output->Write();
}


