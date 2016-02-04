#include<iostream>
#include<TStyle.h>
#include<TFile.h>
#include<TDirectory.h>
#include<TTree.h>
#include<TCanvas.h>
#include<TPaveText.h>
#include<TH1F.h>
#include<TLegend.h>
#include <fstream>

void Style(){
	
	TStyle *tdrStyle = new TStyle("tdrStyle","Style for P-TDR");
	
	// For the canvas:
	tdrStyle->SetCanvasBorderMode(0);
	tdrStyle->SetCanvasColor(kWhite);
	tdrStyle->SetCanvasDefH(600); //Height of canvas
	tdrStyle->SetCanvasDefW(1100); //Width of canvas
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
	
	tdrStyle->SetHistLineWidth(1.5);
	//For the fit/function:
	tdrStyle->SetOptFit(1);
	tdrStyle->SetStatColor(kWhite);
	tdrStyle->SetStatFont(42);
	tdrStyle->SetStatFontSize(0.025);
	tdrStyle->SetOptStat(000000);
	tdrStyle->SetStatColor(kWhite);
	
	// Margins:
	tdrStyle->SetPadTopMargin(0.05);
	tdrStyle->SetPadBottomMargin(0.13);
	tdrStyle->SetPadLeftMargin(0.10);
	tdrStyle->SetPadRightMargin(0.02);
	
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
	tdrStyle->SetTitleSize(0.04, "XYZ");
	tdrStyle->SetTitleXOffset(1.1);
	tdrStyle->SetTitleYOffset(1.1);
	
	// For the axis labels:
	
	tdrStyle->SetLabelColor(1, "XYZ");
	tdrStyle->SetLabelFont(42, "XYZ");
	tdrStyle->SetLabelOffset(0.007, "XYZ");
	tdrStyle->SetLabelSize(0.04, "XYZ");
	
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
	
	tdrStyle->cd();
	
	gStyle->SetFillColor(-1);
}


void plotDJR(const TString & infile,const TString & outfile){
	Style();
	ifstream file(infile);
	int nbin=60;
	float xmax=3.2;
	TPaveText *text[4];
	TH1F *DJR[4][5],*sum[4];
	TFile *f = new TFile(outfile,"RECREATE");
        TTree *events = new TTree("QPar","Qpar");
	float Qpar1, Qpar2, Qpar3, Qpar4;
	int Npart;
        events->Branch("Qpar1",&Qpar1,"Qpar1/F");
	events->Branch("Qpar2",&Qpar2,"Qpar2/F");
	events->Branch("Qpar3",&Qpar3,"Qpar3/F");
	events->Branch("Qpar4",&Qpar4,"Qpar4/F");
	events->Branch("Npart",&Npart,"Npart/I");
	for(int k=0;k<4;k++){
		text[k]=new TPaveText(0.4,0.90,0.7,0.95,"NDC");
		text[k]->AddText(Form("DJR(%d#rightarrow%d)",k+1,k));
		text[k]->SetTextSize(0.05);
		if(k>=2)text[k]->SetTextSize(0.04);
		text[k]->SetBorderSize(0);
		for(int m=0;m<5;m++){
			DJR[k][m]=new TH1F(Form("djr%d%d",k,m),";Q;Normalized scale;",nbin,0.01,xmax);
			DJR[k][m]->SetLineColor(m+1);
			DJR[k][m]->SetLineStyle(2);	
		}	
		sum[k]=new TH1F(Form("sum%d",k),";Log_{10}(Merging scale);Normalized scale;",nbin,0.01,xmax);	
		sum[k]->SetLineColor(38);
		sum[k]->GetYaxis()->SetTitleSize(0.05-k*0.005);
		sum[k]->GetYaxis()->SetLabelSize(0.05-k*0.005);
		sum[k]->GetYaxis()->SetTitleOffset(1+k*0.1);
		sum[k]->GetYaxis()->SetLabelOffset(0.01);
	}
	int npmax=0;
	while(1)
	{
		float djr_val[4];
		for(int k=0;k<4;k++){djr_val[k]=-100000;}
		Qpar1=-10;
		Qpar2=-10;
		Qpar3=-10;
		Qpar4=-10;
		float np;
		file >> np >> djr_val[0]>>djr_val[1]>>djr_val[2]>>djr_val[3];
		Npart=np;
		for(int k=0;k<4;k++){
			if(djr_val[k]>0){
				DJR[k][(int)np]->Fill(log10(djr_val[k]));
				if(k==0)Qpar1=log10(djr_val[k]);
				else if(k==1)Qpar2=log10(djr_val[k]);
				else if(k==2)Qpar3=log10(djr_val[k]);
				else if(k==3)Qpar4=log10(djr_val[k]);
			}
		}
		events->Fill();
		if(np>npmax)npmax=np;
		if(!file.good()) break;
	}
	for(int k=0;k<4;k++){
		for(int n=0;n<=npmax;n++){
			sum[k]->Add(DJR[k][n]);
		}
	}
	events->Write();
	
	TCanvas *c1= new TCanvas("c1", "c1", 800, 600);
	c1->SetLeftMargin(0.0);
	c1->SetTopMargin(0.00);
	c1->SetRightMargin(0.00);
	c1->SetBottomMargin(0.0);
	
	TLegend *legend=new TLegend(0.75,0.95-npmax*0.06,0.95,0.95);
	legend->SetTextSize(0.050);
	legend->SetBorderSize(0);
	legend->SetTextFont(62);
	legend->SetLineColor(0);
	legend->SetLineStyle(1);
	legend->SetLineWidth(1);
	legend->SetFillColor(0);
	legend->SetFillStyle(1001);
	
	for(int n=0;n<=npmax;n++){
		legend->AddEntry(DJR[0][n],Form("%d partons",n));
	}
	
	TPad *pad[4];
	pad[0]  =new TPad("pad0","pad",0,0.54,0.54,1.0);
	pad[1]  =new TPad("pad1","pad",0.54,0.54,1.0,1.0);
	pad[2]  =new TPad("pad2","pad",0,0,0.54,0.54);
	pad[3]  =new TPad("pad3","pad",0.54,0.0,1.0,0.54);
	for(int k=0;k<4;k++){
		if(k==0 || k==2){
				pad[k]->SetLeftMargin(0.15);
				pad[k]->SetRightMargin(0.0);
		}
		if(k==1 || k==3){
			pad[k]->SetRightMargin(0.01);
			pad[k]->SetLeftMargin(0.0);
		}
		if(k==0 || k==1){
			pad[k]->SetTopMargin(0.01);
			pad[k]->SetBottomMargin(0.0);
		}
		if(k==2 || k==3){
			pad[k]->SetTopMargin(0.0);
			pad[k]->SetBottomMargin(0.15);

		}
		pad[k]->Draw();
	}
	float Ymax;
	float scalingFactor=1.0/sum[0]->Integral();
	for(int k=0;k<4;k++){
		
		pad[k]->cd();
		 for(int m=0;m<=xmax;m++){
                        DJR[k][m]->Scale(scalingFactor);
                }

		sum[k]->Scale(scalingFactor);
		sum[k]->SetLineWidth(2);
		sum[k]->Draw("hist");
		if(k==0)Ymax=sum[k]->GetBinContent(sum[k]->GetMaximumBin())*2.5;
		sum[k]->SetMaximum(Ymax);
		sum[k]->SetMinimum(0.0007*Ymax);
		for(int m=0;m<=xmax;m++){
			DJR[k][m]->Draw("histsame");
		}
		text[k]->Draw();
		if(k==0)legend->Draw();
		gPad->SetLogy(1);	
		
	}
	
	c1->Print("DJR.pdf"); 

}

