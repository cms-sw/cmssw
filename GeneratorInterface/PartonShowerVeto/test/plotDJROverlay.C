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


void plotDJR(const TString& f1, const TString& f2 ) //,const TString & outfile)
{
	Style();

	// TString location =  "/uscms_data/d2/yarba_j/PS-Matching-Veto/";
	TString location = "./";
	
	TString infile1 = location + f1;
	TString infile2 = location + f2;
	
	std::cout << "infile1=" << infile1 << std::endl;
	std::cout << "infile2=" << infile2 << std::endl;
	
	ifstream file1(infile1);
	ifstream file2(infile2);
	
	std::cout << "file1 is good: " << file1.good() << std::endl; 
	std::cout << "file2 is good: " << file2.good() << std::endl; 

	int nbin=60;
	float xmax=3.2;
	TPaveText *text[4];

	TH1F* DJR1[4][5];
	TH1F* Sum1[4];
	TH1F* DJR2[4][5];
	TH1F* Sum2[4];

	float Qpar1, Qpar2, Qpar3, Qpar4;
	int Npart;

	for(int k=0;k<4;k++)
	{
		text[k]=new TPaveText(0.4,0.90,0.7,0.95,"NDC");
		text[k]->AddText(Form("DJR(%d#rightarrow%d)",k+1,k));
		text[k]->SetTextSize(0.05);
		if(k>=2)text[k]->SetTextSize(0.04);
		text[k]->SetBorderSize(0);
		for(int m=0;m<5;m++)
		{
			DJR1[k][m]=new TH1F(Form("Py6: djr%d%d",k,m),";Q;Normalized scale;",nbin,0.01,xmax);
			DJR1[k][m]->SetLineColor(m+1);
			DJR1[k][m]->SetLineStyle(2);	
			DJR2[k][m]=new TH1F(Form("Py8: djr%d%d",k,m),";Q;Normalized scale;",nbin,0.01,xmax);
			DJR2[k][m]->SetMarkerColor(m+1);
			DJR2[k][m]->SetMarkerStyle(23);	
		}	
		Sum1[k]=new TH1F(Form("Py6: sum%d",k),";Log_{10}(Merging scale);Normalized scale;",nbin,0.01,xmax);	
		Sum1[k]->SetLineColor(38);
		Sum1[k]->SetLineWidth(2);
		Sum1[k]->GetYaxis()->SetTitleSize(0.05-k*0.005);
		Sum1[k]->GetYaxis()->SetLabelSize(0.05-k*0.005);
		Sum1[k]->GetYaxis()->SetTitleOffset(1+k*0.1);
		Sum1[k]->GetYaxis()->SetLabelOffset(0.01);
		Sum2[k]=new TH1F(Form("Py8: sum%d",k),";Log_{10}(Merging scale);Normalized scale;",nbin,0.01,xmax);	
		//Sum2[k]->SetMarkerColor(38);
		//Sum2[k]->SetMarkerStyle(23);
		Sum2[k]->SetLineWidth(3);
		// Sum2[k]->SetLineStyle(2);
		Sum2[k]->SetLineColor(kMagenta);
		Sum2[k]->GetYaxis()->SetTitleSize(0.05-k*0.005);
		Sum2[k]->GetYaxis()->SetLabelSize(0.05-k*0.005);
		Sum2[k]->GetYaxis()->SetTitleOffset(1+k*0.1);
		Sum2[k]->GetYaxis()->SetLabelOffset(0.01);
	}

	int npmax=0;

// take in Pythia6 info

	float djr_val[4];
	float np;
	
	while(1)
	{
		for (int k=0;k<4;k++) 
		{
		   djr_val[k]=-100000;
		}
		Qpar1=-10;
		Qpar2=-10;
		Qpar3=-10;
		Qpar4=-10;

		file1 >> np >> djr_val[0]>>djr_val[1]>>djr_val[2]>>djr_val[3];

		Npart=np;

		for(int k=0;k<4;k++)
		{
			if(djr_val[k]>0)
			{
				DJR1[k][(int)np]->Fill(log10(djr_val[k]));
				if(k==0)Qpar1=log10(djr_val[k]);
				else if(k==1)Qpar2=log10(djr_val[k]);
				else if(k==2)Qpar3=log10(djr_val[k]);
				else if(k==3)Qpar4=log10(djr_val[k]);
			}
		}
		if(np>npmax)npmax=np;

		if(!file1.good()) break;

	}

	for(int k=0;k<4;k++)
	{
		for(int n=0;n<=npmax;n++)
		{
		   Sum1[k]->Add(DJR1[k][n]);
		}
	}

// now take Pythia8 info

	npmax = 0;
	
	while(1)
	{
		for (int k=0;k<4;k++) 
		{
		   djr_val[k]=-100000;
		}
		Qpar1=-10;
		Qpar2=-10;
		Qpar3=-10;
		Qpar4=-10;

		file2 >> np >> djr_val[0]>>djr_val[1]>>djr_val[2]>>djr_val[3];

		Npart=np;

		for(int k=0;k<4;k++)
		{
			if(djr_val[k]>0)
			{
				DJR2[k][(int)np]->Fill(log10(djr_val[k]));
				if(k==0)Qpar1=log10(djr_val[k]);
				else if(k==1)Qpar2=log10(djr_val[k]);
				else if(k==2)Qpar3=log10(djr_val[k]);
				else if(k==3)Qpar4=log10(djr_val[k]);
			}
		}
		if(np>npmax)npmax=np;

		if(!file2.good()) break;

	}


	for(int k=0;k<4;k++)
	{
	   for(int n=0;n<=npmax;n++)
	   {
	      Sum2[k]->Add(DJR2[k][n]);
	   }
	}
	
	TCanvas *c1= new TCanvas("c1", "c1", 1100, 900);
	c1->SetLeftMargin(0.0);
	c1->SetTopMargin(0.00);
	c1->SetRightMargin(0.00);
	c1->SetBottomMargin(0.0);
	
	TLegend* legend1=new TLegend(0.20,0.90-(npmax+1)*0.05,0.45,0.90);
	TLegend* legend2=new TLegend(0.65,0.90-(npmax+1)*0.05,0.90,0.90);
	legend1->SetTextSize(0.040);
	legend1->SetBorderSize(0);
	legend1->SetTextFont(62);
	legend1->SetLineColor(0);
	legend1->SetLineStyle(1);
	legend1->SetLineWidth(1);
	legend1->SetFillColor(0);
	legend1->SetFillStyle(1001);
	legend2->SetTextSize(0.040);
	legend2->SetBorderSize(0);
	legend2->SetTextFont(62);
	legend2->SetLineColor(0);
	legend2->SetLineStyle(1);
	legend2->SetLineWidth(1);
	legend2->SetFillColor(0);
	legend2->SetFillStyle(1001);
	
	for(int n=0;n<=npmax;n++)
	{
	   legend1->AddEntry(DJR1[0][n],Form("%d partons, Py6",n));
	   legend2->AddEntry(DJR2[0][n],Form("%d partons, Py8",n));
	}
	legend1->AddEntry(Sum1[0], "Sum, Py6");
	legend2->AddEntry(Sum2[0], "Sum, Py8");
	
	TPad *pad[4];
	pad[0]  =new TPad("pad0","pad",0,0.54,0.54,1.0);
	pad[1]  =new TPad("pad1","pad",0.54,0.54,1.0,1.0);
	pad[2]  =new TPad("pad2","pad",0,0,0.54,0.54);
	pad[3]  =new TPad("pad3","pad",0.54,0.0,1.0,0.54);

	for(int k=0;k<4;k++)
	{
		if(k==0 || k==2)
		{
		   pad[k]->SetLeftMargin(0.15);
		   pad[k]->SetRightMargin(0.0);
		}
		if(k==1 || k==3)
		{
		   pad[k]->SetRightMargin(0.01);
		   pad[k]->SetLeftMargin(0.0);
		}
		if(k==0 || k==1)
		{
		   pad[k]->SetTopMargin(0.01);
		   pad[k]->SetBottomMargin(0.0);
		}
		if(k==2 || k==3)
		{
		   pad[k]->SetTopMargin(0.0);
		   pad[k]->SetBottomMargin(0.15);
		}
		pad[k]->Draw();
	}

	float Ymax1, Ymax2;
	float scalingFactor1=1.0/Sum1[0]->Integral();
	float scalingFactor2=1.0/Sum2[0]->Integral();

	for(int k=0;k<4;k++){
		
		pad[k]->cd();
		for(int m=0;m<=xmax;m++)
		{
                   DJR1[k][m]->Scale(scalingFactor1);
                   DJR2[k][m]->Scale(scalingFactor2);
                }

		Sum1[k]->Scale(scalingFactor1);
		Sum2[k]->Scale(scalingFactor2);
		Sum1[k]->Draw("hist");
		Sum2[k]->Draw("same");

		if (k==0) Ymax1=Sum1[k]->GetBinContent(Sum1[k]->GetMaximumBin())*2.5;
		Sum1[k]->SetMaximum(20.*Ymax1);
		Sum1[k]->SetMinimum(0.0007*Ymax1);

		for(int m=0;m<=xmax;m++)
		{
		   DJR1[k][m]->Draw("histsame");
		   DJR2[k][m]->SetMarkerSize(0.7);
		   DJR2[k][m]->Draw("psame");
		}
		text[k]->Draw();
		if(k==0) 
		{
		   legend1->Draw();
		   legend2->Draw();
		}
		gPad->SetLogy(1);	
		
	}
	
	c1->Print("DJR.gif"); 

}

