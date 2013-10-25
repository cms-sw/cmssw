#include <vector>
#include <fstream>
#include <sstream>
#include "TTree.h"
#include "TFile.h"
#include "TMath.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TGraphAsymmErrors.h"
#include "TPaveStats.h"
#include "TStyle.h"
#include "TEfficiency.h"
#include "TProfile.h"

#include "TStyle.h"

void setTDRStyle() {
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
  tdrStyle->SetHistLineWidth(1);
  // tdrStyle->SetLegoInnerR(Float_t rad = 0.5);
  // tdrStyle->SetNumberContours(Int_t number = 20);

  tdrStyle->SetEndErrorSize(2);
//  tdrStyle->SetErrorMarker(20);
  tdrStyle->SetErrorX(0.5);
  
  tdrStyle->SetMarkerStyle(20);

//For the fit/function:
  tdrStyle->SetOptFit(1);
  tdrStyle->SetFitFormat("5.4g");
  tdrStyle->SetFuncColor(2);
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
  tdrStyle->SetPadLeftMargin(0.16);
  tdrStyle->SetPadRightMargin(0.02);

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
  tdrStyle->SetTitleYOffset(1.25);
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
  tdrStyle->SetPaperSize(20.,20.);
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

struct MyHisto{

  TH2F * PtResVsEta_NoGEM;
  TH2F * InvPtResVsEta_NoGEM;
  TH2F * PtResVsEta_GEM;
  TH2F * InvPtResVsEta_GEM;

  TProfile * prof_PtResVsEta_NoGEM;
  TProfile * prof_InvPtResVsEta_NoGEM;
  TProfile * prof_PtResVsEta_GEM;
  TProfile * prof_InvPtResVsEta_GEM;

  TEfficiency * pEffEta_GEM;
  TEfficiency * pEffSimEta_GEM;
  TEfficiency * pEffPhiPlus_GEM;
  TEfficiency * pEffPhiMinus_GEM;
  TEfficiency * pEffSimPhiPlus_GEM;
  TEfficiency * pEffSimPhiMinus_GEM;

};

struct MyPar{

	double mean;
	double rms;
	double sigma;
	double sigmaErr;
	double sigmaFR;
	double sigmaErrFR;
	double DeltaSigma;
	TH1D * histo;

};

MyPar extractSigma(TH1D * hist, std::string postFix = "none"){

	gStyle->SetOptStat(000002210);

	std::string name = "fit_";

   	MyPar obj;

	obj.mean = hist->GetMean();
	obj.rms = hist->GetRMS();
  	TCanvas * canvasTMP = new TCanvas("canvasTMP","canvas",700,700);

	std::string label;
	if(postFix.find("InvRes") != std::string::npos) label = "(q^{Reco}/p_{T}^{Reco} - q^{Sim}/p_{T}^{Sim}) / q^{Sim}/p_{T}^{Sim}";
	else label = "(p_{T}^{Reco} - p_{T}^{Sim}) / p_{T}^{Sim}";

	hist->GetXaxis()->SetTitle(label.c_str());
	hist->Draw();
        TF1 *myfitFR = new TF1("myfitFR","gaus", -1, +1);
        hist->Fit("myfitFR");
        TF1 *myfit = new TF1("myfit","gaus", -(obj.mean+2*obj.rms), obj.mean+2*obj.rms);
        hist->Fit("myfit", "R");

	obj.sigma = myfit->GetParameter(2);
	obj.sigmaErr = myfit->GetParError(2);
	obj.sigmaFR = myfitFR->GetParameter(2);
	obj.sigmaErrFR = myfitFR->GetParError(2);
	double delta = abs(myfit->GetParameter(2) - myfitFR->GetParameter(2));
	obj.DeltaSigma = sqrt(delta*delta + obj.sigmaErr*obj.sigmaErr);
	obj.histo = hist;

	//hist->GetXaxis()->SetRangeUser(-0.5, +0.5);
	//hist->Draw();
	canvasTMP->SaveAs((name + postFix + ".png").c_str());
	return obj;

}

MyHisto extractHistos(std::string name){

  TFile * f1 = TFile::Open(name.c_str());
  f1->cd();
  MyHisto temp;

  std::cout<<("aperto "+name).c_str()<<std::endl;

  temp.PtResVsEta_NoGEM = (TH2F*)gDirectory->Get("PtResVsEta_NoGEM");
  temp.InvPtResVsEta_NoGEM = (TH2F*)gDirectory->Get("InvPtResVsEta_NoGEM");
  temp.PtResVsEta_GEM = (TH2F*)gDirectory->Get("PtResVsEta_GEM");
  temp.InvPtResVsEta_GEM = (TH2F*)gDirectory->Get("InvPtResVsEta_GEM");

  temp.prof_PtResVsEta_NoGEM = (TProfile*)gDirectory->Get("prof1_2_NoGEM");
  temp.prof_InvPtResVsEta_NoGEM = (TProfile*)gDirectory->Get("prof2_2_NoGEM");
  temp.prof_PtResVsEta_GEM = (TProfile*)gDirectory->Get("prof1_2_GEM");
  temp.prof_InvPtResVsEta_GEM = (TProfile*)gDirectory->Get("prof2_2_GEM");

  temp.pEffEta_GEM = (TEfficiency*)gDirectory->Get("pEffEta_GEM");
  temp.pEffSimEta_GEM = (TEfficiency*)gDirectory->Get("pEffSimEta_GEM");
  temp.pEffPhiPlus_GEM = (TEfficiency*)gDirectory->Get("pEffPhiPlus_GEM");
  temp.pEffPhiMinus_GEM = (TEfficiency*)gDirectory->Get("pEffPhiMinus_GEM");
  temp.pEffSimPhiPlus_GEM = (TEfficiency*)gDirectory->Get("pEffSimPhiPlus_GEM");
  temp.pEffSimPhiMinus_GEM = (TEfficiency*)gDirectory->Get("pEffSimPhiMinus_GEM");

  return temp;

}

std::string bin[] = {"5", "10", "50", "100", "200", "500", "1000"};

std::vector<TH1D*> projectAndFit(TH2F * Histo2D, std::string postFix = "none"){

	std::vector<TH1D*> vec;

	TH1D * h1 = new TH1D((postFix + "RMS").c_str(),(postFix + "RMS").c_str(),100,-2.5,+2.5);
	TH1D * h2 = new TH1D((postFix + "Sigma").c_str(),(postFix + "Sigma").c_str(),100,-2.5,+2.5);

	for(int i = 1; i <= Histo2D->GetNbinsX(); i++){

		TH1D * proj1 = Histo2D->ProjectionY("proj1",i,i);
		if(proj1->GetEntries() == 0) continue;
		std::stringstream ss;
		ss<<postFix.c_str()<<i;	
		std::string pf = ss.str();
		MyPar temp = extractSigma(proj1,pf.c_str());

		h1->SetBinContent(i,temp.rms);	
		h1->SetBinError(i,0);

		h2->SetBinContent(i,temp.sigma);	
		h2->SetBinError(i,temp.DeltaSigma);

	}

	vec.push_back(h1);
	vec.push_back(h2);
	return vec;

}

TH1F * makeRatio(TH1F * plot1, TH1F * plot2){

	TH1F * plotTMP = (TH1F*)plot2->Clone();
	plotTMP->Divide(plot1);
	return plotTMP;

}

TH1D * makeRatio2(TH1D * plot1, TH1D * plot2){

	TH1D * plotTMP = (TH1D*)plot2->Clone();
	plotTMP->SetStats(kFALSE);
	plotTMP->SetLineColor(1);
  	plotTMP->SetMaximum(1.1);
  	plotTMP->SetMinimum(0.6);
	plotTMP->Divide(plot1);
	return plotTMP;

}

TH1D * makeRatio3(TEfficiency * plot1, TEfficiency * plot2){

 	TH1D * plotTMP1 = new TH1D("plotTMP1","plotTMP1",221,-2.5,1102.5);
 	TH1D * plotTMP2 = new TH1D("plotTMP2","plotTMP2",221,-2.5,1102.5);

	for(int i = 1; i <= plotTMP1->GetSize(); i++){

		double eff1 = plot1->GetEfficiency(i);
		double eff2 = plot2->GetEfficiency(i);

		//std::cout<<"eff1: "<<eff1<<" eff2: "<<eff2<<std::endl;

		double err1 = (plot1->GetEfficiencyErrorLow(i) + plot1->GetEfficiencyErrorUp(i))/2;
		double err2 = (plot2->GetEfficiencyErrorLow(i) + plot2->GetEfficiencyErrorUp(i))/2;

		plotTMP1->SetBinContent(i,eff1);
		plotTMP1->SetBinError(i,err1);
		plotTMP2->SetBinContent(i,eff2);
		plotTMP2->SetBinError(i,err2);

  	}

  	plotTMP2->SetMaximum(4);
  	plotTMP2->SetMinimum(0.01);
	plotTMP2->Divide(plotTMP1);
	return plotTMP2;

}
// 0  1  2   3   4   5    6
// 5 10 50 100 200 500 1000
int colorVec[] = {1, 3, 2, 4, 5, 7, 6};

void superimposeHistos(std::vector<MyHisto> allHistos){

  	TCanvas * canvas = new TCanvas("canvas","canvas",700,700);

  	TLegend *leg5 = new TLegend(0.50,0.40,0.70,0.80);
  	leg5->SetFillColor(kWhite);
  	leg5->SetLineColor(kWhite);
  	//leg5->SetHeader("Standard Reco");

  	allHistos[0].pEffSimEta_GEM->Draw("AP");
  	gPad->Update();
    	allHistos[0].pEffSimEta_GEM->GetPaintedGraph()->SetMaximum(1);
	gPad->Update();
  	allHistos[0].pEffSimEta_GEM->GetPaintedGraph()->GetXaxis()->SetTitle("#eta^{Sim}");
  	gPad->Update();
  	allHistos[0].pEffSimEta_GEM->GetPaintedGraph()->GetYaxis()->SetTitle("#varepsilon");
	allHistos[0].pEffSimEta_GEM->SetMarkerStyle(20);
	allHistos[0].pEffSimEta_GEM->Draw("AP");
  	leg5->AddEntry(allHistos[0].pEffSimEta_GEM,"p_{T} = 5 GeV/c","lp");

	for(int i=1; i<(int)allHistos.size(); i++){

		if(!(i == 0 || i == 2 || i == 6)) continue;
		allHistos[i].pEffSimEta_GEM->SetMarkerStyle(20);
		std::cout<<"color "<<i<<" "<<colorVec[i]<<std::endl;
		allHistos[i].pEffSimEta_GEM->SetLineColor(colorVec[i]);
		allHistos[i].pEffSimEta_GEM->SetMarkerColor(colorVec[i]);
		allHistos[i].pEffSimEta_GEM->Draw("SAME");
		leg5->AddEntry(allHistos[i].pEffSimEta_GEM,("p_{T} = " + bin[i] + " GeV/c").c_str(),"lp");

	}
	leg5->Draw();

	canvas->SaveAs("plot1.png");

	///////////////////////////////////////////////////////////////////////////////////////////////////////////

  	allHistos[0].pEffEta_GEM->Draw("AP");
  	gPad->Update();
    	allHistos[0].pEffEta_GEM->GetPaintedGraph()->SetMaximum(1);
	gPad->Update();
  	allHistos[0].pEffEta_GEM->GetPaintedGraph()->GetXaxis()->SetTitle("#eta^{Reco}");
  	gPad->Update();
  	allHistos[0].pEffEta_GEM->GetPaintedGraph()->GetYaxis()->SetTitle("#varepsilon");
	allHistos[0].pEffEta_GEM->SetMarkerStyle(20);
	allHistos[0].pEffEta_GEM->Draw("AP");

	for(int i=1; i<(int)allHistos.size(); i++){

		if(!(i == 0 || i == 2 || i == 6)) continue;
		allHistos[i].pEffEta_GEM->SetMarkerStyle(20);
		allHistos[i].pEffEta_GEM->SetLineColor(colorVec[i]);
		allHistos[i].pEffEta_GEM->SetMarkerColor(colorVec[i]);
		allHistos[i].pEffEta_GEM->Draw("SAME");

	}
	leg5->Draw();

	canvas->SaveAs("plot2.png");

	///////////////////////////////////////////////////////////////////////////////////////////////////////////

	//TCanvas * canvas = new TCanvas("canvas","canvas",700,700);

  	TLegend *leg6 = new TLegend(0.80,0.15,1.00,0.55);
  	leg6->SetFillColor(kWhite);
  	leg6->SetLineColor(kWhite);
  	//leg6->SetHeader("Standard Reco");

  	allHistos[0].pEffPhiPlus_GEM->Draw("AP");
  	gPad->Update();
    	allHistos[0].pEffPhiPlus_GEM->GetPaintedGraph()->SetMaximum(1);
	gPad->Update();
  	allHistos[0].pEffPhiPlus_GEM->GetPaintedGraph()->GetXaxis()->SetTitle("#phi^{Reco}");
  	gPad->Update();
  	allHistos[0].pEffPhiPlus_GEM->GetPaintedGraph()->GetYaxis()->SetTitle("#varepsilon (#eta > 0)");
	allHistos[0].pEffPhiPlus_GEM->SetMarkerStyle(20);
	allHistos[0].pEffPhiPlus_GEM->Draw("AP");
  	leg6->AddEntry(allHistos[0].pEffPhiPlus_GEM,"p_{T} = 5 GeV/c","lp");

	for(int i=1; i<(int)allHistos.size(); i++){

		if(!(i == 0 || i == 2 || i == 6)) continue;
		allHistos[i].pEffPhiPlus_GEM->SetMarkerStyle(20);
		allHistos[i].pEffPhiPlus_GEM->SetLineColor(colorVec[i]);
		allHistos[i].pEffPhiPlus_GEM->SetMarkerColor(colorVec[i]);
		allHistos[i].pEffPhiPlus_GEM->Draw("SAME");
		leg6->AddEntry(allHistos[i].pEffPhiPlus_GEM,("p_{T} = " + bin[i] + " GeV/c").c_str(),"lp");

	}
	leg6->Draw();

	canvas->SaveAs("plot3.png");

	///////////////////////////////////////////////////////////////////////////////////////////////////////////

  	allHistos[0].pEffPhiMinus_GEM->Draw("AP");
  	gPad->Update();
    	allHistos[0].pEffPhiMinus_GEM->GetPaintedGraph()->SetMaximum(1);
	gPad->Update();
  	allHistos[0].pEffPhiMinus_GEM->GetPaintedGraph()->GetXaxis()->SetTitle("#phi^{Reco}");
  	gPad->Update();
  	allHistos[0].pEffPhiMinus_GEM->GetPaintedGraph()->GetYaxis()->SetTitle("#varepsilon (#eta < 0)");
	allHistos[0].pEffPhiMinus_GEM->SetMarkerStyle(20);
	allHistos[0].pEffPhiMinus_GEM->Draw("AP");

	for(int i=1; i<(int)allHistos.size(); i++){

		if(!(i == 0 || i == 2 || i == 6)) continue;
		allHistos[i].pEffPhiMinus_GEM->SetMarkerStyle(20);
		allHistos[i].pEffPhiMinus_GEM->SetLineColor(colorVec[i]);
		allHistos[i].pEffPhiMinus_GEM->SetMarkerColor(colorVec[i]);
		allHistos[i].pEffPhiMinus_GEM->Draw("SAME");

	}
	leg6->Draw();

	canvas->SaveAs("plot4.png");

	///////////////////////////////////////////////////////////////////////////////////////////////////////////

  	allHistos[0].pEffSimPhiPlus_GEM->Draw("AP");
  	gPad->Update();
    	allHistos[0].pEffSimPhiPlus_GEM->GetPaintedGraph()->SetMaximum(1);
	gPad->Update();
  	allHistos[0].pEffSimPhiPlus_GEM->GetPaintedGraph()->GetXaxis()->SetTitle("#phi^{Sim}");
  	gPad->Update();
  	allHistos[0].pEffSimPhiPlus_GEM->GetPaintedGraph()->GetYaxis()->SetTitle("#varepsilon (#eta > 0)");
	allHistos[0].pEffSimPhiPlus_GEM->SetMarkerStyle(20);
	allHistos[0].pEffSimPhiPlus_GEM->Draw("AP");

	for(int i=1; i<(int)allHistos.size(); i++){

		if(!(i == 0 || i == 2 || i == 6)) continue;
		allHistos[i].pEffSimPhiPlus_GEM->SetMarkerStyle(20);
		allHistos[i].pEffSimPhiPlus_GEM->SetLineColor(colorVec[i]);
		allHistos[i].pEffSimPhiPlus_GEM->SetMarkerColor(colorVec[i]);
		allHistos[i].pEffSimPhiPlus_GEM->Draw("SAME");

	}
	leg6->Draw();

	canvas->SaveAs("plot6.png");

	///////////////////////////////////////////////////////////////////////////////////////////////////////////

  	allHistos[0].pEffSimPhiMinus_GEM->Draw("AP");
  	gPad->Update();
    	allHistos[0].pEffSimPhiMinus_GEM->GetPaintedGraph()->SetMaximum(1);
	gPad->Update();
  	allHistos[0].pEffSimPhiMinus_GEM->GetPaintedGraph()->GetXaxis()->SetTitle("#phi^{Sim}");
  	gPad->Update();
  	allHistos[0].pEffSimPhiMinus_GEM->GetPaintedGraph()->GetYaxis()->SetTitle("#varepsilon (#eta < 0)");
	allHistos[0].pEffSimPhiMinus_GEM->SetMarkerStyle(20);
	allHistos[0].pEffSimPhiMinus_GEM->Draw("AP");

	for(int i=1; i<(int)allHistos.size(); i++){

		if(!(i == 0 || i == 2 || i == 6)) continue;
		allHistos[i].pEffSimPhiMinus_GEM->SetMarkerStyle(20);
		allHistos[i].pEffSimPhiMinus_GEM->SetLineColor(colorVec[i]);
		allHistos[i].pEffSimPhiMinus_GEM->SetMarkerColor(colorVec[i]);
		allHistos[i].pEffSimPhiMinus_GEM->Draw("SAME");

	}
	leg6->Draw();

	canvas->SaveAs("plot7.png");

	///////////////////////////////////////////////////////////////////////////////////////////////////////////

	/*for(int i=0; i<(int)allHistos.size(); i++){

	  	TLegend *leg2 = new TLegend(0.40,0.50,0.70,0.70);
  		leg2->SetFillColor(0);
  		leg2->SetLineColor(1);
		leg2->SetHeader(("p_{T} = " + bin[i] + " GeV/c").c_str());

		allHistos[i].prof_PtResVsEta_NoGEM->SetStats(kFALSE);
		allHistos[i].prof_PtResVsEta_NoGEM->SetMaximum(+0.025);
		allHistos[i].prof_PtResVsEta_NoGEM->SetMinimum(-0.025);
		allHistos[i].prof_PtResVsEta_NoGEM->SetLineColor(9);
  		allHistos[i].prof_PtResVsEta_NoGEM->GetXaxis()->SetTitle("#eta^{Sim}");
  		allHistos[i].prof_PtResVsEta_NoGEM->GetYaxis()->SetTitle("< (p_{T}^{Reco} - p_{T}^{Sim}) / p_{T}^{Sim} >");
		allHistos[i].prof_PtResVsEta_NoGEM->Draw("E1P");
		allHistos[i].prof_PtResVsEta_GEM->SetLineColor(2);
		allHistos[i].prof_PtResVsEta_GEM->Draw("E1PSAME");

		leg2->AddEntry(allHistos[i].prof_PtResVsEta_NoGEM, "Standard Reco", "lp");
  		leg2->AddEntry(allHistos[i].prof_PtResVsEta_GEM, "GEMsReco+GEMRecHit", "lp");
		leg2->Draw();
		canvas->SaveAs(("plot_Res_" + bin[i] + ".png").c_str());

	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////

	for(int i=0; i<(int)allHistos.size(); i++){

	  	TLegend *leg2 = new TLegend(0.40,0.50,0.70,0.70);
  		leg2->SetFillColor(0);
  		leg2->SetLineColor(1);
		leg2->SetHeader(("p_{T} = " + bin[i] + " GeV/c").c_str());

		allHistos[i].prof_InvPtResVsEta_NoGEM->SetStats(kFALSE);
		allHistos[i].prof_InvPtResVsEta_NoGEM->SetMaximum(+0.03);
		allHistos[i].prof_InvPtResVsEta_NoGEM->SetMinimum(-0.03);
		allHistos[i].prof_InvPtResVsEta_NoGEM->SetLineColor(9);
  		allHistos[i].prof_InvPtResVsEta_NoGEM->GetXaxis()->SetTitle("#eta^{Sim}");
  		allHistos[i].prof_InvPtResVsEta_NoGEM->GetYaxis()->SetTitle("< (p_{T}^{Reco} - p_{T}^{Sim}) / p_{T}^{Sim} >");
		allHistos[i].prof_InvPtResVsEta_NoGEM->Draw("E1P");
		allHistos[i].prof_InvPtResVsEta_GEM->SetLineColor(2);
		allHistos[i].prof_InvPtResVsEta_GEM->Draw("E1PSAME");

		leg2->AddEntry(allHistos[i].prof_InvPtResVsEta_NoGEM, "Standard Reco", "lp");
  		leg2->AddEntry(allHistos[i].prof_InvPtResVsEta_GEM, "GEMsReco+GEMRecHit", "lp");
		leg2->Draw();
		canvas->SaveAs(("plot_InvRes_" + bin[i] + ".png").c_str());

	}*/

	///////////////////////////////////////////////////////////////////////////////////////////////////////////

	for(int i = 0; i < (int)allHistos.size(); i++){

		std::vector<TH1D*> resGEM = projectAndFit(allHistos[i].PtResVsEta_GEM,("Res_GEM_" + bin[i] + "GeV_").c_str());
		std::vector<TH1D*> resNoGEM = projectAndFit(allHistos[i].PtResVsEta_NoGEM,("Res_NoGEM_" + bin[i] + "GeV_").c_str());

	  	TLegend *leg2 = new TLegend(0.40,0.50,0.70,0.70);
  		leg2->SetFillColor(0);
  		leg2->SetLineColor(1);
		leg2->SetHeader(("RMS: p_{T} = " + bin[i] + " GeV/c").c_str());

		TCanvas * canvas2 = new TCanvas("canvas2","canvas",700,700);
		canvas2->Divide(1,2);
		canvas2->cd(1);

		TH1D * hRMSNoGEM = (TH1D*)resNoGEM[0]->Clone();
		TH1D * hRMSGEM = (TH1D*)resGEM[0]->Clone();

		hRMSNoGEM->SetStats(kFALSE);
		hRMSNoGEM->SetMarkerColor(9);
		hRMSNoGEM->GetXaxis()->SetTitle("#eta^{Sim}");
		hRMSNoGEM->GetYaxis()->SetTitle("RMS_{Res}");
		hRMSNoGEM->SetMarkerStyle(20);
		hRMSNoGEM->Draw("P");
		hRMSGEM->SetMarkerColor(2);
		hRMSGEM->SetMarkerStyle(20);
		hRMSGEM->Draw("PSAME");
		leg2->AddEntry(hRMSNoGEM, "Standard Reco", "p");
  		leg2->AddEntry(hRMSGEM, "GEMsReco+GEMRecHit", "p");
		leg2->Draw();

		canvas2->cd(2);
  		TH1D * ratioComp1 = makeRatio2(hRMSNoGEM, hRMSGEM);
  		ratioComp1->SetMarkerColor(1);
  		ratioComp1->GetXaxis()->SetTitle("#eta^{Sim}");
  		ratioComp1->GetYaxis()->SetTitle("RMS_{Res}^{GEM} / RMS_{Res}^{NoGEM}");
  		ratioComp1->Draw("E1P");

		canvas2->SaveAs(("RMS_Res_" + bin[i] + "GeV.png").c_str());

		canvas2->Clear();
		canvas2->Update();
		canvas2->Divide(1,2);
		canvas2->cd(1);

		TH1D * hResNoGEM = (TH1D*)resNoGEM[1]->Clone();
		TH1D * hResGEM = (TH1D*)resGEM[1]->Clone();

	  	TLegend *leg3 = new TLegend(0.40,0.50,0.70,0.70);
  		leg3->SetFillColor(0);
  		leg3->SetLineColor(1);
		leg3->SetHeader(("#sigma: p_{T} = " + bin[i] + " GeV/c").c_str());

		hResNoGEM->SetLineColor(9);
		hResNoGEM->SetMarkerColor(9);
		hResNoGEM->SetStats(kFALSE);
		hResNoGEM->GetXaxis()->SetTitle("#eta^{Sim}");
		hResNoGEM->GetYaxis()->SetTitle("#sigma_{Res}");
		hResNoGEM->Draw("E1P");
		hResGEM->SetLineColor(2);
		hResGEM->SetMarkerColor(2);
		hResGEM->Draw("E1PSAME");
		leg3->AddEntry(hResNoGEM, "Standard Reco", "pl");
  		leg3->AddEntry(hResGEM, "GEMsReco+GEMRecHit", "pl");
		leg3->Draw();

		canvas2->cd(2);
  		TH1D * ratioComp2 = makeRatio2(hResNoGEM, hResGEM);
  		ratioComp2->SetMarkerColor(1);
  		ratioComp2->GetXaxis()->SetTitle("#eta^{Sim}");
  		ratioComp2->GetYaxis()->SetTitle("#sigma_{Res}^{GEM} / #sigma_{Res}^{NoGEM}");
  		ratioComp2->Draw("E1P");

		canvas2->SaveAs(("Res_" + bin[i] + "GeV.png").c_str());

		canvas2->Clear();
		canvas2->Update();

	}

	for(int i = 0; i < (int)allHistos.size(); i++){

		std::vector<TH1D*> invResGEM = projectAndFit(allHistos[i].InvPtResVsEta_GEM,("InvRes_GEM_" + bin[i] + "GeV_").c_str());
		std::vector<TH1D*> invResNoGEM = projectAndFit(allHistos[i].InvPtResVsEta_NoGEM,("InvRes_NoGEM_" + bin[i] + "GeV_").c_str());

	  	TLegend *leg2 = new TLegend(0.40,0.50,0.70,0.70);
  		leg2->SetFillColor(0);
  		leg2->SetLineColor(1);
		leg2->SetHeader(("RMS: p_{T} = " + bin[i] + " GeV/c").c_str());

		TCanvas * canvas2 = new TCanvas("canvas2","canvas",700,700);
		canvas2->Divide(1,2);
		canvas2->cd(1);

		TH1D * hRMSNoGEM = (TH1D*)invResNoGEM[0]->Clone();
		TH1D * hRMSGEM = (TH1D*)invResGEM[0]->Clone();

		hRMSNoGEM->SetStats(kFALSE);
		hRMSNoGEM->SetMarkerColor(9);
		hRMSNoGEM->GetXaxis()->SetTitle("#eta^{Sim}");
		hRMSNoGEM->GetYaxis()->SetTitle("RMS_{InvRes}");
		hRMSNoGEM->SetMarkerStyle(20);
		hRMSNoGEM->Draw("P");
		//hRMSNoGEM->SetMaximum(hRMSGEM->GetMaximum());
		hRMSGEM->SetMarkerColor(2);
		hRMSGEM->SetMarkerStyle(20);
		hRMSGEM->Draw("PSAME");
		leg2->AddEntry(hRMSNoGEM, "Standard Reco", "p");
  		leg2->AddEntry(hRMSGEM, "GEMsReco+GEMRecHit", "p");
		leg2->Draw();

		canvas2->cd(2);
  		TH1D * ratioComp1 = makeRatio2(hRMSNoGEM, hRMSGEM);
  		ratioComp1->SetMarkerColor(1);
  		ratioComp1->GetXaxis()->SetTitle("#eta^{Sim}");
  		ratioComp1->GetYaxis()->SetTitle("RMS_{InvRes}^{GEM} / RMS_{InvRes}^{NoGEM}");
  		ratioComp1->Draw("E1P");

		canvas2->SaveAs(("RMS_InvRes_" + bin[i] + "GeV.png").c_str());

		canvas2->Clear();
		canvas2->Update();
		canvas2->Divide(1,2);
		canvas2->cd(1);

		TH1D * hInvResNoGEM = (TH1D*)invResNoGEM[1]->Clone();
		TH1D * hInvResGEM = (TH1D*)invResGEM[1]->Clone();

	  	TLegend *leg3 = new TLegend(0.40,0.50,0.70,0.70);
  		leg3->SetFillColor(0);
  		leg3->SetLineColor(1);
		leg3->SetHeader(("#sigma: p_{T} = " + bin[i] + " GeV/c").c_str());

		hInvResNoGEM->SetLineColor(9);
		hInvResNoGEM->SetMarkerColor(9);
		//hInvResNoGEM->SetMarkerSize(0);
		hInvResNoGEM->SetStats(kFALSE);
		hInvResNoGEM->GetXaxis()->SetTitle("#eta^{Sim}");
		hInvResNoGEM->GetYaxis()->SetTitle("#sigma_{InvRes}");
		hInvResNoGEM->Draw("E1P");
		hInvResGEM->SetLineColor(2);
		hInvResGEM->SetMarkerColor(2);
		//hInvResGEM->SetMarkerSize(0);
		hInvResGEM->Draw("E1PSAME");
		leg3->AddEntry(hInvResNoGEM, "Standard Reco", "pl");
  		leg3->AddEntry(hInvResGEM, "GEMsReco+GEMRecHit", "pl");
		leg3->Draw();

		canvas2->cd(2);
  		TH1D * ratioComp2 = makeRatio2(hInvResNoGEM, hInvResGEM);
  		ratioComp2->SetMarkerColor(1);
  		ratioComp2->GetXaxis()->SetTitle("#eta^{Sim}");
  		ratioComp2->GetYaxis()->SetTitle("#sigma_{InvRes}^{GEM} / #sigma_{InvRes}^{NoGEM}");
  		ratioComp2->Draw("E1P");

		canvas2->SaveAs(("InvRes_" + bin[i] + "GeV.png").c_str());

	}

}

void makePlots(){

   	setTDRStyle();
	std::vector<MyHisto> allHistos;

	MyHisto struct5 = extractHistos("plots_5GeV.root");
	MyHisto struct10 = extractHistos("plots_10GeV.root");
	MyHisto struct50 = extractHistos("plots_50GeV.root");
	MyHisto struct100 = extractHistos("plots_100GeV.root");
	MyHisto struct200 = extractHistos("plots_200GeV.root");
	MyHisto struct500 = extractHistos("plots_500GeV.root");
	MyHisto struct1000 = extractHistos("plots_1000GeV.root");

	allHistos.push_back(struct5);
	allHistos.push_back(struct10);
	allHistos.push_back(struct50);
	allHistos.push_back(struct100);
	allHistos.push_back(struct200);
	allHistos.push_back(struct500);
	allHistos.push_back(struct1000);
	
	superimposeHistos(allHistos);

}
