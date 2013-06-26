// call with
// root -l pileupJetAnalysis_mkplots.C'("analysis.root", "pileupJetAnalyzer", 0.5)'

#include <iostream>
#include <stdlib.h>

#include <RVersion.h>
#include <TH1.h>
#include <TFile.h>
#include <TList.h>
#include <TColor.h>
#include <TGraph.h>
#include <TString.h>
#include <TSystem.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TObject.h>
#include <TObjString.h>
#include <TIterator.h>
#include <TControlBar.h>
#include <TGWindow.h>
#include <TGMenu.h>
#include <TGClient.h>
#include <TGResourcePool.h>
#include <TVirtualX.h>
#ifndef __CINT__
#	include <GuiTypes.h>
#endif

static TFile *file = 0;
static TStyle *style = 0;
static const TGWindow *root = 0;

static const char *base = "abs(eta) < 2.4 && et > 15";

static Double_t Min(Double_t a, Double_t b)
{ return a < b ? a : b; }

static Double_t Max(Double_t a, Double_t b)
{ return a > b ? a : b; }

static void FindRange(TH1 *histo, Int_t &b1, Int_t &b2,
                      Double_t &x1, Double_t &x2)
{
	for(Int_t i = 1; i <= histo->GetNbinsX(); i++) {
		if (histo->GetBinContent(i) > 0.0) {
			if (i < b1)
				b1 = i;
			if (i > b2)
				b2 = i;

			Double_t low = histo->GetBinLowEdge(i);
			Double_t high = histo->GetBinLowEdge(i + 1);

			if (low < x1)
				x1 = low;
			if (high > x2)
				x2 = high;
		}
	}
}

class PadService {
    public:
	PadService(TString name, TString title, Int_t nPlots);
	~PadService();

	TVirtualPad *Next();

    private:
	TCanvas	*last;
	Int_t	index, count;

	TString	name, title;

	Int_t	width, height;
	Int_t	nPadsX, nPadsY;
};

PadService::PadService(TString name, TString title, Int_t nPlots) :
	last(0), index(0), count(0), name(name), title(title)
{
	switch (nPlots) {
	    case 1:
		nPadsX = 1; nPadsY = 1; width = 500; height = 1.00 * width;
		break;
	    case 2:
		nPadsX = 2; nPadsY = 1; width = 600; height = 0.55 * width;
		break;
	    case 3:
		nPadsX = 3; nPadsY = 1; width = 900; height = 0.40 * width;
		break;
	    case 4:
		nPadsX = 2; nPadsY = 2; width = 600; height = 1.00 * width;
		break;
	    default:
		nPadsX = 3; nPadsY = 2; width = 800; height = 0.55 * width;
		break;
	}
}

PadService::~PadService()
{
}

TVirtualPad *PadService::Next()
{
	if (!last) {
		last = new TCanvas(Form("%s_%d", (const char*)name,
		                        ++count),
		                   title, count * 50 + 200, count * 20,
		                   width, height);
		last->SetBorderSize(2);
		last->SetFrameFillColor(0);
		last->SetHighLightColor(0);
		last->Divide(nPadsX, nPadsY);
		index = 0;
		count++;
	}

	last->cd(++index);
	TVirtualPad *pad = last->GetPad(index);

	if (index == nPadsX * nPadsY)
		last = 0;

	return pad;
}

void pileupJetAnalysis_mkplots(TString fname, TString dir, double cut)
{
	TFile *f = TFile::Open(fname);
	TString name = "discriminator";

	TH1 *sig = new TH1F("sig", "sig", 4000, 0, 5);
	TH1 *bkg = new TH1F("bkg", "bkg", 4000, 0, 5);

	TTree *t = dynamic_cast<TTree*>(f->Get(dir + "/jets"));
	TString sigCut = Form("%s && mc >= %f", base, cut);
	TString bkgCut = Form("%s && mc < %f", base, cut);

	t->Draw("min(9.999, tag) >> sig", sigCut, "goff");
	t->Draw("min(9.999, tag) >> bkg", bkgCut, "goff");

	Int_t bin1 = sig->GetNbinsX();
	Int_t bin2 = 1;
	Double_t x1 = sig->GetXaxis()->GetXmax();
	Double_t x2 = sig->GetXaxis()->GetXmin();
	FindRange(bkg, bin1, bin2, x1, x2);
	FindRange(sig, bin1, bin2, x1, x2);

	PadService pads("discriminator", "discriminator & performance", 3);

	TVirtualPad *pad = pads.Next();

	TH1 *bkg_ = (TH1*)bkg->Clone(name + "_tmpO1");
	bkg_->Rebin(8);
	double bkgN = bkg_->Integral() / bkg_->Integral("width");
	bkg_->SetNormFactor(bkg_->Integral() / bkg_->Integral("width"));

	TH1 *sig_ = (TH1*)sig->Clone(name + "_tmpO2");
	sig_->Rebin(8);
	double sigN = sig_->Integral() / sig_->Integral("width");
	sig_->SetNormFactor(sig_->Integral() / sig_->Integral("width"));

	Double_t y = Max(bkg_->GetMaximum() / bkgN, sig_->GetMaximum() / sigN);
	bkg_->SetStats(0);
	bkg_->GetXaxis()->SetRangeUser(x1, x2);
	bkg_->GetYaxis()->SetRangeUser(0.0, y * 1.275);
	bkg_->SetTitle("discriminator");
	bkg_->SetXTitle(name);
	bkg_->SetFillColor(2);
	bkg_->SetLineColor(2);
	bkg_->SetLineWidth(2);
	bkg_->SetFillStyle(3554);
	bkg_->Draw();
	sig_->GetXaxis()->SetRangeUser(x1, x2);
	sig_->SetFillColor(0);
	sig_->SetLineColor(4);
	sig_->SetLineWidth(2);
	sig_->SetFillStyle(1);
	sig_->Draw("same");

	TLegend *leg = new TLegend(0.6 - pad->GetRightMargin(),
	                           1.0 - pad->GetTopMargin() - 0.15,
	                           1.0 - pad->GetRightMargin(),
	                           1.0 - pad->GetTopMargin());
	leg->SetFillStyle(1);
	leg->AddEntry(sig_, "Signal", "F");
	leg->AddEntry(bkg_, "Background", "F");
	leg->SetBorderSize(1);
	leg->SetMargin(0.3);
	leg->Draw("same");

	pad->RedrawAxis();

	TVirtualPad *pad = pads.Next();

	unsigned int n = bin2 - bin1 + 1;
	Double_t incr = (x2 - x1) * 0.5 / n;
	x1 -= incr;
	x2 += incr;
	TH1F *sum[4];
	for(unsigned int i = 0; i < 4; i++) {
		sum[i] = new TH1F(Form("%s_tmpO%d", (const char*)name, 3 + i),
		                  "cut efficiencies",
		                  n + 1, x1 - incr, x2 + incr);
		sum[i]->SetBit(kCanDelete);
	}

	Double_t sumBkg = 0.0, sumSig = 0.0;
	for(unsigned int i = 0; i < n; i++) {
		sum[0]->SetBinContent(i + 1, sumBkg);
		sum[1]->SetBinContent(i + 1, sumSig);
		sumBkg += bkg->GetBinContent(i + bin1);
		sumSig += sig->GetBinContent(i + bin1);
	}
	sum[0]->SetBinContent(n + 1, sumBkg);
	sum[1]->SetBinContent(n + 1, sumSig);
	for(unsigned int i = 0; i <= n; i++) {
		sum[2]->SetBinContent(i + 1, sumBkg);
		sum[3]->SetBinContent(i + 1, sumSig);
	}
	for(unsigned int i = 0; i <= n; i++) {
		sum[0]->SetBinContent(
			i + 1, sumBkg - sum[0]->GetBinContent(i + 1));
		sum[1]->SetBinContent(
			i + 1, sumSig - sum[1]->GetBinContent(i + 1));
	}

	for(unsigned int i = 0; i < 2; i++)
		sum[i]->Divide(sum[i], sum[2 + i], 1, 1, "b");

	sum[2]->Delete();
	sum[3]->Delete();

	pad->SetGridy();
	pad->SetTicky();
	sum[0]->SetLineColor(2);
	sum[0]->SetLineWidth(2);
	sum[0]->GetYaxis()->SetRangeUser(0.0, 1.005);
	sum[0]->SetXTitle(name);
	sum[0]->GetYaxis()->SetTitleOffset(1.25);
	sum[0]->SetYTitle("efficiency");
	sum[0]->SetStats(0);
	sum[0]->Draw("C");
	sum[1]->SetLineColor(4);
	sum[1]->SetLineWidth(2);
	sum[1]->Draw("C same");

	pad->RedrawAxis();

	gStyle->SetFillColor(kWhite);
	pad->SetFillColor(kWhite);
	TVirtualPad *pad = pads.Next();

	TH1F *tmp = new TH1F(name + "_tmpO6",
	                     "efficiency vs. purity", n, 0, 1);
	tmp->SetBit(kCanDelete);

	pad->SetGridx();
	pad->SetGridy();
	pad->SetTickx();
	pad->SetTicky();
	tmp->GetXaxis()->SetRangeUser(0.0, 1.005);
	tmp->GetYaxis()->SetRangeUser(0.0, 1.005);
	tmp->SetXTitle("efficiency");
	tmp->SetYTitle("purity");
	tmp->SetStats(0);
	tmp->SetLineWidth(2);
	tmp->SetFillColor(kWhite);
	tmp->Draw();

	Float_t *values[2];
	values[0] = new Float_t[n + 1];
	values[1] = new Float_t[n + 1];
	for(unsigned int i = 0; i <= n; i++) {
		values[0][i] = sum[1]->GetBinContent(i + 1);
		values[1][i] = 1.0 - sum[0]->GetBinContent(i + 1);
	}
	TGraph *graph = new TGraph(n + 1, values[0], values[1]);
	delete[] values[0];
	delete[] values[1];

	graph->SetName(name + "_tmpO7");
	graph->SetBit(kCanDelete);
	graph->SetLineColor(4);
	graph->SetLineWidth(2);
	graph->Draw("C");

	pad->RedrawAxis();
}
