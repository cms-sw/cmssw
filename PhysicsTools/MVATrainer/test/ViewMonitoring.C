#include <iostream>
#include <iomanip>
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

static Double_t Min(Double_t a, Double_t b)
{ return a < b ? a : b; }

static Double_t Max(Double_t a, Double_t b)
{ return a > b ? a : b; }

class SelectMenu : public TGPopupMenu {
    public:
	SelectMenu(TString what, int idx);

	void HandleMenu(Int_t id);

    private:
	TString	mode;
	int	index;
	TList	entries;
};

static void ShowMenu(TString what, int idx = -1);

static void SetStyle()
{
	style = gROOT->GetStyle("Plain");
	style->SetCanvasColor(0);
	style->SetLineStyleString(5, "[52 12]");
	style->SetLineStyleString(6, "[22 12]");
	style->SetLineStyleString(7, "[22 10 7 10]");

	Double_t stops[] = { 0.00, 0.25, 0.50, 0.75, 1.00 };
	Double_t red[]   = { 0.00, 0.00, 1.00, 1.00, 1.00 };
	Double_t green[] = { 0.00, 1.00, 1.00, 1.00, 0.00 };
	Double_t blue[]  = { 1.00, 1.00, 1.00, 0.00, 0.00 };

	Int_t ourPalette = TColor::CreateGradientColorTable(
					5, stops, red, green, blue, 127);
	style->SetNumberContours(127);

	Int_t pal[127];
	for(Int_t i = 0; i < 127; i++)
		pal[i] = ourPalette + i;
	style->SetPalette(127, pal);
}

void ViewMonitoring(TString fileName = "train_monitoring.root")
{
	file = TFile::Open(fileName);
	if (!file)
		abort();
	root = gClient->GetRoot();

	TControlBar *main =
		new TControlBar("vertical", "MVA Trainer Monitoring", 0, 0);

	SetStyle();
	style->cd();

	main->AddButton("Input Variables", "ShowMenu(\"input\")",
	                "plot input variables for variable processors",
	                "button");

	main->AddButton("Discriminator && Performance", "ShowOutput()",
	                "plot discriminator output & efficiency/purity curve",
	                "button");

	main->AddButton("ProcNormalize", "ShowMenu(\"ProcNormalize\")",
	                "show normalizer PDF distributions", "button");

	main->AddButton("ProcLikelihood (S, B)", "ShowMenu(\"ProcLikelihood\", 1)",
	                "show likelihood ratio PDF distributions", "button");

	main->AddButton("ProcLikelihood (S / (S+B))", "ShowMenu(\"ProcLikelihood\", 0)",
	                "show likelihood ratio S/(S+B) distributions", "button");

	main->AddButton("ProcMatrix", "ShowMenu(\"ProcMatrix\")",
	                "show correlation matrix", "button");

	main->AddButton("Draw All", "DrawAll()",
	                "draw all plots", "button");

	main->AddButton("Options", "ShowMenu(\"options\")",
	                "set options for output", "button");

	main->AddButton("Quit", ".q",
	                "quit", "button");

	main->Show();
}

bool printPS  = true;
bool printEPS = false;
bool printPDF = false;
bool printPNG = false;

void DrawInputs(TDirectory *dir);
void DrawOutput(TDirectory *dir);
void DrawProcNormalize(TDirectory *dir);
void DrawProcLikelihood(TDirectory *dir, int mode);
void DrawProcMatrix(TDirectory *dir);
void DrawAll();

void ShowOutput()
{
	DrawOutput((TDirectory*)file->Get("output"));
}

SelectMenu::SelectMenu(TString what, int idx) : mode(what), index(idx)
{
	if (what == "options") {

	        AddLabel("output options");
	        AddSeparator();

        	unsigned int n = 0;
		entries.Add(new TObjString("common ps-file"));
		AddEntry("common ps-file", n++);
		if ( printPS ) CheckEntry(n-1);
		entries.Add(new TObjString("individual eps-files"));
		AddEntry("individual eps-files", n++);
		if ( printEPS ) CheckEntry(n-1);
		entries.Add(new TObjString("individual pdf-files"));
		AddEntry("individual pdf-files", n++);
		if ( printPDF ) CheckEntry(n-1);
		entries.Add(new TObjString("individual png-files"));
		AddEntry("individual png-files",n++);
		if ( printPNG ) CheckEntry(n-1);

	}
	else {

        	AddLabel(what + " plots");
        	AddSeparator();
        
        	unsigned int n = 0;
        	TIter iter(file->GetListOfKeys());
        	TObject *obj = 0;
        	while((obj = iter.Next()) != 0) {
        		TString name = obj->GetName();
        		if (!name.BeginsWith(what + "_"))
        			continue;
        
        		entries.Add(new TObjString(name));
        
        		Int_t len = what.Length();
        		name = name(len + 1, name.Length() - (len + 1));
        
        		AddEntry(name, n++);
        	}
        
        	if (!n)
        		AddEntry("no entry", -1);

	}

	Connect("Activated(Int_t)", "SelectMenu", this, "HandleMenu(Int_t)");
}

void ShowMenu(TString what, int idx)
{
	SelectMenu *menu = new SelectMenu(what, idx);

	Window_t dum1, dum2;
	Int_t xroot, yroot, x, y;
	UInt_t state;
	gVirtualX->QueryPointer((Window_t)root->GetId(),
	                        dum1, dum2, xroot, yroot, x, y, state);

	menu->Move(x - 10, y - 10);
	menu->MapWindow();

	gVirtualX->GrabPointer((Window_t)menu->GetId(),
	                       kButtonPressMask | kButtonReleaseMask |
	                       kPointerMotionMask, kNone,
	                       gClient->GetResourcePool()->GetGrabCursor());
}

void SelectMenu::HandleMenu(Int_t id)
{
	if (id < 0)
		return;

	TString name = ((TObjString*)entries.At(id))->GetString();

	if (mode == "options") {

	          if (name == "common ps-file")
		          printPS = !printPS;
	          else if (name == "individual eps-files")
		          printEPS = !printEPS;
	          else if (name == "individual pdf-files")
		          printPDF = !printPDF;
	          else if (name == "individual png-files")
		          printPNG = !printPNG;
	          break;

	}
	else {

        	TDirectory *dir = dynamic_cast<TDirectory*>(file->Get(name));
        	if (mode == "input")
        		DrawInputs(dir);
        	else if (mode == "ProcNormalize")
        		DrawProcNormalize(dir);
        	else if (mode == "ProcLikelihood")
        		DrawProcLikelihood(dir, index);
        	else if (mode == "ProcMatrix")
        		DrawProcMatrix(dir);

	}
}

class PadService {
    public:
	PadService(TString name, TString title, Int_t nPlots);
	~PadService();

	TVirtualPad *Next();

        TList canvases;

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
		canvases.Add(last);
		last->SetBorderSize(2);
		last->SetFrameFillColor(0);
		last->SetHighLightColor(0);
		last->Divide(nPadsX, nPadsY);
		index = 0;
	}

	last->cd(++index);
	TVirtualPad *pad = last->GetPad(index);

	if (index == nPadsX * nPadsY)
		last = 0;

	return pad;
}

void Save(TVirtualPad *pad, TDirectory *dir, TString name = "")
{
	gSystem->mkdir("plots");
	TString baseName = TString("plots/") + dir->GetName();
	if (name.Length())
		baseName += "_" + name;

	if ( printEPS )
                pad->Print(baseName + ".eps");
	if ( printPDF )
	        pad->Print(baseName + ".pdf");
	if ( printPNG )
	        pad->Print(baseName + ".png");
}

void Save(PadService &padService)
{
        if ( !printPS ) return;

	gSystem->mkdir("plots");

	for(unsigned int i = 0; i < padService.canvases.GetEntries(); i++)
	      padService.canvases.At(i)->Print("plots/summary.ps");
}

void DrawInputs(TDirectory *dir)
{
	TList *keys = dir->GetListOfKeys();
	TString name = dir->GetName();
	name = name(6, name.Length() - 6);

	PadService pads(dir->GetName(), "\"" + name + "\" input variables",
	                keys->GetSize() / 2);

	TIter iter(keys);
	TObject *obj = 0;
	Int_t idx = 0;
	while((obj = iter.Next()) != 0) {
		TString name = obj->GetName();
		if (!name.EndsWith("_sig"))
			continue;
		name = name(0, name.Length()-4);

		TH1 *bkg = dynamic_cast<TH1*>(dir->Get(name + "_bkg"));
		TH1 *sig = dynamic_cast<TH1*>(dir->Get(name + "_sig"));

		if (!bkg || !sig)
			continue;

		TVirtualPad *pad = pads.Next();

		bkg = (TH1*)bkg->Clone(name + "_tmpI1");
		bkg->SetNormFactor(bkg->Integral() / bkg->Integral("width"));

		sig = (TH1*)sig->Clone(name + "_tmpI2");
		sig->SetNormFactor(sig->Integral() / sig->Integral("width"));

		Double_t scale = (++idx == 1) ? 1.275 : 1.05;
		Double_t x1 = Min(bkg->GetXaxis()->GetXmin(),
		                  sig->GetXaxis()->GetXmin());
		Double_t x2 = Max(bkg->GetXaxis()->GetXmax(),
		                  sig->GetXaxis()->GetXmax());
		Double_t incr = (x2 - x1) * 0.01;
		x1 -= incr;
		x2 += incr;
		Double_t y = Max(bkg->GetMaximum() / bkg->Integral("width"),
		                 sig->GetMaximum() / sig->Integral("width"));
		TH1F *tmp = new TH1F(name + "_tmpI3", name, 1, x1, x2);
		tmp->SetBit(kCanDelete);
		tmp->SetStats(0);
		tmp->GetYaxis()->SetRangeUser(0.0, y * scale);
		tmp->SetXTitle(name);
		tmp->Draw();
		bkg->SetFillColor(2);
		bkg->SetLineColor(2);
		bkg->SetLineWidth(2);
		bkg->SetFillStyle(3554);
		bkg->Draw("same");
		sig->SetFillColor(0);
		sig->SetLineColor(4);
		sig->SetLineWidth(2);
		sig->SetFillStyle(1);
		sig->Draw("same");

		if (idx == 1) {
			TLegend *leg =
				new TLegend(0.6 - pad->GetRightMargin(),
				            1.0 - pad->GetTopMargin() - 0.15,
				            1.0 - pad->GetRightMargin(),
				            1.0 - pad->GetTopMargin());
			leg->SetFillStyle(1);
			leg->AddEntry(sig, "Signal", "F");
			leg->AddEntry(bkg, "Background", "F");
			leg->SetBorderSize(1);
			leg->SetMargin(0.3);
			leg->Draw("same");
		}

		pad->RedrawAxis();
		Save(pad, dir, name);
	}

	Save(pads);
}

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

void DrawOutput(TDirectory *dir)
{
	TList *keys = dir->GetListOfKeys();

	if (keys->GetSize() != 2)
		abort();

	TString name = keys->At(0)->GetName();
	if (!name.EndsWith("_sig") && !name.EndsWith("_bkg"))
		abort();

	name = name(0, name.Length() - 4);

	TH1 *bkg = dynamic_cast<TH1*>(dir->Get(name + "_bkg"));
	TH1 *sig = dynamic_cast<TH1*>(dir->Get(name + "_sig"));

	if (!bkg || !sig)
		abort();

	if (bkg->GetNbinsX() != sig->GetNbinsX())
		abort();

	Int_t bin1 = sig->GetNbinsX();
	Int_t bin2 = 1;
	Double_t x1 = sig->GetXaxis()->GetXmax();
	Double_t x2 = sig->GetXaxis()->GetXmin();
	FindRange(bkg, bin1, bin2, x1, x2);
	FindRange(sig, bin1, bin2, x1, x2);

	PadService pads(dir->GetName(), "discriminator & performance", 5);

	TVirtualPad *pad = pads.Next();

	TH1 *bkg_ = (TH1*)bkg->Clone(name + "_tmpO1");
	bkg_->Rebin(8);
	if (bkg_->Integral("width") != 0.)
	        bkg_->SetNormFactor(bkg_->Integral() / bkg_->Integral("width"));

	TH1 *sig_ = (TH1*)sig->Clone(name + "_tmpO2");
	sig_->Rebin(8);
	if (sig_->Integral("width") != 0.)
	        sig_->SetNormFactor(sig_->Integral() / sig_->Integral("width"));

	Double_t y = Max(bkg_->GetMaximum(), sig_->GetMaximum());
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
	Save(pad, dir, name + "_pdf");

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
	Save(pad, dir, name + "_effs");

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
	Save(pad, dir, name + "_effpur");

	TVirtualPad *pad = pads.Next();

	tmp = new TH1F(name + "_tmpO8",
	               "signal efficiency vs. background rate", n, 0, 1);
	tmp->SetBit(kCanDelete);

	Double_t min = 1.0e-9;
	for(unsigned int i = n + 1; i > 0; i--) {
		if (sum[1]->GetBinContent(i) < 0.1) {
			if (sum[0]->GetBinContent(i) > min)
				min = sum[0]->GetBinContent(i);
		} else
			break;
	}

	pad->SetLogy();
	pad->SetGridx();
	pad->SetGridy();
	pad->SetTickx();
	pad->SetTicky();
	tmp->GetXaxis()->SetRangeUser(0.0, 1.005);
	tmp->GetYaxis()->SetRangeUser(min, 1.005);
	tmp->SetXTitle("signal efficiency");
	tmp->GetYaxis()->SetTitleOffset(1.25);
	tmp->SetYTitle("background rate");
	tmp->SetStats(0);
	tmp->SetLineWidth(2);
	tmp->Draw();

	values[0] = new Float_t[n + 1];
	values[1] = new Float_t[n + 1];
	for(unsigned int i = 0; i <= n; i++) {
		values[0][i] = sum[1]->GetBinContent(i + 1);
		values[1][i] = sum[0]->GetBinContent(i + 1);
	}
	graph = new TGraph(n + 1, values[0], values[1]);
	delete[] values[0];
	delete[] values[1];

	graph->SetName(name + "_tmpO9");
	graph->SetBit(kCanDelete);
	graph->SetLineColor(4);
	graph->SetLineWidth(2);
	graph->Draw("C");

	pad = pads.Next();

	TH1 *rel = (TH1*)sig->Clone(name + "_tmpO10");
	rel->Add(bkg);
	rel->Sumw2();
	rel->Divide(sig, rel, 1.0, 1.0, "B");
	rel->Rebin(8);
	rel->Scale(0.125);
	rel->SetLineColor(4);
	rel->SetMarkerColor(4);
	rel->SetLineWidth(2);
	rel->SetStats(0);
	rel->SetTitle("discriminator signal probability");
	rel->SetXTitle("discriminator");
	rel->SetYTitle("S / (S+B)");

	rel->Draw();

	pad->RedrawAxis();
	Save(pad, dir, name + "_effsigbkg");
	
	Save(pads);
}

void DrawProcNormalize(TDirectory *dir)
{
	TList *keys = dir->GetListOfKeys();
	TString name = dir->GetName();
	name = name(14, name.Length() - 14);

	PadService pads(dir->GetName(),
	                "\"" + name + "\" normalization PDFs",
	                keys->GetSize());

	TIter iter(keys);
	TObject *obj = 0;
	while((obj = iter.Next()) != 0) {
		TString name = obj->GetName();

		TH1 *pdf = dynamic_cast<TH1*>(dir->Get(name));
		if (!pdf)
			continue;

		TVirtualPad *pad = pads.Next();

		pdf = (TH1*)pdf->Clone(name + "_tmpPN1");
		pdf->SetXTitle(name);
		if (pdf->Integral("width") != 0.)
		        pdf->SetNormFactor(pdf->Integral() / pdf->Integral("width"));
		pdf->SetStats(0);
		pdf->SetFillColor(4);
		pdf->SetLineColor(4);
		pdf->SetLineWidth(0);
		pdf->SetFillStyle(3554);
		pdf->Draw("C");

		TH1 *pdf2 = (TH1*)pdf->Clone(name + "_tmpPN2");
		pdf2->SetFillStyle(0);
		pdf2->SetLineWidth(2);
		pdf2->Draw("C same");

		pad->RedrawAxis();
		Save(pad, dir, name);
	}

	Save(pads);
}

void DrawProcLikelihood(TDirectory *dir, int mode)
{
	TList *keys = dir->GetListOfKeys();
	TString name = dir->GetName();
	name = name(15, name.Length() - 15);

	TString what = mode ? "likelihood PDFs" : "S / (S+B) distributions";
	PadService pads(dir->GetName(), "\"" + name + "\" " + what,
	                keys->GetSize() / 2);

	TIter iter(keys);
	TObject *obj = 0;
	Int_t idx = 0;
	while((obj = iter.Next()) != 0) {
		TString name = obj->GetName();
		if (!name.EndsWith("_sig"))
			continue;
		name = name(0, name.Length()-4);

		TH1 *bkg = dynamic_cast<TH1*>(dir->Get(name + "_bkg"));
		TH1 *sig = dynamic_cast<TH1*>(dir->Get(name + "_sig"));

		if (!bkg || !sig)
			continue;

		TVirtualPad *pad = pads.Next();

		bkg = (TH1*)bkg->Clone(name + "_tmpPL1");
		sig = (TH1*)sig->Clone(name + "_tmpPL12");

		Double_t x1 = Min(bkg->GetXaxis()->GetXmin(),
		                  sig->GetXaxis()->GetXmin());
		Double_t x2 = Max(bkg->GetXaxis()->GetXmax(),
		                  sig->GetXaxis()->GetXmax());
		Double_t incr = (x2 - x1) * 0.01;
		x1 -= incr;
		x2 += incr;

		if (mode) {
			bkg->SetNormFactor(bkg->Integral() /
			                   bkg->Integral("width"));
			sig->SetNormFactor(sig->Integral() /
			                   sig->Integral("width"));

			Double_t scale = (++idx == 1) ? 1.275 : 1.05;
			Double_t y = Max(bkg->GetMaximum() /
			                 bkg->Integral("width"),
			                 sig->GetMaximum() /
			                 sig->Integral("width"));
			TH1F *tmp = new TH1F(name + "_tmpPL3", name, 1, x1, x2);
			tmp->SetBit(kCanDelete);
			tmp->SetStats(0);
			tmp->GetYaxis()->SetRangeUser(0.0, y * scale);
			tmp->SetXTitle(name);
			tmp->Draw();
			bkg->SetFillColor(2);
			bkg->SetLineColor(2);
			bkg->SetLineWidth(0);
			bkg->SetFillStyle(3554);
			bkg->Draw("C same");
			TH1 *bkg2 = (TH1*)bkg->Clone(name + "_tmpPL4");
			bkg2->SetFillStyle(0);
			bkg2->SetLineWidth(2);
			bkg2->Draw("C same");
			sig->SetFillColor(0);
			sig->SetLineColor(4);
			sig->SetLineWidth(2);
			sig->SetFillStyle(1);
			sig->Draw("C same");

			if (idx == 1) {
				TLegend *leg =
					new TLegend(0.6 - pad->GetRightMargin(),
					            1.0 - pad->GetTopMargin() - 0.15,
					            1.0 - pad->GetRightMargin(),
					            1.0 - pad->GetTopMargin());
				leg->SetFillStyle(1);
				leg->AddEntry(sig, "Signal", "F");
				bkg->SetLineWidth(2);
				leg->AddEntry(bkg, "Background", "F");
				leg->SetBorderSize(1);
				leg->SetMargin(0.3);
				leg->Draw("same");
			}
		} else {
			bkg->Sumw2();
			sig->Sumw2();
			bkg->Scale(1.0 / bkg->Integral());
			sig->Scale(1.0 / sig->Integral());

			bkg->Add(sig);
			sig->Divide(sig, bkg, 1, 1, "B");

			Double_t scale = (++idx == 1) ? 1.16 : 1.05;

			sig->SetStats(0);
			sig->GetYaxis()->SetRangeUser(0.0, scale);
			sig->SetXTitle(name);
			sig->SetLineColor(4);
			sig->SetMarkerColor(4);
			sig->SetLineWidth(2);
			sig->Draw();

			if (idx == 1) {
				TLegend *leg =
					new TLegend(0.7 - pad->GetRightMargin(),
					            1.0 - pad->GetTopMargin() - 0.07,
					            1.0 - pad->GetRightMargin(),
					            1.0 - pad->GetTopMargin());
				leg->SetFillStyle(1);
				leg->AddEntry(sig, "S / (S+B)", "F");
				leg->SetBorderSize(1);
				leg->SetMargin(0.3);
				leg->Draw("same");
			}
		}

		pad->RedrawAxis();
		Save(pad, dir, name);
	}

	Save(pads);
}

void DrawProcMatrix(TDirectory *dir)
{
	TString name = dir->GetName();
	name = name(11, name.Length() - 11);

	PadService pads(dir->GetName(),
	                "\"" + name + "\" correlation matrix", 2);

	TVirtualPad *pad = pads.Next();
	pad->SetLeftMargin(0.05);

	const unsigned int showNbins = 10;
	TH1 *rank = dynamic_cast<TH1*>(dir->Get("Ranking"));
	TString *labels;
	if (rank) {
		labels = new TString[rank->GetNbinsX()];
		for(unsigned int i = 1; i <= rank->GetNbinsX(); i++) {
			labels[i - 1] = rank->GetXaxis()->GetBinLabel(i);
			rank->GetXaxis()->SetBinLabel(i,
				Form("%d.", rank->GetNbinsX() - i + 1));
		}
		rank->SetStats(0);
		rank->SetFillColor(kGreen);
		if (rank->GetNbinsX() > showNbins)
		        rank->GetXaxis()->SetRange(rank->GetNbinsX() - showNbins + 1, rank->GetNbinsX());
		rank->Draw("hbar2");
		std::cout << std::endl
			  << "====================================================" << std::endl
			  << "             Result of Variable Ranking             " << std::endl
			  << "----------------------------------------------------" << std::endl;
		for(unsigned int i = rank->GetNbinsX(); i >= 1; i--) {
			double v = fabs(rank->GetBinContent(i + 1) -
			                rank->GetBinContent(i));
			TString text = labels[i - 1] +
			               Form(": %+1.2f%%", v * 100.0);
			double off = rank->GetMaximum() * 0.1;
			TText *t = new TText(off, i - 0.5, text);
			if ((int) i >= rank->GetNbinsX() - (int) showNbins + 1)
			        t->Draw();
			std::cout << std::setw(4) << rank->GetXaxis()->GetBinLabel(i) << " "
				  << std::setw(38) << std::setiosflags(ios::left) << labels[i - 1]
				  << std::setiosflags(ios::right) << " +" 
				  << std::setw(5) << std::setprecision(2) << std::setiosflags(ios::fixed)
				  << (v * 100.0) << "%" << std::endl;
		}
		std::cout << "====================================================" << std::endl;
		Save(pad, dir, "ranking");
		pad = pads.Next();
	}

	pad->SetLeftMargin(0.15);
	pad->SetRightMargin(0.13);
	pad->SetTopMargin(0.13);

	TH2 *matrixTmp = dynamic_cast<TH2*>(dir->Get("CorrMatrix"));
	matrixTmp = (TH2*)matrixTmp->Clone(name + "_tmpPM");

	TH2 *matrix = matrixTmp;
	if (rank && rank->GetNbinsX() > showNbins) {
		unsigned int iSave[showNbins];
		for(unsigned int i = 0; i < showNbins; i++) {
		        unsigned int iS = 1;
		        for(; iS <= matrixTmp->GetNbinsX(); iS++) {
		                if (matrixTmp->GetXaxis()->GetBinLabel(iS) == labels[rank->GetNbinsX() - i - 1]) {
		                        iSave[i] = iS;
		                        break;
		                }
		        }
		}
	        matrix = new TH2F(matrixTmp->GetName(), matrixTmp->GetTitle(),
				  showNbins + 1, 0., showNbins + 1., showNbins + 1, 0., showNbins + 1.);
		matrix->SetMinimum(-1.0);
		matrix->SetMaximum(+1.0);
		for(unsigned int i = 1; i <= showNbins; i++) {
		        matrix->GetXaxis()->SetBinLabel(i, Form("%d.", i));
		        matrix->GetYaxis()->SetBinLabel(i, Form("%d.", i));
		        for(unsigned int j = i + 1; j <= showNbins; j++) {
		                matrix->SetBinContent(i, j, matrixTmp->GetBinContent(iSave[i-1], iSave[j-1]));
		                matrix->SetBinContent(j, i, matrixTmp->GetBinContent(iSave[i-1], iSave[j-1]));
		        }
		        matrix->SetBinContent(i, showNbins + 1, matrixTmp->GetBinContent(iSave[i-1], matrixTmp->GetNbinsX()));
		        matrix->SetBinContent(showNbins + 1, i, matrixTmp->GetBinContent(iSave[i-1], matrixTmp->GetNbinsX()));
		}
		matrix->GetXaxis()->SetBinLabel(showNbins + 1, "target");
		matrix->GetYaxis()->SetBinLabel(showNbins + 1, "target");
	}

	matrix->SetStats(0);
	matrix->SetLabelOffset(0.011);
	matrix->LabelsOption("d");
	matrix->GetXaxis()->SetLabelSize(0.04);
	matrix->GetYaxis()->SetLabelSize(0.04);
	matrix->Draw("colz");
	TBox box;
	TLine line;
	box.SetFillColor(kBlack);
	line.SetLineColor(kBlack);
	for(unsigned int i = 0; i < matrix->GetNbinsX(); i++)
		box.DrawBox(i, i, i + 1, i + 1);
	line.DrawLine(matrix->GetNbinsX() - 1, 0,
	              matrix->GetNbinsX() - 1, matrix->GetNbinsX());
	line.DrawLine(0, matrix->GetNbinsX() - 1,
	              matrix->GetNbinsX(), matrix->GetNbinsX() - 1);

	pad->RedrawAxis();
	Save(pad, dir, "corrMatrix");

	Save(pads);

	delete[] labels;
}

void DrawAll()
{

        TCanvas *dummyCanvas = new TCanvas("dummyCanvas", "MVA Trainer Monitoring Summary", 0, 0, 800, 440);

	TPaveText paveText(0.1, 0.6, 0.9, 0.9);
	paveText.SetFillColor(0);
	paveText.SetBorderSize(0);

        gSystem->mkdir("plots");

	if ( printPS )
	        dummyCanvas->Print("plots/summary.ps[", "Landscape");

	paveText.AddText("MVA Trainer Monitoring:");
	paveText.AddText("discriminator output & efficiency/purity curve");
	paveText.Draw();
	if ( printPS )
	        dummyCanvas->Print("plots/summary.ps");

	DrawOutput((TDirectory*)file->Get("output"));

	paveText.Clear();
	paveText.AddText("MVA Trainer Monitoring:");
	paveText.AddText("input variables for variable processors");
	paveText.Draw();
	if ( printPS )
	        dummyCanvas->Print("plots/summary.ps");

	TIter iter(file->GetListOfKeys());
	TObject *obj = 0;
	while((obj = iter.Next()) != 0) {
	  TString name = obj->GetName();
	  if (!name.BeginsWith("input_"))
	    continue;

	    TDirectory *dir = dynamic_cast<TDirectory*>(file->Get(name));
	    DrawInputs(dir);
	}

	paveText.Clear();
	paveText.AddText("MVA Trainer Monitoring:");
	paveText.AddText("normalizer PDF distributions");
	paveText.Draw();
	if ( printPS )
	        dummyCanvas->Print("plots/summary.ps");

	TIter iter(file->GetListOfKeys());
	TObject *obj = 0;
	while((obj = iter.Next()) != 0) {
	  TString name = obj->GetName();
	  if (!name.BeginsWith("ProcNormalize_"))
	    continue;

	    TDirectory *dir = dynamic_cast<TDirectory*>(file->Get(name));
	    DrawProcNormalize(dir);
	}

	paveText.Clear();
	paveText.AddText("MVA Trainer Monitoring:");
	paveText.AddText("correlation matrix");
	paveText.Draw();
	if ( printPS )
	        dummyCanvas->Print("plots/summary.ps");

	TIter iter(file->GetListOfKeys());
	TObject *obj = 0;
	while((obj = iter.Next()) != 0) {
	  TString name = obj->GetName();
	  if (!name.BeginsWith("ProcMatrix_"))
	    continue;

	    TDirectory *dir = dynamic_cast<TDirectory*>(file->Get(name));
	    DrawProcMatrix(dir);
	}

	paveText.Clear();
	paveText.AddText("MVA Trainer Monitoring:");
	paveText.AddText("likelihood ratio PDF distributions");
	paveText.Draw();
	if ( printPS )
	        dummyCanvas->Print("plots/summary.ps");

	TIter iter(file->GetListOfKeys());
	TObject *obj = 0;
	while((obj = iter.Next()) != 0) {
	  TString name = obj->GetName();
	  if (!name.BeginsWith("ProcLikelihood_"))
	    continue;

	    TDirectory *dir = dynamic_cast<TDirectory*>(file->Get(name));
	    DrawProcLikelihood(dir, 1);
	}

	paveText.Clear();
	paveText.AddText("MVA Trainer Monitoring:");
	paveText.AddText("likelihood ratio S/(S+B) distributions");
	paveText.Draw();
	if ( printPS )
	        dummyCanvas->Print("plots/summary.ps");

	TIter iter(file->GetListOfKeys());
	TObject *obj = 0;
	while((obj = iter.Next()) != 0) {
	  TString name = obj->GetName();
	  if (!name.BeginsWith("ProcLikelihood_"))
	    continue;

	    TDirectory *dir = dynamic_cast<TDirectory*>(file->Get(name));
	    DrawProcLikelihood(dir, 0);
	}

        if ( printPS )
	        dummyCanvas->Print("plots/summary.ps]");

	dummyCanvas->Close();

}
