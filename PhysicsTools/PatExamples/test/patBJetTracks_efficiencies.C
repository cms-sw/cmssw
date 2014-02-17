// name of analyzer
static const char *directory = "analyzeBJetTracks";

static const char *flavours[] = { "b", "c", "udsg", 0 };

void patBJetTracks_efficiencies()
{
	// define proper canvas style
	setNiceStyle();
	gStyle->SetOptStat(0);

	// open file
	TFile* file = new TFile("analyzePatBJetTracks.root");

	TLegend *legend[3] = { 0, 0, 0 };

	// draw canvas with efficiencies

	TCanvas *canv;
	canv = new TCanvas("canv0", "hand-crafted track counting efficiencies", 800, 300);
	canv->Divide(3, 1);

	TH1 *total = (TH1*)file->Get(Form("%s/flavours", directory));
	TH1 *effVsCutB = 0;
	unsigned int i = 0;
	for(const char **flavour = flavours; *flavour; flavour++, i++) {
		TH1 *h = (TH1*)file->Get(Form("%s/trackIPSig_%s", directory, *flavour));
		TH1 *discrShape = (TH1*)h->Clone(Form("%s_discrShape", h->GetName()));
		discrShape->Scale(1.0 / discrShape->Integral());
		discrShape->SetMaximum(discrShape->GetMaximum() * 5);
		TH1 *effVsCut = computeEffVsCut(h, total->GetBinContent(4 - i));
		TH1 *effVsBEff = 0;

		if (flavour == flavours)	// b-jets
			effVsCutB = effVsCut;
		else
			effVsBEff = computeEffVsBEff(effVsCut, effVsCutB);

		discrShape->SetTitle("discriminator shape");
		effVsCut->SetTitle("efficiency versus discriminator cut");
		if (effVsBEff)
			effVsBEff->SetTitle("mistag versus b efficiency");

		setHistStyle(discrShape);
		setHistStyle(effVsCut);
		setHistStyle(effVsBEff);

		canv->cd(1);
		gPad->SetLogy(1);
		gPad->SetGridy(1);
		discrShape->SetLineColor(i + 1);
		discrShape->SetMarkerColor(i + 1);
		discrShape->Draw(i > 0 ? "same" : "");
		if (!legend[0])
			legend[0] = new TLegend(0.5, 0.7, 0.78, 0.88);
		legend[0]->AddEntry(discrShape, *flavour);

		canv->cd(2);
		gPad->SetLogy(1);
		gPad->SetGridy(1);
		effVsCut->SetLineColor(i + 1);
		effVsCut->SetMarkerColor(i + 1);
		effVsCut->Draw(i > 0 ? "same" : "");
		if (!legend[1])
			legend[1] = new TLegend(0.12, 0.12, 0.40, 0.30);
		legend[1]->AddEntry(effVsCut, *flavour);

		if (!effVsBEff)
			continue;
		canv->cd(3);
		gPad->SetLogy(1);
		gPad->SetGridx(1);
		gPad->SetGridy(1);
		effVsBEff->SetLineColor(i + 1);
		effVsBEff->SetMarkerColor(i + 1);
		effVsBEff->Draw(i > 1 ? "same" : "");
		if (!legend[2])
			legend[2] = new TLegend(0.12, 0.7, 0.40, 0.88);
		legend[2]->AddEntry(effVsBEff, *flavour);
	}

	canv->cd(1);
	legend[0]->Draw();

	canv->cd(2);
	legend[1]->Draw();

	canv->cd(3);
	legend[2]->Draw();

	////////////////////////////////////////////

	// canvas to compare negative tagger with light flavour mistag

	TCanvas *canv;
	canv = new TCanvas("canv1", "comparing light flavour mistag with negative tagger", 530, 300);
	canv->Divide(2, 1);

	TH1 *h1 = (TH1*)file->Get(Form("%s/trackIPSig_udsg", directory));
	TH1 *h2 = (TH1*)file->Get(Form("%s/negativeIPSig_all", directory));
	h2 = invertHisto(h2);	// invert x-axis

	TH1 *discrShape1 = (TH1*)h1->Clone("discrShape1");
	TH1 *discrShape2 = (TH1*)h2->Clone("discrShape2");

	discrShape1->Scale(1.0 / discrShape1->Integral());
	discrShape1->SetMaximum(discrShape1->GetMaximum() * 5);
	discrShape2->Scale(1.0 / discrShape2->Integral());

	TH1 *effVsCut1 = computeEffVsCut(h1, total->GetBinContent(2));
	TH1 *effVsCut2 = computeEffVsCut(h2, total->GetBinContent(1));

	discrShape1->SetTitle("discriminator shape");
	effVsCut1->SetTitle("efficiency versus discriminator cut");

	setHistStyle(discrShape1);
	setHistStyle(discrShape2);
	setHistStyle(effVsCut1);
	setHistStyle(effVsCut2);

	canv->cd(1);
	gPad->SetLogy(1);
	gPad->SetGridy(1);
	discrShape1->SetLineColor(1);
	discrShape1->SetMarkerColor(1);
	discrShape2->SetLineColor(2);
	discrShape2->SetMarkerColor(2);

	discrShape1->Draw();
	discrShape2->Draw("same");

	TLegend *l = new TLegend(0.5, 0.7, 0.78, 0.88);
	l->AddEntry(discrShape1, "udsg");
	l->AddEntry(discrShape2, "inv. neg");
	l->Draw();

	canv->cd(2);
	gPad->SetLogy(1);
	gPad->SetGridy(1);
	effVsCut1->SetLineColor(1);
	effVsCut1->SetMarkerColor(1);
	effVsCut2->SetLineColor(2);
	effVsCut2->SetMarkerColor(2);

	effVsCut1->Draw();
	effVsCut2->Draw("same");

	l = new TLegend(0.5, 0.7, 0.78, 0.88);
	l->AddEntry(effVsCut1, "udsg");
	l->AddEntry(effVsCut2, "inv. neg");
	l->Draw();
}

TH1 *computeEffVsCut(TH1 *discrShape, double total)
{
	TH1 *h = discrShape->Clone(Form("%s_effVsCut", discrShape->GetName()));
	h->Sumw2();
	h->SetMaximum(1.5);
	h->SetMinimum(1e-3);

	unsigned int n = h->GetNbinsX();
	for(unsigned int bin = 1; bin <= n; bin++) {
		double efficiency = h->Integral(bin, n + 1) / total;
		double error = sqrt(efficiency * (1 - efficiency) / total);
		h->SetBinContent(bin, efficiency);
		h->SetBinError(bin, error);
	}

	return h;
}

TH1 *computeEffVsBEff(TH1 *effVsCut, TH1 *effVsCutB)
{
	TH1 *h = new TH1F(Form("%s_effVsBEff", effVsCut->GetName()), "effVsBEff",
	                  100, 0, 1);
	h->SetMaximum(1.5);
	h->SetMinimum(1e-3);

	unsigned int n = effVsCut->GetNbinsX();
	for(unsigned int bin = 1; bin <= n; bin++) {
		double eff = effVsCut->GetBinContent(bin);
		double error = effVsCut->GetBinError(bin);
		double effB = effVsCutB->GetBinContent(bin);

		h->SetBinContent(h->FindBin(effB), eff);
		h->SetBinError(h->FindBin(effB), error);
 		// FIXME: The error in effB is not propagated
	}

	return h;
}

TH1 *invertHisto(TH1 *h)
{
	unsigned int n = h->GetNbinsX();
	TH1 *inv = new TH1F(Form("%s_inverted", h->GetName()), "inverted",
	                    n, -h->GetXaxis()->GetXmax(), -h->GetXaxis()->GetXmin());
	for(unsigned int i = 0; i <= n + 1; i++)
		inv->SetBinContent(n + 1 - i, h->GetBinContent(i));

	return inv;
}

void setAxisStyle(TH1 *hist) {
	// --------------------------------------------------
	// define proper axsis style for a given histogram
	// --------------------------------------------------
	hist->GetXaxis()->SetTitleSize( 0.06);
	hist->GetXaxis()->SetTitleColor( 1);
	hist->GetXaxis()->SetTitleOffset( 0.8);
	hist->GetXaxis()->SetTitleFont( 62);
	hist->GetXaxis()->SetLabelSize( 0.05);
	hist->GetXaxis()->SetLabelFont( 62);
	hist->GetXaxis()->CenterTitle();
	hist->GetXaxis()->SetNdivisions( 505);

	hist->GetYaxis()->SetTitleSize( 0.07);
	hist->GetYaxis()->SetTitleColor( 1);
	hist->GetYaxis()->SetTitleOffset( 0.5);
	hist->GetYaxis()->SetTitleFont( 62);
	hist->GetYaxis()->SetLabelSize( 0.05);
	hist->GetYaxis()->SetLabelFont( 62);
}

void setHistStyle(TH1 *hist)
{
	if (!hist)
		return;

	// --------------------------------------------------
	// define proper histogram style
	// --------------------------------------------------
	setAxisStyle(hist);
	hist->GetXaxis()->SetTitle(hist->GetTitle());
	hist->SetTitle();
	hist->SetLineColor(4.);
	hist->SetLineWidth(2.);
	hist->SetMarkerSize(0.75);
	hist->SetMarkerColor(4.);
	hist->SetMarkerStyle(20.);
}

void setNiceStyle() 
{
	gROOT->SetStyle("Plain");

	// --------------------------------------------------
	// define proper canvas style
	// --------------------------------------------------
	TStyle *MyStyle = new TStyle ("MyStyle", "My style for nicer plots");
	
	Float_t xoff = MyStyle->GetLabelOffset("X"),
	        yoff = MyStyle->GetLabelOffset("Y"),
	        zoff = MyStyle->GetLabelOffset("Z");

	MyStyle->SetCanvasBorderMode ( 0 );
	MyStyle->SetPadBorderMode    ( 0 );
	MyStyle->SetPadColor         ( 0 );
	MyStyle->SetCanvasColor      ( 0 );
	MyStyle->SetTitleColor       ( 0 );
	MyStyle->SetStatColor        ( 0 );
	MyStyle->SetTitleBorderSize  ( 0 );
	MyStyle->SetTitleFillColor   ( 0 );
	MyStyle->SetTitleH        ( 0.07 );
	MyStyle->SetTitleW        ( 1.00 );
	MyStyle->SetTitleFont     (  132 );

	MyStyle->SetLabelOffset (1.5*xoff, "X");
	MyStyle->SetLabelOffset (1.5*yoff, "Y");
	MyStyle->SetLabelOffset (1.5*zoff, "Z");

	MyStyle->SetTitleOffset (0.9,      "X");
	MyStyle->SetTitleOffset (0.9,      "Y");
	MyStyle->SetTitleOffset (0.9,      "Z");

	MyStyle->SetTitleSize   (0.045,    "X");
	MyStyle->SetTitleSize   (0.045,    "Y");
	MyStyle->SetTitleSize   (0.045,    "Z");

	MyStyle->SetLabelFont   (132,      "X");
	MyStyle->SetLabelFont   (132,      "Y");
	MyStyle->SetLabelFont   (132,      "Z");

	MyStyle->SetPalette(1);

	MyStyle->cd();
}
