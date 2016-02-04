// name of analyzer
static const char *directory = "analyzeBJetVertex";

static const char *plots[] = {
	"dist", "distErr", "distSig",
	"nTracks", "nVertices", "mass", "chi2", "deltaR",
	0
};

static const char *flavours[] = { "b", "c", "udsg", 0 };

void patBJetVertex_showPlots()
{
	// define proper canvas style
	setNiceStyle();
	gStyle->SetOptStat(0);

	// open file
	TFile* file = new TFile("analyzePatBJetVertex.root");

	// draw canvas with track observables
	unsigned int i = 3;
	unsigned int j = 0;

	for(const char **plot = plots; *plot; plot++) {
		if (i >= 3) {
			canv = new TCanvas(Form("canv%d", j++), "secondary vertex variables", 800, 400);
			canv->Divide(3, 2);
			i -= 3;
		}

		canv->cd(i + 1);
		if (TString(*plot).Contains("dist"))
			gPad->SetLogy(1);
		TH1 *h = (TH1*)file->Get(Form("%s/%s_all", directory, *plot));
		TString title = h->GetTitle();
		title.Resize(title.Index(" in "));
		h->SetTitle(title);
		setHistStyle(h);
		h->Draw();
		TLegend *l = new TLegend(0.6, 0.75, 0.85, 0.85);
		l->AddEntry(h, "all");
		l->Draw();

		canv->cd(i + 4);
		unsigned int k = 1;
		if (TString(*plot).Contains("dist"))
			gPad->SetLogy(1);
		l = new TLegend(0.5, 0.6, 0.85, 0.85);
		for(const char **flavour = flavours; *flavour; flavour++) {
			h = (TH1*)file->Get(Form("%s/%s_%s", directory, *plot, *flavour));
			title = h->GetTitle();
			title.Resize(title.Index(" in "));
			h->SetTitle(title);
			setHistStyle(h);
			h->SetMarkerColor(k);
			h->SetLineColor(k++);
			h->DrawNormalized(k > 1 ? "same" : "");
			l->AddEntry(h, *flavour);
		}
		l->Draw();
		i++;
	}

	canv->cd(3);
	TH1 *h = (TH1*)file->Get(Form("%s/flavours", directory));
	setHistStyle(h);
	h->Draw();
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
