// change this to analyzeMuons to analyze the selected PAT layer 1 muons
static const char *directory = "analyzeTracks";
//static const char *directory = "analyzeMuons";

static const char *qualities[] = {
	"loose", "tight", "highPurity",
//	"undefQuality",
	0
};

static const char *plots[] = {
	"pt", "ptErr", "eta",
	"invPt", "invPtErr", "phi",
	"d0", "d0Err", "nHits",
	0
};

static const char *components[] = {
	"pxb", "pxe",
	"tib", "tid", "tob", "tec",
	0
};

void patTracks_showPlots()
{
	// define proper canvas style
	setNiceStyle();
	gStyle->SetOptStat(0);

	// open file
	TFile* file = new TFile("analyzePatTracks.root");

	// draw canvas with track observables
	TCanvas* canv0 = new TCanvas("canv0", "track variables", 800, 600);
	canv0->Divide(3, 3);

	unsigned int i = 1;
	for(const char **plot = plots; *plot; plot++, i++) {
		canv0->cd(i);
		unsigned int j = 0;

		TLegend *l = new TLegend(0.6, 0.6, 0.85, 0.85);
		
		for(const char **quality = qualities; *quality; quality++, j++) {
			TH1 *h = file->Get(Form("%s/%s_%s", directory, *plot, *quality));
			TString title = h->GetTitle();
			title.Resize(title.Length() - (strlen(*quality) + 3));
			h->SetTitle(title);
			setHistStyle(h);
			h->SetLineColor(j + 2);
			h->SetMarkerColor(j + 2);
			h->Draw(j ? "same" : "");
			l->AddEntry(h, *quality);
		}

		l->Draw();
	}

	// draw canvas for hits tracking components
	TCanvas* canv1 = new TCanvas("canv1", "tracking components", 800, 600);

	THStack *hs = new THStack("components", "average #hits in tracking components");
	TLegend *l = new TLegend(0.12, 0.15, 0.45, 0.35);
	const char *quality = qualities[0];
	unsigned int i = 0;
	for(const char **component = components; *component; component++, i++) {
		TProfile *p = file->Get(Form("%s/%sHitsEta_%s", directory, *component, quality));
		TString title = p->GetTitle();
		title.Resize(title.Length() - (strlen(quality) + 3));
		setHistStyle(p);
		TH1 *h = new TH1F(*component, title, p->GetNbinsX(), p->GetXaxis()->GetXmin(), p->GetXaxis()->GetXmax());
		for(unsigned int j = 1; j <= p->GetNbinsX(); j++)
			h->SetBinContent(j, p->GetBinContent(j));
		h->SetLineColor(kBlack);
		h->SetMarkerColor(kBlack);
		h->SetFillColor(i + 2);
		hs->Add(h);
		l->AddEntry(h);
	}
	hs->Draw();
	l->Draw();
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
