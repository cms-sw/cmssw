static const char *names[] = {
	"pt0",
	"pt1",
	"pt2",
	"pt3",
	"pt4",
	"delta1",
	"delta2",
	"delta3",
	"delta4"
};

static const char *xtitles[] = {
	"log_{10}(E_{T} [GeV/c^{2}])",
	"log_{10}(E_{T} [GeV/c^{2}])",
	"log_{10}(E_{T} [GeV/c^{2}])",
	"log_{10}(E_{T} [GeV/c^{2}])",
	"log_{10}(E_{T} [GeV/c^{2}])",
	"#DeltaR",
	"#DeltaR",
	"#DeltaR",
	"#DeltaR"
};
static int n = 9;

void plotMatchingAnalysis()
{
	TFile *f0 = TFile::Open("ana_tt0j.root");
	TFile *f1 = TFile::Open("ana_tt1j.root");
	TFile *f2 = TFile::Open("ana_tt2j.root");
	TFile *f3 = TFile::Open("ana_tt3j.root");
	TFile *f4 = TFile::Open("ana_tt4j.root");

	for(int i = 0; i < n; i++) {
		TH1 *h0 = (TH1*)f0->Get(TString("lheAnalyzer/") + names[i]);
		TH1 *h1 = (TH1*)f1->Get(TString("lheAnalyzer/") + names[i]);
		TH1 *h2 = (TH1*)f2->Get(TString("lheAnalyzer/") + names[i]);
		TH1 *h3 = (TH1*)f3->Get(TString("lheAnalyzer/") + names[i]);
		TH1 *h4 = (TH1*)f4->Get(TString("lheAnalyzer/") + names[i]);

		h0->Sumw2();
		h1->Sumw2();
		h2->Sumw2();
		h3->Sumw2();
		h4->Sumw2();

		// FIXME: add weights here (xsec / #events)
		h0->Scale(3.1821e-07/165633);
		h1->Scale(8.98764e-08/55910);
		h2->Scale(1.72561e-08/7800);
		h3->Scale(2.71928e-09/11747);
		h4->Scale(7.20582e-10/11774);

		TH1 *sum = h0->Clone(TString("sum_") + names[i]);
		sum->Add(h1);
		sum->Add(h2);
		sum->Add(h3);
		sum->Add(h4);

		sum->SetMarkerColor(kBlack);
		sum->SetLineColor(kBlack);  
		h0->SetMarkerColor(kRed); 
		h0->SetLineColor(kRed);   
		h1->SetMarkerColor(kBlue);
		h1->SetLineColor(kBlue);  
		h2->SetMarkerColor(kGreen);
		h2->SetLineColor(kGreen);  
		h3->SetMarkerColor(kCyan); 
		h3->SetLineColor(kCyan);   
		h4->SetMarkerColor(kMagenta);
		h4->SetLineColor(kMagenta);      

		TCanvas *c = new TCanvas(TString("c_") + names[i]);
		c->SetLogy();

		sum->SetXTitle(xtitles[i]);
		sum->SetStats(0);
		sum->Draw();
		h0->Draw("same");
		h1->Draw("same");
		h2->Draw("same");
		h3->Draw("same");
		h4->Draw("same");

		c->Print(TString(names[i]) + ".png");
	}
}
