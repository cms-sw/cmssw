static void setStyle()
{
	style = gROOT->GetStyle("Plain");

	Double_t stops[] = { 0.00, 0.25, 0.50, 0.75, 1.00 };
	Double_t red[]   = { 0.00, 0.00, 1.00, 1.00, 1.00 };
	Double_t green[] = { 0.00, 1.00, 1.00, 1.00, 0.00 };
	Double_t blue[]  = { 1.00, 1.00, 1.00, 0.00, 0.00 };

	Int_t ourPalette = TColor::CreateGradientColorTable(
					5, stops, red, green, blue, 511);

	style->SetNumberContours(511);

	Int_t pal[511];
	for(Int_t i = 0; i < 511; i++)
		pal[i] = ourPalette + i;
	style->SetPalette(511, pal);
}

void testMVATrainerAnalysis()
{
	using namespace PhysicsTools;

	setStyle();

	MVAComputer mva("testMVAComputerEvaluate.mva");

	Variable::Value values[3];
	values[0].setName("x");
	values[1].setName("y");

	TH2F *f = new TH2F("discr", "Discriminator", 200, -10, 10, 200, -10, 10);
	f->SetXTitle("x");
	f->SetYTitle("y");

	TH2F *g = new TH2F("dx", "dD/dx", 200, -10, 10, 200, -10, 10);
	g->SetXTitle("x");
	g->SetYTitle("y");

	TH2F *h = new TH2F("dy", "dD/dy", 200, -10, 10, 200, -10, 10);
	h->SetXTitle("x");
	h->SetYTitle("y");

	for(double x = -10 + 0.05; x < 10; x += 0.1)  {
		for(double y = -10 + 0.05; y < 10; y += 0.1) {
			values[0].setValue(x);
			values[1].setValue(y);
			double v = mva.deriv(values, values + 2);
			f->SetBinContent(f->FindBin(x, y), v);
			g->SetBinContent(g->FindBin(x, y), values[0].getValue());
			h->SetBinContent(h->FindBin(x, y), values[1].getValue());
		}
	}

	TCanvas *c1 = new TCanvas("c1");
	c1->Divide(2, 2);
	c1->cd(1);
	f->SetStats(0);
	f->SetContour(511);
	f->Draw("colz");
	c1->cd(3);
	g->SetStats(0);
	g->SetContour(511);
	g->Draw("colz");
	c1->cd(4);
	h->SetStats(0);
	h->SetContour(511);
	h->Draw("colz");
}
