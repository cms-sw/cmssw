void paint(TString a, TString b)
{
	TChain *t = new TChain(a);
	t->Add(a + "_" + b +"*.root");

	TDirectory *dir = new TDirectory("dir", "dir");
	dir->cd();

//	histo = new TH2D("jets", "jets", 50, -2.5, 2.5, 60, 10, 610);
//	histo = new TH2D("jets", "jets", 50, -2.5, 2.5, 40, 4.7004803657924166, 6.404803657924166);
	histo = new TH2D("jets", "jets", 50, -2.5, 2.5, 40, 4.0943445622221004, 6.1943445622221004);
	histo->SetDirectory(dir);

	t->Draw("log(jetPt+50):jetEta >> +jets", "", "Lego goff");

	TFile g(a + "_" + b +"_histo.root", "RECREATE");
	histo->SetDirectory(&g);
	delete dir;

	g.cd();
	histo->Write();
	g.Close();
}
