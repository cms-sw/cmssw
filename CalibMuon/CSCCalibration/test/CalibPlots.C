
{
	gROOT->ProcessLine(".x /home/darinb/CMSStyle.C");
	TFile f("All.root");

	TTree *DiffGains = (TTree*)f->Get("DiffGains");
	TTree *DiffMatrix = (TTree*)gDirectory->Get("DiffMatrix");
	TTree *DiffPeds = (TTree*)gDirectory->Get("DiffPeds");
	TTree *DiffXtalk = (TTree*)gDirectory->Get("DiffXtalk");

	TH2F h2("h2","",500,-1,1,230000,0,230000);
	h2->SetMarkerSize(.1);
	h2->SetStats(0);

	TLatex* txt =new TLatex(-.9,200000, "#frac{new - old}{old} ");
	txt->SetTextFont(132);
	txt->SetTextSize(0.04);
	h2->GetXaxis()->SetTitleSize(.05);
	h2->GetYaxis()->SetTitleSize(.05);
	h2->GetYaxis()->SetTitle("Strip Index");

	h2->GetXaxis()->SetTitle("Gains (Relative Change)");
								 // txt->Draw();
	DiffGains->Draw("index:diffGains>>h2");
	c1.SetGridx(1);

	c1->Print("diffGains_ind.png");

	h2->GetXaxis()->SetTitle("Pedestals (Relative Change)");
								 // txt->Draw();
	DiffPeds->Draw("index:diffPeds>>h2");
	c1->Print("diffPeds_ind.png");

	h2->GetXaxis()->SetTitle("CrossTalk Right (Relative Change)");
								 //txt->Draw();
	DiffXtalk->Draw("index:diffXtalkR>>h2");
	c1->Print("diffXtalkR_ind.png");

	h2->GetXaxis()->SetTitle("CrossTalk Right Int (Relative Change)");
								 //txt->Draw();
	DiffXtalk->Draw("index:diffIntR>>h2");
	c1->Print("diffIntR_ind.png");

	h2->GetXaxis()->SetTitle("CrossTalk Left (Relative Change)");
								 //txt->Draw();
	DiffXtalk->Draw("index:diffXtalkL>>h2");
	c1->Print("diffXtalkL_ind.png");

	h2->GetXaxis()->SetTitle("CrossTalk Left Int (Relative Change)");
								 // txt->Draw();
	DiffXtalk->Draw("index:diffIntL>>h2");
	c1->Print("diffIntL_ind.png");

	//	DiffMatrix->Draw("index:diffElem33");
	//	c1->Print("diffElem33_ind.png");
	//	DiffMatrix->Draw("index:diffElem34");
	//	c1->Print("diffElem34_ind.png");
	//	DiffMatrix->Draw("index:diffElem44");
	//	c1->Print("diffElem44_ind.png");
	//	DiffMatrix->Draw("index:diffElem35");
	//	c1->Print("diffElem35_ind.png");
	//	DiffMatrix->Draw("index:diffElem45");
	//	c1->Print("diffElem45_ind.png");
	//	DiffMatrix->Draw("index:diffElem55");
	//	c1->Print("diffElem55_ind.png");
	//	DiffMatrix->Draw("index:diffElem46");
	//	c1->Print("diffElem46_ind.png");
	//	DiffMatrix->Draw("index:diffElem56");
	//	c1->Print("diffElem56_ind.png");
	//	DiffMatrix->Draw("index:diffElem66");
	//	c1->Print("diffElem66_ind.png");
	//	DiffMatrix->Draw("index:diffElem57");
	//	c1->Print("diffElem57_ind.png");
	//	DiffMatrix->Draw("index:diffElem67");
	//	c1->Print("diffElem67_ind.png");
	//	DiffMatrix->Draw("index:diffElem77");
	//	c1->Print("diffElem77_ind.png");

	/// Histograms.

	c1->SetLogy();

	TH1F h("h", "", 500, -1, 1);
	h->SetStats(0);

	TLatex* txt2 =new TLatex(-.9,1000, "#frac{new - old}{old} ");
	txt2->SetTextFont(132);
	txt2->SetTextSize(0.04);

	h->GetXaxis()->SetTitle("Gains (Relative Change)");
								 //  txt2->Draw();
	DiffGains->Draw("diffGains>>h");
	c1->Print("diffGains.png");

	h->GetXaxis()->SetTitle("Pedestals (Relative Change)");
								 //txt2->Draw();
	DiffPeds->Draw("diffPeds>>h");
	c1->Print("diffPeds.png");

	h->GetXaxis()->SetTitle("CrossTalk - Right (Relative Change)");
								 //        txt2->Draw();
	DiffXtalk->Draw("diffXtalkR>>h");
	c1->Print("diffXtalkR.png");

	h->GetXaxis()->SetTitle("CrossTalk - Right Int (Relative Change)");
								 //txt2->Draw();
	DiffXtalk->Draw("diffIntR>>h");
	c1->Print("diffIntR.png");

	h->GetXaxis()->SetTitle("CrossTalk - Left (Relative Change)");
								 // txt2->Draw();
	DiffXtalk->Draw("diffXtalkL>>h");
	c1->Print("diffXtalkL.png");

	h->GetXaxis()->SetTitle("CrossTalk - Left Int (Relative Change)");
								 //txt2->Draw();
	DiffXtalk->Draw("diffIntL>>h");
	c1->Print("diffIntL.png");

	//	DiffMatrix->Draw("diffElem33");
	//	c1->Print("diffElem33.png");
	//	DiffMatrix->Draw("diffElem34");
	//	c1->Print("diffElem34.png");
	//	DiffMatrix->Draw("diffElem44");
	//	c1->Print("diffElem44.png");
	//	DiffMatrix->Draw("diffElem35");
	//	c1->Print("diffElem35.png");
	//	DiffMatrix->Draw("diffElem45");
	//	c1->Print("diffElem45.png");
	//	DiffMatrix->Draw("diffElem55");
	//	c1->Print("diffElem55.png");
	//	DiffMatrix->Draw("diffElem46");
	//	c1->Print("diffElem46.png");
	//	DiffMatrix->Draw("diffElem56");
	//	c1->Print("diffElem56.png");
	//	DiffMatrix->Draw("diffElem66");
	//	c1->Print("diffElem66.png");
	//	DiffMatrix->Draw("diffElem57");
	//	c1->Print("diffElem57.png");
	//	DiffMatrix->Draw("diffElem67");
	//	c1->Print("diffElem67.png");
	//	DiffMatrix->Draw("diffElem77");
	//	c1->Print("diffElem77.png");

}
