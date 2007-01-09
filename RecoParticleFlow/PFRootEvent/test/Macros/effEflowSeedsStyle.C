{

gSystem->Load("libChain");
gROOT->LoadMacro("effIneff.C");
Init("ScanOut_eb_seed_0.2_0.4/lowE.root");
EffPlateau2(chain,0,"eflow0.2",0.05,1);
Init("ScanOut_eb_seed_0.15_0.8/lowE.root");
EffPlateau2(chain,0,"eflow0.15",0.05,1);
Init("ScanOut_eb_seed_0.3_0.6/lowE.root");
EffPlateau2(chain,0,"eflow0.3",0.05,1);


TCanvas c("c","",600,600);
c.cd();

TH1F* eflow02_0 = (TH1F*) gDirectory->Get("eflow0.20");
TH1F* eflow015_0 = (TH1F*) gDirectory->Get("eflow0.150");
TH1F* eflow03_0 = (TH1F*) gDirectory->Get("eflow0.30");

eflow02_0->SetStats(0);
eflow02_0->SetTitle("");
eflow02_0->GetYaxis()->SetTitle("#varepsilon");  
eflow02_0.GetYaxis()->SetTitleSize(0.08);
eflow02_0.GetYaxis()->SetTitleOffset(0.5);
eflow02_0.GetXaxis()->SetTitleSize(0.05);
eflow02_0.GetXaxis()->SetTitleOffset(0.8);

eflow02_0->GetXaxis()->SetTitle("E_{#gamma}");
eflow02_0->SetLineColor(2);
eflow02_0->SetLineWidth(2);


eflow015_0->SetStats(0);
eflow015_0->SetTitle("");
eflow015_0->SetLineColor(4);
eflow015_0->SetLineWidth(2);

eflow03_0->SetStats(0);
eflow03_0->SetTitle("");
eflow03_0->SetLineColor(5);
eflow03_0->SetLineWidth(2);



TLegend leg(0.36,0.138,0.88,0.44);
leg.AddEntry(eflow015_0, "Eflow, T_{seed}=150 MeV", "l");
leg.AddEntry(eflow02_0, "Eflow, T_{seed}=200 MeV", "l");
leg.AddEntry(eflow03_0, "Eflow, T_{seed}=300 MeV", "l");


eflow02_0->Draw();
eflow03_0->Draw("same");
eflow015_0->Draw("same");

leg.Draw();

TLine l;
l.SetLineStyle(2);

l.DrawLine(0, 1, 5, 1);

gPad->Modified();

}
