{

TH1F* eflow02_0 = (TH1F*) gDirectory->Get("eflow0.2_0");
TH1F* eflow015_0 = (TH1F*) gDirectory->Get("eflow0.15_0");
TH1F* eflow03_0 = (TH1F*) gDirectory->Get("eflow0.3_0");

eflow02_0->SetStats(0);
eflow02_0->SetTitle("");
eflow015_0->GetYaxis()->SetTitle("#varepsilon");  
eflow015_0.GetYaxis()->SetTitleSize(0.08);
eflow015_0.GetYaxis()->SetTitleOffset(0.5);
eflow015_0.GetXaxis()->SetTitleSize(0.05);
eflow015_0.GetXaxis()->SetTitleOffset(0.8);

eflow015_0->GetXaxis()->SetTitle("E_{#gamma}");
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



TLegend leg(0.5,0.4,0.8,0.7);
leg.AddEntry(eflow015_0, "Eflow, T_{seed}=150 MeV", "l");
leg.AddEntry(eflow02_0, "Eflow, T_{seed}=200 MeV", "l");
leg.AddEntry(eflow03_0, "Eflow, T_{seed}=300 MeV", "l");


leg.Draw();

TLine l;
l.SetLineStyle(2);

l.DrawLine(0, 1, 5, 1);

gPad->Modified();

}
