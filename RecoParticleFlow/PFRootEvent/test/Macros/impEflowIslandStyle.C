{
gSystem->Load("libChain");
gROOT->LoadMacro("effIneff.C");

Init("ScanOut_eb_seed_0.2_0.4/out_singlegamma_repf_*.root");
// Init("ScanOut_eb_seed_0.15_0.8/out_singlegamma_repf_*.root");
Impurity(0,0.1,1,"impurity");

TCanvas c("c","",600,600);
c.cd();


impurity_island->ProjectionX("impisland",3,-1);
impurity_island->ProjectionX("refisland");

impurity_eflow->ProjectionX("impeflow",3,-1);
impurity_eflow->ProjectionX("refeflow");

impeflow.Divide(impurity_ref);
impisland.Divide(impurity_ref);

impeflow->SetStats(0);
impeflow->SetLineColor(2);
impeflow->SetLineWidth(2);

impisland->SetTitle("");
impisland->GetYaxis()->SetTitle("Impurity (#DeltaR=0.1)");  
impisland->GetYaxis()->SetTitleSize(0.05);
impisland->GetYaxis()->SetTitleOffset(1.2);
impisland->GetXaxis()->SetTitleSize(0.05);
impisland->GetXaxis()->SetTitleOffset(0.95);
impisland->GetXaxis()->SetTitle("E_{#gamma}");  

impisland->SetStats(0);
impisland->SetLineColor(1);
impisland->SetLineWidth(2);

impisland->SetMaximum(0.5);
impisland->SetMinimum(0.);

impisland->Draw();
impeflow->Draw("same");

TLegend leg(0.16,0.69,0.60,0.88);
leg.AddEntry(impisland, "Island", "l");
leg.AddEntry(impeflow, "Eflow, T_{seed}=200 MeV", "l");
leg.Draw();

gPad->SetLeftMargin(0.13);
gPad->Modified();
}
