{

gSystem->Load("libChain");
gROOT->LoadMacro("effIneff.C");
// Init("ScanOut_eb_seed_0.2_0.4/out_singlegamma_repf_*.root");
Init("ScanOut_eb_seed_0.15_0.8/out_singlegamma_repf_*.root");
Impurity(0,0.1,1,"impurity_015");
Init("ScanOut_eb_seed_0.2_0.4/out_singlegamma_repf_*.root");
Impurity(0,0.1,1,"impurity_02");
Init("ScanOut_eb_seed_0.3_0.6/out_singlegamma_repf_*.root");
Impurity(0,0.1,1,"impurity_03");


TCanvas c("c","",600,600);
c.cd();

impurity_015_eflow->ProjectionX("impeflow015",3,-1);
impeflow015.Divide(impurity_015_ref);

impurity_02_eflow->ProjectionX("impeflow02",3,-1);
impeflow02.Divide(impurity_02_ref);

impurity_03_eflow->ProjectionX("impeflow03",3,-1);
impeflow03.Divide(impurity_03_ref);

impeflow015->SetTitle("");
impeflow015->GetYaxis()->SetTitle("Impurity (#DeltaR=0.1)");  
impeflow015->GetYaxis()->SetTitleSize(0.05);
impeflow015->GetYaxis()->SetTitleOffset(1.2);
impeflow015->GetXaxis()->SetTitleSize(0.05);
impeflow015->GetXaxis()->SetTitleOffset(0.95);
impeflow015->GetXaxis()->SetTitle("E_{#gamma}");  

impeflow015->SetStats(0);
impeflow015->SetLineColor(4);
impeflow015->SetLineWidth(2);

impeflow02->SetLineColor(2);
impeflow02->SetLineWidth(2);

impeflow03->SetLineColor(5);
impeflow03->SetLineWidth(2);


impeflow015->SetMaximum(0.5);
impeflow015->SetMinimum(0.);

impeflow015->Draw();
impeflow02->Draw("same");
impeflow03->Draw("same");



// impeflow->SetStats(0);
// impeflow->SetLineColor(2);
// impeflow->SetLineWidth(2);


// impeflow->Draw("same");

TLegend leg(0.16,0.69,0.60,0.88);
leg.AddEntry(impeflow015, "Eflow, T_{seed}=150 MeV", "l");
leg.AddEntry(impeflow02, "Eflow, T_{seed}=200 MeV", "l");
leg.AddEntry(impeflow03, "Eflow, T_{seed}=300 MeV", "l");
leg.Draw();

gPad->SetLeftMargin(0.13);
gPad->Modified();
}
