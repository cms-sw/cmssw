{
gROOT->Reset();
gROOT->Macro("init.C");

gSystem->Load("libChain");


// Chain chain1("Eff","ScanOut_nCrystals_PosCalc_Ecal/clustering_nCrystals_PosCalc_Ecal_-1/*.root");
// Chain chain2("Eff","ScanOut_nCrystals_PosCalc_Ecal/clustering_nCrystals_PosCalc_Ecal_9/*.root");
// Chain chain3("Eff","ScanOut_nCrystals_PosCalc_Ecal/clustering_nCrystals_PosCalc_Ecal_5/*.root");

Chain chain1("Eff","ScanOut_10Jan2007_161241/clustering_thresh_Ecal_Barrel_0.02/*.root");
Chain chain2("Eff","ScanOut_10Jan2007_161241/clustering_thresh_Ecal_Barrel_0.06/*.root");
Chain chain3("Eff","ScanOut_10Jan2007_161241/clustering_thresh_Ecal_Barrel_0.1/*.root");


// Chain chain2("Eff","Out_clustering_thresh_Ecal_Endcap_b_singlegamma_*/*_0.5.root");


ResidualFitter::SetCanvas(400,400);

ResidualFitter eri1("eri1","eri1",1,0,1,6,1,7,100,-0.03,0.03);
chain1.Draw("clusters_.phi-particles_.phi:particles_.e:particles_.eta>>eri1","@particles_.size()==1","goff");
eri1.SetAutoRange(3);
eri1.SetFitOptions("");
eri1.FitSlicesZ();
eri1.cd();
eri1_sigma.Draw("colz");
eri1.GetCanvas()->Iconify();


ResidualFitter eri2("eri2","eri2",1,0,1,6,1,7,100,-0.03,0.03);
chain2.Draw("clusters_.phi-particles_.phi:particles_.e:particles_.eta>>eri2","@particles_.size()==1","goff");
eri2.SetAutoRange(3);
eri2.SetFitOptions("");
eri2.FitSlicesZ();
eri2.cd();
eri2_sigma.Draw("colz");
eri2.GetCanvas()->Iconify();


ResidualFitter eri3("eri3","eri3",1,0,1,6,1,7,100,-0.03,0.03);
chain3.Draw("clusters_.phi-particles_.phi:particles_.e:particles_.eta>>eri3","@particles_.size()==1","goff");
eri3.SetAutoRange(3);
eri3.SetFitOptions("");
eri3.FitSlicesZ();
eri3.cd();
eri3_sigma.Draw("colz");
eri3.GetCanvas()->Iconify();

ResidualFitter erii("erii","erii",1,0,1,6,1,7,100,-0.03,0.03);
chain2.Draw("clustersIsland_.phi-particles_.phi:particles_.e:particles_.eta>>erii","@particles_.size()==1","goff");
erii.SetAutoRange(3);
erii.SetFitOptions("");
erii.FitSlicesZ();
erii.cd();
erii_sigma.Draw("colz");
erii.GetCanvas()->Iconify();




// ResidualFitter eri2("eri2","eri2",36,-3,3,1,0,10,100,-0.03,0.03);
// chain2.Draw("clusters_.eta-particles_.eta:particles_.e:particles_.eta>>eri2","@particles_.size()==1","goff");
// eri2.SetAutoRange(3);
// eri2.SetFitOptions("");
// eri2.FitSlicesZ();
// eri2.cd();
// eri2_sigma.Draw("colz");

TCanvas c3("c3","",600,600);


TH1D* hi = erii_sigma->ProjectionY();
TH1D* h1 = eri1_sigma->ProjectionY();
TH1D* h2 = eri2_sigma->ProjectionY();
TH1D* h3 = eri3_sigma->ProjectionY();

hi->SetStats(0);
hi->SetTitle(";E_{#gamma} (GeV);#sigma_{#phi}");
hi->GetYaxis()->SetNdivisions(5);
hi->GetYaxis()->SetTitleSize(0.07);
hi->GetXaxis()->SetTitleSize(0.05);

hi->SetLineWidth(2);
h1->SetLineWidth(2);
h2->SetLineWidth(2);
h3->SetLineWidth(2);

hi->SetLineColor(1);
h1->SetLineColor(2);
h2->SetLineColor(4);
h3->SetLineColor(5);


hi->Draw();
h1->Draw("same");
h2->Draw("same");
h3->Draw("same");

gPad->SetLeftMargin(0.15);
gPad->SetBottomMargin(0.12);

TLegend leg(0.66,0.66,0.88,0.88);
leg.AddEntry(hi, "Island","l");
leg.AddEntry(h1, "Eflow, chain 1","l");
leg.AddEntry(h2, "Eflow, chain 2","l");
leg.AddEntry(h3, "Eflow, chain 3","l");
leg.Draw();
}
