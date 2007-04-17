{
gROOT->Macro("init.C");

Chain chain1("Eff","ScanOut_eb_seed_0.15_0.8/*root");

// Chain chain2("Eff","Out_clustering_thresh_Ecal_Endcap_b_singlegamma_*/*_0.5.root");


ResidualFitter::SetCanvas(400,400);

ResidualFitter eri1("eri1","eri1",36,-3,3,100,0,200,100,-0.05,0.05);
chain1.Draw("clusters_.eta-particles_.eta:particles_.e:particles_.eta>>eri1","@particles_.size()==1","goff");
eri1.SetAutoRange(3);
eri1.SetFitOptions("W");
eri1.FitSlicesZ();
eri1.cd();
eri1_sigma.Draw("lego2");


// ResidualFitter eri2("eri2","eri2",36,-3,3,1,0,10,100,-0.05,0.05);
// chain2.Draw("clusters_.eta-particles_.eta:particles_.e:particles_.eta>>eri2","@particles_.size()==1","goff");
// eri2.SetAutoRange(3);
// eri2.SetFitOptions("W");
// eri2.FitSlicesZ();
// eri2.cd();
// eri2_sigma.Draw("colz");

// TCanvas c3("ratio","",300,300);
// TH2D* ratio = (TH2D* ) eri2_sigma->Clone("ratio");
// ratio->Divide(eri1_sigma);
// ratio->Draw("box");



}
