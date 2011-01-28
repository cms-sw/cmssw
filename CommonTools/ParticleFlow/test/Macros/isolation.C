{


gSystem->Load("libFWCoreFWLite.so");
AutoLibraryLoader::enable();
gSystem->Load("libCintex.so");
ROOT::Cintex::Cintex::Enable();
TFile f("testPFPAT.root");

Events.Draw("recoIsolatedPFCandidates_pfPionsIsolation__PFPAT.obj.isolation_>>h2");

Events.Draw("recoIsolatedPFCandidates_pfLeptonsPtGt5Isolation__PFPAT.obj.isolation_>>h1","","same");

h1.SetLineColor(4);
h1.SetLineWidth(3);
h1.SetFillStyle(3005);
h1.SetFillColor(4);

h2.SetTitle("H to ZZ to 4 mu. Isolation cone 0.2; p_{T} fraction");
h2.SetStats(0);
h2.SetLineWidth(3);

TLegend leg(0.35, 0.6, 0.89,0.89);
leg.AddEntry(&h1, "Muon PFCandidates","lf");
leg.AddEntry(&h2, "Pion PFCandidates","lf");
leg.Draw();

gPad->SetLogy();
gPad->SaveAs("isolation.eps");
}
