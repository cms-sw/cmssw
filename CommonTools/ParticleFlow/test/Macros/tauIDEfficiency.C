{
TFile f("testPFPAT.root");

Events.Draw("recoPFTaus_pfRecoTauProducer__PFPAT.obj.pt()>>h1","recoPFTaus_pfRecoTauProducer__PFPAT.obj.pt()<80","");
Events.Draw("recoPFTaus_pfTauID__PFPAT.obj.pt()>>h2","","same");

h2.SetLineColor(4);
h2.SetLineWidth(3);
h2.SetFillStyle(3005);
h2.SetFillColor(4);
h1.SetStats(0);

h1.SetTitle("tau ID. p_{T}(tau) = 50 GeV, |#eta|<1; p_{T} tau jet (GeV)");

h2.SetLineWidth(3);

TLegend leg(0.35, 0.6, 0.89,0.89);
leg.AddEntry(&h1, "PFTaus","lf");
leg.AddEntry(&h2, "ConeIsolated PFTaus","lf");
leg.Draw();

gPad->SaveAs("tauID.eps");

}

