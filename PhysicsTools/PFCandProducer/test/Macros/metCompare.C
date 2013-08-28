{
TFile f("testPFPAT.root");

Events.Draw("recoMETs_pfMET__PFPAT.obj.pt_>>h1","recoMETs_pfMET__PFPAT.obj.pt_<100","");
Events.Draw("recoMETs_htMetIC5__PROD.obj.pt_>>h2","","same");

h1.SetLineColor(4);
h1.SetLineWidth(3);
h1.SetFillStyle(3005);
h1.SetFillColor(4);
h1.SetStats(0);

h1.SetTitle("MET, di-pion p_{T}=100 GeV, -5<#eta<5; MET (GeV)");

h2.SetLineWidth(3);

TLegend leg(0.35, 0.6, 0.89,0.89);
leg.AddEntry(&h1, "PFlow MET","lf");
leg.AddEntry(&h2, "recoMETs_htMetIC5","lf");
leg.Draw();

gPad->SaveAs("met.eps");

}

