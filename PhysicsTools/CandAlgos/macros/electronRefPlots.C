{
TH1D electronPt("electronPt", "p_{t} electrons", 
		20, 0, 100 );
TH1D electronEta("electronEta", "#eta electrons", 
		20, -3, 3 );
Events.Project("electronPt", "genParticleCandidates.data_[electronRefs.refVector_.items_.index_].pt()" );
Events.Project("electronEta", "genParticleCandidates.data_[electronRefs.refVector_.items_.index_].eta()" );
               
electronPt.SetLineWidth(2);
electronPt.SetLineColor(kRed);
electronPt.GetXaxis().SetTitle("p_{t} (GeV/c)");
electronEta.SetLineWidth(2);
electronEta.SetLineColor(kRed);
electronEta.GetXaxis().SetTitle("#eta");

TCanvas c;
electronPt.Draw();
c.SaveAs("electronPt_ref.eps");
electronEta.Draw();
c.SaveAs("electronEta_ref.eps");
}
