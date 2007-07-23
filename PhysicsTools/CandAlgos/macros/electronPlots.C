{
TH1D electronPt("electronPt", "p_{t} electrons", 
		20, 0, 100 );
TH1D electronEta("electronEta", "#eta electrons", 
		20, -3, 3 );
std::string electronCut = "abs(genParticleCandidates.data_.pdgId())==11";;
Events.Project("electronPt", "genParticleCandidates.data_.pt()", 
               electronCut.c_str() );
Events.Project("electronEta", "genParticleCandidates.data_.eta()", 
               electronCut.c_str() );
               
electronPt.SetLineWidth(2);
electronPt.SetLineColor(kRed);
electronPt.GetXaxis().SetTitle("p_{t} (GeV/c)");
electronEta.SetLineWidth(2);
electronEta.SetLineColor(kRed);
electronEta.GetXaxis().SetTitle("#eta");

TCanvas c;
electronPt.Draw();
c.SaveAs("electronPt.eps");
electronEta.Draw();
c.SaveAs("electronEta.eps");
}
