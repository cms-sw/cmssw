{
eflow0->SetStats(0);
eflow0->SetTitle("");

eflow0->GetYaxis()->SetTitle("#varepsilon");
eflow0->GetXaxis()->SetTitle("E_{#gamma}");
eflow0->SetLineColor(2);
eflow0->SetLineWidth(2);


island1->SetStats(0);
island1->SetTitle("");


// island1->GetYaxis()->SetTitle("#varepsilon");
// island1->GetXaxis()->SetTitle("E_{#gamma}");

island1->SetLineColor(1);
island1->SetLineWidth(2);

TLegend leg(0.6,0.4,0.8,0.7);
leg.AddEntry(eflow0, "Eflow", "l");
leg.AddEntry(island1, "Island", "l");
leg.Draw();

gPad->Modified();


}
