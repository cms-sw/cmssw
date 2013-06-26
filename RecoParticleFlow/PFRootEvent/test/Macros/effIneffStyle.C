{

TCanvas c("c","",600,600);
c.cd();

eflow0->SetStats(0);
eflow0->SetTitle("");

eflow0->GetYaxis()->SetTitle("#varepsilon");  
eflow0.GetYaxis()->SetTitleSize(0.08);
eflow0.GetYaxis()->SetTitleOffset(0.5);
eflow0.GetXaxis()->SetTitleSize(0.05);
eflow0.GetXaxis()->SetTitleOffset(0.8);

eflow0->GetXaxis()->SetTitle("E_{#gamma}");
eflow0->SetLineColor(2);
eflow0->SetLineWidth(2);


island1->SetStats(0);
island1->SetTitle("");


// island1->GetYaxis()->SetTitle("#varepsilon");
// island1->GetXaxis()->SetTitle("E_{#gamma}");

island1->SetLineColor(1);
island1->SetLineWidth(2);

TLegend leg(0.41,0.16,0.88,0.30);
leg.AddEntry(eflow0, "Eflow, T_{seed}=200 MeV", "l");
leg.AddEntry(island1, "Island", "l");

bool famos = false;
if(famos) {
  gSystem->Load("libTextTree");
  TextTree t("/afs/cern.ch/user/c/cbern/scratch0/Logbook_CMS/Analysis/EFLOW/ECAL_clustering/SinglePhotons/Efficiency_Purity/Barrel/effpur.dat","xxx");
  
  bool comp = false;
  
  t.SetLineWidth(3);
  t.Draw("c4:c1","c0==0.2","goff");
  // t.Draw("c2:c0","c1==0.3","plsame");
  TGraph *gra = new TGraph(t.GetSelectedRows(),
                             t.GetV2(), t.GetV1());
  gra->SetLineWidth(3);
  gra->SetMarkerStyle(20);
  gra->SetMarkerColor(1);
  gra->Draw("psame");

  t.Draw("c2:c1","c0==0.2","goff");
  TGraph *grb = new TGraph(t.GetSelectedRows(),
			   t.GetV2(), t.GetV1());
  grb->SetMarkerStyle(20);
  grb->SetLineWidth(3);
  grb->SetMarkerColor(2);
  grb->Draw("psame");
  
  leg->AddEntry(grb, "Eflow (FAMOS)", "p");
  leg->AddEntry(gra, "Island (FAMOS)", "p");
}

eflow0->Draw();
island1->Draw("same");
leg.Draw();

TLine l;
l.SetLineStyle(2);

l.DrawLine(0, 1, 5, 1);

gPad->Modified();

}
