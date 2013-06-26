{
  c1.Clear();
  c1.Divide(3,3);

  c1_1.cd();
  hcharge1.Draw();
  gStyle->SetHistLineStyle(2);
  gStyle->SetHistLineColor(2);
  hcharge2->UseCurrentStyle();
  hcharge2.Draw("same");
  gStyle->SetHistLineStyle(3);
  gStyle->SetHistLineColor(3);
  hcharge3->UseCurrentStyle();
  hcharge3.Draw("same");

  c1_2.cd();
  hcols1.Draw();
  gStyle->SetHistLineStyle(2);
  gStyle->SetHistLineColor(2);
  hcols2->UseCurrentStyle();
  hcols2.Draw("same");
  gStyle->SetHistLineStyle(3);
  gStyle->SetHistLineColor(3);
  hcols3->UseCurrentStyle();
  hcols3.Draw("same");

  c1_3.cd();
  hrows1.Draw();
  gStyle->SetHistLineStyle(2);
  gStyle->SetHistLineColor(2);
  hrows2->UseCurrentStyle();
  hrows2.Draw("same");
  gStyle->SetHistLineStyle(3);
  gStyle->SetHistLineColor(3);
  hrows3->UseCurrentStyle();
  hrows3.Draw("same");

  c1_4.cd();
  hsize1->GetXaxis().SetRangeUser(0.,20.);
  hsize1.Draw();
  gStyle->SetHistLineStyle(2);
  gStyle->SetHistLineColor(2);
  hsize2->UseCurrentStyle();
  hsize2.Draw("same");
  gStyle->SetHistLineStyle(3);
  gStyle->SetHistLineColor(3);
  hsize3->UseCurrentStyle();
  hsize3.Draw("same");

  c1_5.cd();
  hsizex1.Draw();
  gStyle->SetHistLineStyle(2);
  gStyle->SetHistLineColor(2);
  hsizex2->UseCurrentStyle();
  hsizex2.Draw("same");
  gStyle->SetHistLineStyle(3);
  gStyle->SetHistLineColor(3);
  hsizex3->UseCurrentStyle();
  hsizex3.Draw("same");

  c1_6.cd();
  hsizey1.Draw();
  gStyle->SetHistLineStyle(2);
  gStyle->SetHistLineColor(2);
  hsizey2->UseCurrentStyle();
  hsizey2.Draw("same");
  gStyle->SetHistLineStyle(3);
  gStyle->SetHistLineColor(3);
  hsizey3->UseCurrentStyle();
  hsizey3.Draw("same");

  c1_7.cd();
  hclusPerDet1->GetXaxis().SetRangeUser(0.,10.);
  hclusPerDet1.Draw();
  gStyle->SetHistLineStyle(2);
  gStyle->SetHistLineColor(2);
  hclusPerDet2->UseCurrentStyle();
  hclusPerDet2.Draw("same");
  gStyle->SetHistLineStyle(3);
  gStyle->SetHistLineColor(3);
  hclusPerDet3->UseCurrentStyle();
  hclusPerDet3.Draw("same");

  c1_8.cd();
  hclusPerLay1->GetXaxis().SetRangeUser(0.,20.);
  hclusPerLay1.Draw();
  gStyle->SetHistLineStyle(2);
  gStyle->SetHistLineColor(2);
  hclusPerLay2->UseCurrentStyle();
  hclusPerLay2.Draw("same");
  gStyle->SetHistLineStyle(3);
  gStyle->SetHistLineColor(3);
  hclusPerLay3->UseCurrentStyle();
  hclusPerLay3.Draw("same");

  c1_9.cd();
  hdetz.Draw();

}
