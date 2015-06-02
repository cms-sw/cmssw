

{

   TChain* t = new TChain("akVs3CaloJetAnalyzer/t");
   TChain* t2 = new TChain("hiEvtAnalyzer/HiTree");

   t->Add("/mnt/hadoop/cms/store/user/pkurt/Hydjet1p8_TuneDrum_Quenched_MinBias_2760GeV/HydjetDrum_Pyquen_Dijet30_FOREST_Track8_Jet24_FixedPtHatJES_v0/29413e33cdbed4171760b9d0eb97af9d/Hi*.root");
   t2->Add("/mnt/hadoop/cms/store/user/pkurt/Hydjet1p8_TuneDrum_Quenched_MinBias_2760GeV/HydjetDrum_Pyquen_Dijet30_FOREST_Track8_Jet24_FixedPtHatJES_v0/29413e33cdbed4171760b9d0eb97af9d/Hi*.root");

  t->AddFriend(t2);

  new TCanvas();

  //  t->Draw("jtpt/refpt:refpt","hiBin < 4 && abs(refeta) < 2 && refpt > 0","colz");
  //  t->Draw("jtpt/refpt:refpt","hiBin < 4 && abs(refeta) < 2 && refpt > 0","prof same");

  new TCanvas();

  t->Draw("jtpt/refpt:hiBin","refpt > 40 && abs(refeta) < 2 && refpt > 0","colz");
  t->Draw("jtpt/refpt:hiBin","refpt > 40 && abs(refeta) < 2 && refpt > 0","prof same");


}

