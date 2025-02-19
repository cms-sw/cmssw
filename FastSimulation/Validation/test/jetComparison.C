{
  TH1F * skeleton = new TH1F("skeleton","skeleton", 100 , 0., 5.);
skeleton->SetMinimum(0.);
skeleton->SetMaximum(2.);
skeleton->Draw();
Graph->Draw("sameP");
}
