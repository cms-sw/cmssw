{
  TFile *_file0 = TFile::Open("pull_matched.root");

  //TIB L1 x
  TH1F * TIB_X_L1=(TH1F*)_file0->Get("mTIBpull_x_l1");
  TH1F * TIB_X_L1_COMBINED=(TH1F*)_file0->Get("mTIBpull_x_l1_combined");
  TH1F * TIB_X_L1_SIM=(TH1F*)_file0->Get("mTIBpull_x_l1_sim");

  //TIB L2 x
  TH1F * TIB_X_L2=(TH1F*)_file0->Get("mTIBpull_x_l2");
  TH1F * TIB_X_L2_COMBINED=(TH1F*)_file0->Get("mTIBpull_x_l2_combined");
  TH1F * TIB_X_L2_SIM=(TH1F*)_file0->Get("mTIBpull_x_l2_sim");

  //TIB L1 y
  TH1F * TIB_Y_L1=(TH1F*)_file0->Get("mTIBpull_y_l1");
  TH1F * TIB_Y_L1_COMBINED=(TH1F*)_file0->Get("mTIBpull_y_l1_combined");
  TH1F * TIB_Y_L1_SIM=(TH1F*)_file0->Get("mTIBpull_y_l1_sim");

  //TIB L2 y
  TH1F * TIB_Y_L2=(TH1F*)_file0->Get("mTIBpull_y_l2");
  TH1F * TIB_Y_L2_COMBINED=(TH1F*)_file0->Get("mTIBpull_y_l2_combined");
  TH1F * TIB_Y_L2_SIM=(TH1F*)_file0->Get("mTIBpull_y_l2_sim");

  //TOB L1 x
  TH1F * TOB_X_L1=(TH1F*)_file0->Get("mTOBpull_x_l1");
  TH1F * TOB_X_L1_COMBINED=(TH1F*)_file0->Get("mTOBpull_x_l1_combined");
  TH1F * TOB_X_L1_SIM=(TH1F*)_file0->Get("mTOBpull_x_l1_sim");

  //TOB L2 x
  TH1F * TOB_X_L2=(TH1F*)_file0->Get("mTOBpull_x_l2");
  TH1F * TOB_X_L2_COMBINED=(TH1F*)_file0->Get("mTOBpull_x_l2_combined");
  TH1F * TOB_X_L2_SIM=(TH1F*)_file0->Get("mTOBpull_x_l2_sim");

  //TOB L1 y
  TH1F * TOB_Y_L1=(TH1F*)_file0->Get("mTOBpull_y_l1");
  TH1F * TOB_Y_L1_COMBINED=(TH1F*)_file0->Get("mTOBpull_y_l1_combined");
  TH1F * TOB_Y_L1_SIM=(TH1F*)_file0->Get("mTOBpull_y_l1_sim");

  //TOB L2 y
  TH1F * TOB_Y_L2=(TH1F*)_file0->Get("mTOBpull_y_l2");
  TH1F * TOB_Y_L2_COMBINED=(TH1F*)_file0->Get("mTOBpull_y_l2_combined");
  TH1F * TOB_Y_L2_SIM=(TH1F*)_file0->Get("mTOBpull_y_l2_sim");

  TCanvas *cTIB=new TCanvas("TIBpull","TIBpull",1250,930);
  cTIB->Divide(2,2);
  cTIB->cd(1);
  TIB_X_L1_SIM->Fit("gaus");
  TIB_X_L1_COMBINED->Fit("gaus");
  TIB_X_L1->Fit("gaus");
  TIB_X_L1_SIM->SetLineColor(kRed);
  TIB_X_L1_SIM->GetFunction("gaus")->SetLineColor(kRed);
  TIB_X_L1_COMBINED->SetLineColor(kBlue);
  TIB_X_L1_COMBINED->GetFunction("gaus")->SetLineColor(kBlue);
  gStyle->SetOptFit(1111);
  //  TIB_X_L1_SIM->Draw();
  //  TIB_X_L1_COMBINED->Draw("sames");
  //  TIB_X_L1->Draw("sames");
  TIB_X_L1->Draw("sames");
  cTIB->cd(2);
  TIB_X_L2_SIM->Fit("gaus");
  TIB_X_L2_COMBINED->Fit("gaus");
  TIB_X_L2->Fit("gaus");
  TIB_X_L2_SIM->SetLineColor(kRed);
  TIB_X_L2_SIM->GetFunction("gaus")->SetLineColor(kRed);
  TIB_X_L2_COMBINED->SetLineColor(kBlue);
  TIB_X_L2_COMBINED->GetFunction("gaus")->SetLineColor(kBlue);
  gStyle->SetOptFit(1111);
  //  TIB_X_L2_SIM->Draw();
  //  TIB_X_L2_COMBINED->Draw("sames");
  TIB_X_L2->Draw("sames");
  cTIB->cd(3);
  TIB_Y_L1_SIM->Fit("gaus");
  TIB_Y_L1_COMBINED->Fit("gaus");
  TIB_Y_L1->Fit("gaus");
  TIB_Y_L1_SIM->SetLineColor(kRed);
  TIB_Y_L1_SIM->GetFunction("gaus")->SetLineColor(kRed);
  TIB_Y_L1_COMBINED->SetLineColor(kBlue);
  TIB_Y_L1_COMBINED->GetFunction("gaus")->SetLineColor(kBlue);
  gStyle->SetOptFit(1111);
  //  TIB_Y_L1_SIM->Draw();
  //  TIB_Y_L1_COMBINED->Draw("sames");
  TIB_Y_L1->Draw("sames");
  cTIB->cd(4);
  TIB_Y_L2_SIM->Fit("gaus");
  TIB_Y_L2_COMBINED->Fit("gaus");
  TIB_Y_L2->Fit("gaus");
  TIB_Y_L2_SIM->SetLineColor(kRed);
  TIB_Y_L2_SIM->GetFunction("gaus")->SetLineColor(kRed);
  TIB_Y_L2_COMBINED->SetLineColor(kBlue);
  TIB_Y_L2_COMBINED->GetFunction("gaus")->SetLineColor(kBlue);
  gStyle->SetOptFit(1111);
  //  TIB_Y_L2_SIM->Draw();
  //  TIB_Y_L2_COMBINED->Draw("sames");
  TIB_Y_L2->Draw("sames"); 

  TCanvas *cTOB=new TCanvas("TOBpull","TOBpull",1250,930);
  cTOB->Divide(2,2);
  cTOB->cd(1);
  TOB_X_L1_SIM->Fit("gaus");
  TOB_X_L1_COMBINED->Fit("gaus");
  TOB_X_L1->Fit("gaus");
  TOB_X_L1_SIM->SetLineColor(kRed);
  TOB_X_L1_SIM->GetFunction("gaus")->SetLineColor(kRed);
  TOB_X_L1_COMBINED->SetLineColor(kBlue);
  TOB_X_L1_COMBINED->GetFunction("gaus")->SetLineColor(kBlue);
  gStyle->SetOptFit(1111);
  //  TOB_X_L1_SIM->Draw();
  //  TOB_X_L1_COMBINED->Draw("sames");
  TOB_X_L1->Draw("sames");
  cTOB->cd(2);
  TOB_X_L2_SIM->Fit("gaus");
  TOB_X_L2_COMBINED->Fit("gaus");
  TOB_X_L2->Fit("gaus");
  TOB_X_L2_SIM->SetLineColor(kRed);
  TOB_X_L2_SIM->GetFunction("gaus")->SetLineColor(kRed);
  TOB_X_L2_COMBINED->SetLineColor(kBlue);
  TOB_X_L2_COMBINED->GetFunction("gaus")->SetLineColor(kBlue);
  gStyle->SetOptFit(1111);
  // TOB_X_L2_SIM->Draw();
  //  TOB_X_L2_COMBINED->Draw("sames");
    TOB_X_L2->Draw("sames");
  cTOB->cd(3);
  TOB_Y_L1_SIM->Fit("gaus");
  TOB_Y_L1_COMBINED->Fit("gaus");
  TOB_Y_L1->Fit("gaus");
  TOB_Y_L1_SIM->SetLineColor(kRed);
  TOB_Y_L1_SIM->GetFunction("gaus")->SetLineColor(kRed);
  TOB_Y_L1_COMBINED->SetLineColor(kBlue);
  TOB_Y_L1_COMBINED->GetFunction("gaus")->SetLineColor(kBlue);
  gStyle->SetOptFit(1111);
  //  TOB_Y_L1_SIM->Draw();
  //  TOB_Y_L1_COMBINED->Draw("sames");
  TOB_Y_L1->Draw("sames");
  cTOB->cd(4);
  TOB_Y_L2_SIM->Fit("gaus");
  TOB_Y_L2_COMBINED->Fit("gaus");
  TOB_Y_L2->Fit("gaus");
  TOB_Y_L2_SIM->SetLineColor(kRed);
  TOB_Y_L2_SIM->GetFunction("gaus")->SetLineColor(kRed);
  TOB_Y_L2_COMBINED->SetLineColor(kBlue);
  TOB_Y_L2_COMBINED->GetFunction("gaus")->SetLineColor(kBlue);
  gStyle->SetOptFit(1111);
  //  TOB_Y_L2_SIM->Draw();
  //  TOB_Y_L2_COMBINED->Draw("sames");
  TOB_Y_L2->Draw("sames"); 
}
