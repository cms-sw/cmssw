{
  TFile *_file0 = TFile::Open("pull_matched.root");

  //TID L1 x
  TH1F * TID_X_L1=(TH1F*)_file0->Get("mTIDpull_x_l1");
  TH1F * TID_X_L1_COMBINED=(TH1F*)_file0->Get("mTIDpull_x_l1_combined");
  TH1F * TID_X_L1_SIM=(TH1F*)_file0->Get("mTIDpull_x_l1_sim");

  //TID L2 x
  TH1F * TID_X_L2=(TH1F*)_file0->Get("mTIDpull_x_l2");
  TH1F * TID_X_L2_COMBINED=(TH1F*)_file0->Get("mTIDpull_x_l2_combined");
  TH1F * TID_X_L2_SIM=(TH1F*)_file0->Get("mTIDpull_x_l2_sim");

  //TID L1 y
  TH1F * TID_Y_L1=(TH1F*)_file0->Get("mTIDpull_y_l1");
  TH1F * TID_Y_L1_COMBINED=(TH1F*)_file0->Get("mTIDpull_y_l1_combined");
  TH1F * TID_Y_L1_SIM=(TH1F*)_file0->Get("mTIDpull_y_l1_sim");

  //TID L2 y
  TH1F * TID_Y_L2=(TH1F*)_file0->Get("mTIDpull_y_l2");
  TH1F * TID_Y_L2_COMBINED=(TH1F*)_file0->Get("mTIDpull_y_l2_combined");
  TH1F * TID_Y_L2_SIM=(TH1F*)_file0->Get("mTIDpull_y_l2_sim");

  //TEC L1 x
  TH1F * TEC_X_L1=(TH1F*)_file0->Get("mTECpull_x_l1");
  TH1F * TEC_X_L1_COMBINED=(TH1F*)_file0->Get("mTECpull_x_l1_combined");
  TH1F * TEC_X_L1_SIM=(TH1F*)_file0->Get("mTECpull_x_l1_sim");

  //TEC L2 x
  TH1F * TEC_X_L2=(TH1F*)_file0->Get("mTECpull_x_l2");
  TH1F * TEC_X_L2_COMBINED=(TH1F*)_file0->Get("mTECpull_x_l2_combined");
  TH1F * TEC_X_L2_SIM=(TH1F*)_file0->Get("mTECpull_x_l2_sim");

  //TEC L1 y
  TH1F * TEC_Y_L1=(TH1F*)_file0->Get("mTECpull_y_l1");
  TH1F * TEC_Y_L1_COMBINED=(TH1F*)_file0->Get("mTECpull_y_l1_combined");
  TH1F * TEC_Y_L1_SIM=(TH1F*)_file0->Get("mTECpull_y_l1_sim");

  //TEC L2 y
  TH1F * TEC_Y_L2=(TH1F*)_file0->Get("mTECpull_y_l2");
  TH1F * TEC_Y_L2_COMBINED=(TH1F*)_file0->Get("mTECpull_y_l2_combined");
  TH1F * TEC_Y_L2_SIM=(TH1F*)_file0->Get("mTECpull_y_l2_sim");


  //TEC L5 x
  TH1F * TEC_X_L5=(TH1F*)_file0->Get("mTECpull_x_l5");
  TH1F * TEC_X_L5_COMBINED=(TH1F*)_file0->Get("mTECpull_x_l5_combined");
  TH1F * TEC_X_L5_SIM=(TH1F*)_file0->Get("mTECpull_x_l5_sim");

  //TEC L5 y
  TH1F * TEC_Y_L5=(TH1F*)_file0->Get("mTECpull_y_l5");
  TH1F * TEC_Y_L5_COMBINED=(TH1F*)_file0->Get("mTECpull_y_l5_combined");
  TH1F * TEC_Y_L5_SIM=(TH1F*)_file0->Get("mTECpull_y_l5_sim");

  TCanvas *cTID=new TCanvas("TIDpull","TIDpull",1250,930);
  cTID->Divide(2,2);
  cTID->cd(1);
  TID_X_L1_SIM->Fit("gaus");
  TID_X_L1_COMBINED->Fit("gaus");
  TID_X_L1->Fit("gaus");
  TID_X_L1_SIM->SetLineColor(kRed);
  TID_X_L1_SIM->GetFunction("gaus")->SetLineColor(kRed);
  TID_X_L1_COMBINED->SetLineColor(kBlue);
  TID_X_L1_COMBINED->GetFunction("gaus")->SetLineColor(kBlue);
  gStyle->SetOptFit(1111);
  //  TID_X_L1_SIM->Draw();
  //  TID_X_L1_COMBINED->Draw("sames");
  TID_X_L1->Draw("sames");
  cTID->cd(2);
  TID_X_L2_SIM->Fit("gaus");
  TID_X_L2_COMBINED->Fit("gaus");
  TID_X_L2->Fit("gaus");
  TID_X_L2_SIM->SetLineColor(kRed);
  TID_X_L2_SIM->GetFunction("gaus")->SetLineColor(kRed);
  TID_X_L2_COMBINED->SetLineColor(kBlue);
  TID_X_L2_COMBINED->GetFunction("gaus")->SetLineColor(kBlue);
  gStyle->SetOptFit(1111);
  //  TID_X_L2_SIM->Draw();
  //  TID_X_L2_COMBINED->Draw("sames");
  TID_X_L2->Draw("sames");
  cTID->cd(3);
  TID_Y_L1_SIM->Fit("gaus");
  TID_Y_L1_COMBINED->Fit("gaus");
  TID_Y_L1->Fit("gaus");
  TID_Y_L1_SIM->SetLineColor(kRed);
  TID_Y_L1_SIM->GetFunction("gaus")->SetLineColor(kRed);
  TID_Y_L1_COMBINED->SetLineColor(kBlue);
  TID_Y_L1_COMBINED->GetFunction("gaus")->SetLineColor(kBlue);
  gStyle->SetOptFit(1111);
  //  TID_Y_L1_SIM->Draw();
  //  TID_Y_L1_COMBINED->Draw("sames");
  TID_Y_L1->Draw("sames");
  cTID->cd(4);
  TID_Y_L2_SIM->Fit("gaus");
  TID_Y_L2_COMBINED->Fit("gaus");
  TID_Y_L2->Fit("gaus");
  TID_Y_L2_SIM->SetLineColor(kRed);
  TID_Y_L2_SIM->GetFunction("gaus")->SetLineColor(kRed);
  TID_Y_L2_COMBINED->SetLineColor(kBlue);
  TID_Y_L2_COMBINED->GetFunction("gaus")->SetLineColor(kBlue);
  gStyle->SetOptFit(1111);
  //  TID_Y_L2_SIM->Draw();
  //  TID_Y_L2_COMBINED->Draw("sames");
  TID_Y_L2->Draw("sames"); 

  TCanvas *cTEC=new TCanvas("TECpull","TECpull",1250,930);
  cTEC->Divide(3,2);
  cTEC->cd(1);
  TEC_X_L1_SIM->Fit("gaus");
  TEC_X_L1_COMBINED->Fit("gaus");
  TEC_X_L1->Fit("gaus");
  TEC_X_L1_SIM->SetLineColor(kRed);
  TEC_X_L1_SIM->GetFunction("gaus")->SetLineColor(kRed);
  TEC_X_L1_COMBINED->SetLineColor(kBlue);
  TEC_X_L1_COMBINED->GetFunction("gaus")->SetLineColor(kBlue);
  gStyle->SetOptFit(1111);
  //  TEC_X_L1_SIM->Draw();
  //  TEC_X_L1_COMBINED->Draw("sames");
  TEC_X_L1->Draw("sames");

  cTEC->cd(2);
  TEC_X_L2_SIM->Fit("gaus");
  TEC_X_L2_COMBINED->Fit("gaus");
  TEC_X_L2->Fit("gaus");
  TEC_X_L2_SIM->SetLineColor(kRed);
  TEC_X_L2_SIM->GetFunction("gaus")->SetLineColor(kRed);
  TEC_X_L2_COMBINED->SetLineColor(kBlue);
  TEC_X_L2_COMBINED->GetFunction("gaus")->SetLineColor(kBlue);
  gStyle->SetOptFit(1111);
  // TEC_X_L2_SIM->Draw();
  // TEC_X_L2_COMBINED->Draw("sames");
  TEC_X_L2->Draw("sames");

  cTEC->cd(3);
  TEC_X_L5_SIM->Fit("gaus");
  TEC_X_L5_COMBINED->Fit("gaus");
  TEC_X_L5->Fit("gaus");
  TEC_X_L5_SIM->SetLineColor(kRed);
  TEC_X_L5_SIM->GetFunction("gaus")->SetLineColor(kRed);
  TEC_X_L5_COMBINED->SetLineColor(kBlue);
  TEC_X_L5_COMBINED->GetFunction("gaus")->SetLineColor(kBlue);
  gStyle->SetOptFit(1111);
  // TEC_X_L5_SIM->Draw();
  //  TEC_X_L5_COMBINED->Draw("sames");
  TEC_X_L5->Draw("sames");

  cTEC->cd(4);
  TEC_Y_L1_SIM->Fit("gaus");
  TEC_Y_L1_COMBINED->Fit("gaus");
  TEC_Y_L1->Fit("gaus");
  TEC_Y_L1_SIM->SetLineColor(kRed);
  TEC_Y_L1_SIM->GetFunction("gaus")->SetLineColor(kRed);
  TEC_Y_L1_COMBINED->SetLineColor(kBlue);
  TEC_Y_L1_COMBINED->GetFunction("gaus")->SetLineColor(kBlue);
  gStyle->SetOptFit(1111);
  //  TEC_Y_L1_SIM->Draw();
  //  TEC_Y_L1_COMBINED->Draw("sames");
  TEC_Y_L1->Draw("sames");
  cTEC->cd(5);
  TEC_Y_L2_SIM->Fit("gaus");
  TEC_Y_L2_COMBINED->Fit("gaus");
  TEC_Y_L2->Fit("gaus");
  TEC_Y_L2_SIM->SetLineColor(kRed);
  TEC_Y_L2_SIM->GetFunction("gaus")->SetLineColor(kRed);
  TEC_Y_L2_COMBINED->SetLineColor(kBlue);
  TEC_Y_L2_COMBINED->GetFunction("gaus")->SetLineColor(kBlue);
  gStyle->SetOptFit(1111);
  //  TEC_Y_L2_SIM->Draw();
  //  TEC_Y_L2_COMBINED->Draw("sames");
  TEC_Y_L2->Draw("sames"); 

  cTEC->cd(6);
  TEC_Y_L5_SIM->Fit("gaus");
  TEC_Y_L5_COMBINED->Fit("gaus");
  TEC_Y_L5->Fit("gaus");
  TEC_Y_L5_SIM->SetLineColor(kRed);
  TEC_Y_L5_SIM->GetFunction("gaus")->SetLineColor(kRed);
  TEC_Y_L5_COMBINED->SetLineColor(kBlue);
  TEC_Y_L5_COMBINED->GetFunction("gaus")->SetLineColor(kBlue);
  gStyle->SetOptFit(1111);
  //  TEC_Y_L5_SIM->Draw();
  //  TEC_Y_L5_COMBINED->Draw("sames");
  TEC_Y_L5->Draw("sames"); 
}
