{
  int NBINS=32;
  TFile *resfile1=new TFile("resolution_2.5.root");
  TFile *resfile2=new TFile("resolution_3.5.root");
  TFile *resfile3=new TFile("resolution_4.root");

  float tibrms1[NBINS],tobrms1[NBINS],tidrms1[NBINS],tecrms1[NBINS];
  float tibrmserror1[NBINS],tobrmserror1[NBINS],tidrmserror1[NBINS],tecrmserror1[NBINS];

  float tibrms2[NBINS],tobrms2[NBINS],tidrms2[NBINS],tecrms2[NBINS];
  float tibrmserror2[NBINS],tobrmserror2[NBINS],tidrmserror2[NBINS],tecrmserror2[NBINS];

  float tibrms3[NBINS],tobrms3[NBINS],tidrms3[NBINS],tecrms3[NBINS];
  float tibrmserror3[NBINS],tobrmserror3[NBINS],tidrmserror3[NBINS],tecrmserror3[NBINS];

  TH1F *tibres1[NBINS],*tobres1[NBINS],*tidres1[NBINS],*tecres1[NBINS];
  TH1F *allres1[NBINS];
  float allrms1[NBINS],allrmserror1[NBINS];

  TH1F *tibres2[NBINS],*tobres2[NBINS],*tidres2[NBINS],*tecres2[NBINS];
  TH1F *allres2[NBINS];
  float allrms2[NBINS],allrmserror2[NBINS];

  TH1F *tibres3[NBINS],*tobres3[NBINS],*tidres3[NBINS],*tecres3[NBINS];
  TH1F *allres3[NBINS];
  float allrms3[NBINS],allrmserror3[NBINS];

  float proj[NBINS],proje[NBINS];
  TF1 *newparam_old=new TF1("error_param_old","1/sqrt(12)",0.,8);
  for(int i=0;i<NBINS;i++){
    tibres1[i]=(TH1F*)resfile1->Get(Form("TIBres_%f-%f",i*8./NBINS,i*8./NBINS+8./NBINS));
    tobres1[i]=(TH1F*)resfile1->Get(Form("TOBres_%f-%f",i*8./NBINS,i*8./NBINS+8./NBINS));
    tidres1[i]=(TH1F*)resfile1->Get(Form("TIDres_%f-%f",i*8./NBINS,i*8./NBINS+8./NBINS));
    tecres1[i]=(TH1F*)resfile1->Get(Form("TECres_%f-%f",i*8./NBINS,i*8./NBINS+8./NBINS));
    allres1[i]=(TH1F*)tibres1[i]->Clone(Form("Allres1_%f-%f",i*8./NBINS,i*8./NBINS+8./NBINS));
    allres1[i]->Add(tobres1[i]);
    allres1[i]->Add(tidres1[i]);
    allres1[i]->Add(tecres1[i]);
    tibres2[i]=(TH1F*)resfile2->Get(Form("TIBres_%f-%f",i*8./NBINS,i*8./NBINS+8./NBINS));
    tobres2[i]=(TH1F*)resfile2->Get(Form("TOBres_%f-%f",i*8./NBINS,i*8./NBINS+8./NBINS));
    tidres2[i]=(TH1F*)resfile2->Get(Form("TIDres_%f-%f",i*8./NBINS,i*8./NBINS+8./NBINS));
    tecres2[i]=(TH1F*)resfile2->Get(Form("TECres_%f-%f",i*8./NBINS,i*8./NBINS+8./NBINS));
    allres2[i]=(TH1F*)tibres2[i]->Clone(Form("Allres2_%f-%f",i*8./NBINS,i*8./NBINS+8./NBINS));
    allres2[i]->Add(tobres2[i]);
    allres2[i]->Add(tidres2[i]);
    allres2[i]->Add(tecres2[i]);
    tibres3[i]=(TH1F*)resfile3->Get(Form("TIBres_%f-%f",i*8./NBINS,i*8./NBINS+8./NBINS));
    tobres3[i]=(TH1F*)resfile3->Get(Form("TOBres_%f-%f",i*8./NBINS,i*8./NBINS+8./NBINS));
    tidres3[i]=(TH1F*)resfile3->Get(Form("TIDres_%f-%f",i*8./NBINS,i*8./NBINS+8./NBINS));
    tecres3[i]=(TH1F*)resfile3->Get(Form("TECres_%f-%f",i*8./NBINS,i*8./NBINS+8./NBINS));
    allres3[i]=(TH1F*)tibres3[i]->Clone(Form("Allres3_%f-%f",i*8./NBINS,i*8./NBINS+8./NBINS));
    allres3[i]->Add(tobres3[i]);
    allres3[i]->Add(tidres3[i]);
    allres3[i]->Add(tecres3[i]);

    //    float range=newparam_old->Eval(float(i)/4+0.125)*sqrt(30);

    float range=10;
    tibres1[i]->GetXaxis()->SetRangeUser(-range,range);
    tidres1[i]->GetXaxis()->SetRangeUser(-range,range);
    tobres1[i]->GetXaxis()->SetRangeUser(-range,range);
    tecres1[i]->GetXaxis()->SetRangeUser(-range,range);
    allres1[i]->GetXaxis()->SetRangeUser(-range,range);
    tibrms1[i]=tibres1[i]->GetRMS();
    tibrmserror1[i]=tibres1[i]->GetRMSError();
    tobrms1[i]=tobres1[i]->GetRMS();
    tobrmserror1[i]=tobres1[i]->GetRMSError();
    tidrms1[i]=tidres1[i]->GetRMS();
    tidrmserror1[i]=tidres1[i]->GetRMSError();
    tecrms1[i]=tecres1[i]->GetRMS();
    tecrmserror1[i]=tecres1[i]->GetRMSError();
    allrms1[i]=allres1[i]->GetRMS();
    allrmserror1[i]=allres1[i]->GetRMSError();
    tibres2[i]->GetXaxis()->SetRangeUser(-range,range);
    tidres2[i]->GetXaxis()->SetRangeUser(-range,range);
    tobres2[i]->GetXaxis()->SetRangeUser(-range,range);
    tecres2[i]->GetXaxis()->SetRangeUser(-range,range);
    allres2[i]->GetXaxis()->SetRangeUser(-range,range);
    tibrms2[i]=tibres2[i]->GetRMS();
    tibrmserror2[i]=tibres2[i]->GetRMSError();
    tobrms2[i]=tobres2[i]->GetRMS();
    tobrmserror2[i]=tobres2[i]->GetRMSError();
    tidrms2[i]=tidres2[i]->GetRMS();
    tidrmserror2[i]=tidres2[i]->GetRMSError();
    tecrms2[i]=tecres2[i]->GetRMS();
    tecrmserror2[i]=tecres2[i]->GetRMSError();
    allrms2[i]=allres2[i]->GetRMS();
    allrmserror2[i]=allres2[i]->GetRMSError();
    tibres3[i]->GetXaxis()->SetRangeUser(-range,range);
    tidres3[i]->GetXaxis()->SetRangeUser(-range,range);
    tobres3[i]->GetXaxis()->SetRangeUser(-range,range);
    tecres3[i]->GetXaxis()->SetRangeUser(-range,range);
    allres3[i]->GetXaxis()->SetRangeUser(-range,range);
    tibrms3[i]=tibres3[i]->GetRMS();
    tibrmserror3[i]=tibres3[i]->GetRMSError();
    tobrms3[i]=tobres3[i]->GetRMS();
    tobrmserror3[i]=tobres3[i]->GetRMSError();
    tidrms3[i]=tidres3[i]->GetRMS();
    tidrmserror3[i]=tidres3[i]->GetRMSError();
    tecrms3[i]=tecres3[i]->GetRMS();
    tecrmserror3[i]=tecres3[i]->GetRMSError();
    allrms3[i]=allres3[i]->GetRMS();
    allrmserror3[i]=allres3[i]->GetRMSError();

    proj[i]=float(i)/4+0.125; proje[i]=0.125;
  }
    TGraphErrors *TIB1=new TGraphErrors(NBINS,proj,tibrms1,proje,tibrmserror1);
    TGraphErrors *TOB1=new TGraphErrors(NBINS,proj,tobrms1,proje,tobrmserror1);
    TGraphErrors *TID1=new TGraphErrors(NBINS,proj,tidrms1,proje,tidrmserror1);
    TGraphErrors *TEC1=new TGraphErrors(NBINS,proj,tecrms1,proje,tecrmserror1);
    TGraphErrors *ALL1=new TGraphErrors(NBINS,proj,allrms1,proje,allrmserror1);

    TGraphErrors *TIB2=new TGraphErrors(NBINS,proj,tibrms2,proje,tibrmserror2);
    TGraphErrors *TOB2=new TGraphErrors(NBINS,proj,tobrms2,proje,tobrmserror2);
    TGraphErrors *TID2=new TGraphErrors(NBINS,proj,tidrms2,proje,tidrmserror2);
    TGraphErrors *TEC2=new TGraphErrors(NBINS,proj,tecrms2,proje,tecrmserror2);
    TGraphErrors *ALL2=new TGraphErrors(NBINS,proj,allrms2,proje,allrmserror2);

    TGraphErrors *TIB3=new TGraphErrors(NBINS,proj,tibrms3,proje,tibrmserror3);
    TGraphErrors *TOB3=new TGraphErrors(NBINS,proj,tobrms3,proje,tobrmserror3);
    TGraphErrors *TID3=new TGraphErrors(NBINS,proj,tidrms3,proje,tidrmserror3);
    TGraphErrors *TEC3=new TGraphErrors(NBINS,proj,tecrms3,proje,tecrmserror3);
    TGraphErrors *ALL3=new TGraphErrors(NBINS,proj,allrms3,proje,allrmserror3);

    TCanvas * plot1=new TCanvas("resolution_subdetectors","resolution_subdetectors");
    plot1->Divide(2,2);
    plot1->cd(1);
    TIB1->SetMaximum(1);
    TIB1->SetMinimum(0.);
    TIB1->GetHistogram()->GetXaxis()->SetTitle("Track projection");
    TIB1->GetHistogram()->SetTitle("TIB");
    TIB1->Draw("ap");
    TIB2->SetMarkerColor(2);
    TIB2->SetLineColor(2);
    TIB3->SetMarkerColor(3);
    TIB3->SetLineColor(3);
    TIB2->Draw("p");
    TIB3->Draw("p");

    plot1->cd(2);
    TOB1->SetMaximum(1);
    TOB1->SetMinimum(0.);
    TOB1->GetHistogram()->GetXaxis()->SetTitle("Track projection");
    TOB1->GetHistogram()->SetTitle("TOB");
    TOB1->Draw("ap");
    TOB2->SetMarkerColor(2);
    TOB2->SetLineColor(2);
    TOB3->SetMarkerColor(3);
    TOB3->SetLineColor(3);
    TOB2->Draw("p");
    TOB3->Draw("p");

    plot1->cd(3);
    TID1->SetMaximum(1);
    TID1->SetMinimum(0.);
    TID1->GetHistogram()->GetXaxis()->SetTitle("Track projection");
    TID1->GetHistogram()->SetTitle("TID");
    TID1->Draw("ap");
    TID2->SetMarkerColor(2);
    TID2->SetLineColor(2);
    TID3->SetMarkerColor(3);
    TID3->SetLineColor(3);
    TID2->Draw("p");
    TID3->Draw("p");

    plot1->cd(4);
    TEC1->SetMaximum(1);
    TEC1->SetMinimum(0.);
    TEC1->GetHistogram()->GetXaxis()->SetTitle("Track projection");
    TEC1->GetHistogram()->SetTitle("TEC");
    TEC1->Draw("ap");
    TEC2->SetMarkerColor(2);
    TEC2->SetLineColor(2);
    TEC3->SetMarkerColor(3);
    TEC3->SetLineColor(3);
    TEC2->Draw("p");
    TEC3->Draw("p");

    TCanvas * plot2=new TCanvas("resolution_all","resolution_all");
    plot2->cd();
    ALL1->SetMaximum(1);
    ALL1->SetMinimum(0.);
    ALL1->GetHistogram()->GetXaxis()->SetTitle("Track projection");
    ALL1->GetHistogram()->SetTitle("ALL");
    ALL1->Draw("ap");
    ALL2->SetMarkerColor(2);
    ALL2->SetLineColor(2);
    ALL3->SetMarkerColor(3);
    ALL3->SetLineColor(3);
    ALL2->Draw("p");
    //    ALL3->Draw("p");
  
}
