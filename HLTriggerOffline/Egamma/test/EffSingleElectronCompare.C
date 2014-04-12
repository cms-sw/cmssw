{
gROOT->Reset();

TH1F::SetDefaultSumw2();

TFile f1("dqm_Wenu_versionA.root");
TFile f2("dqm_Wenu_versionB.root");

const int nf = 7;

string filters[nf] = {
        "l1seedSingle",
        "hltL1IsoSingleL1MatchFilter",
        "hltL1IsoSingleElectronEtFilter",
        "hltL1IsoSingleElectronHcalIsolFilter",
        "hltL1IsoSingleElectronPixelMatchFilter",
        "hltL1IsoSingleElectronHOneOEMinusOneOPFilter",
        "hltL1IsoSingleElectronTrackIsolFilter"
        }

string pathname="singleElectronDQM";
string histosEt[nf];
string histosEta[nf];
for (int i=0; i<nf; i++){
  histosEt[i]=pathname+"/"+filters[i]+"et";
  histosEta[i]=pathname+"/"+filters[i]+"eta";
}
TH1F* het1[nf];
TH1F* heta1[nf];
TH1F* het2[nf];
TH1F* heta2[nf];
for (int i=0; i<nf; i++){
het1[i] = (TH1F*) f1.Get(histosEt[i].c_str());
het1[i]->GetXaxis()->SetTitle("E_{T}");
heta1[i] = (TH1F*) f1.Get(histosEta[i].c_str());
heta1[i]->GetXaxis()->SetTitle("\\eta");

het2[i] = (TH1F*) f2.Get(histosEt[i].c_str());
het2[i]->GetXaxis()->SetTitle("E_{T}");
heta2[i] = (TH1F*) f2.Get(histosEta[i].c_str());
heta2[i]->GetXaxis()->SetTitle("\\eta");

 het2[i]->SetLineColor(2);
 heta2[i]->SetLineColor(2);

}

TH1F* effEt1[nf-1];
TH1F* effEta1[nf-1];
for (int i=2; i<nf; i++){//non ha senso per L1 e L1 match
  string nameEff = "eff" + string(het1[i]->GetTitle())+"_1";
  effEt1[i-2]= new TH1F(nameEff.c_str(),nameEff.c_str(),40,0.,200.);
  effEt1[i-2]->SetTitle(nameEff.c_str());
  effEt1[i-2]->GetXaxis()->SetTitle("E_{T}");
  effEt1[i-2]->GetYaxis()->SetTitle("eff");
  effEt1[i-2]->Divide(het1[i], het1[i-1],1.,1.,"B");

  string nameEffeta = "eff" + string(heta1[i]->GetTitle())+"_1";
  effEta1[i-2]= new TH1F(nameEffeta.c_str(),nameEffeta.c_str(),40,-2.7,2.7.);
  effEta1[i-2]->SetTitle(nameEffeta.c_str());
  effEta1[i-2]->GetXaxis()->SetTitle("\\eta");
  effEta1[i-2]->GetYaxis()->SetTitle("eff");
  effEta1[i-2]->Divide(heta1[i], heta1[i-1],1.,1.,"B");
}
cout<<"AAAAAAAAAAA"<<endl;
effEt1[nf-2]=new TH1F("HLT_L1_et_1","HLT/L1",40,0.,200.);
effEt1[nf-2]->GetXaxis()->SetTitle("E_{T}");
effEt1[nf-2]->GetYaxis()->SetTitle("eff");
effEt1[nf-2]->Divide(het1[nf-1], het1[nf-6],1.,1.,"B");

effEta1[nf-2]= new TH1F("HLT_L1_eta_1","HLT/L1",40,-2.7,2.7.);
effEta1[nf-2]->GetXaxis()->SetTitle("\\eta");
effEta1[nf-2]->GetYaxis()->SetTitle("eff");
effEta1[nf-2]->Divide(heta1[nf-1], heta1[nf-6],1.,1.,"B");
cout<<"BBBAAAAAAAAAAA"<<endl;
TH1F* effEt2[nf-1];
TH1F* effEta2[nf-1];
for (int i=2; i<nf; i++){//non ha senso per L1 e L1 match
  string nameEff = "eff" + string(het2[i]->GetTitle())+"_2";
  effEt2[i-2]= new TH1F(nameEff.c_str(),nameEff.c_str(),40,0.,200.);
  effEt2[i-2]->SetTitle(nameEff.c_str());
  effEt2[i-2]->GetXaxis()->SetTitle("E_{T}");
  effEt2[i-2]->GetYaxis()->SetTitle("eff");
  effEt2[i-2]->Divide(het2[i], het2[i-1],1.,1.,"B");

  string nameEffeta = "eff" + string(heta2[i]->GetTitle())+"_2";
  effEta2[i-2]= new TH1F(nameEffeta.c_str(),nameEffeta.c_str(),40,-2.7,2.7.);
  effEta2[i-2]->SetTitle(nameEffeta.c_str());
  effEta2[i-2]->GetXaxis()->SetTitle("\\eta");
  effEta2[i-2]->GetYaxis()->SetTitle("eff");
  effEta2[i-2]->Divide(heta2[i], heta2[i-1],1.,1.,"B");

  effEt2[i-2]->SetLineColor(2);
  effEta2[i-2]->SetLineColor(2);
 }
cout<<"CCCCAAAAAAAAAAA"<<endl;
effEt2[nf-2]=new TH1F("HLT_L1_et_2","HLT/L1",40,0.,200.);
effEt2[nf-2]->GetXaxis()->SetTitle("E_{T}");
effEt2[nf-2]->GetYaxis()->SetTitle("eff");
effEt2[nf-2]->Divide(het2[nf-1], het2[nf-6],1.,1.,"B");
effEta2[nf-2]= new TH1F("HLT_L1_eta_2","HLT/L1",40,-2.7,2.7.);
effEta2[nf-2]->GetXaxis()->SetTitle("\\eta");
effEta2[nf-2]->GetYaxis()->SetTitle("eff");
effEta2[nf-2]->Divide(heta2[nf-1], heta2[nf-6],1.,1.,"B");
effEt2[nf-2]->SetLineColor(2);
effEta2[nf-2]->SetLineColor(2);
cout<<"DDDDDDDDAAAAAAAAAAA"<<endl;
TCanvas* cEt[nf];
TCanvas* cEta[nf];
TCanvas* ceffEt[nf-1];
TCanvas* ceffEta[nf-1];

for (int i=0; i<nf; i++){
  cEt[i] = new TCanvas();
  cEt[i]->Draw();
  het1[i]->DrawNormalized();
  het2[i]->DrawNormalized("same");
  string imName= "images/"+string(het1[i]->GetTitle())+".gif";
  cEt[i]->Print(imName.c_str());

  cEta[i] = new TCanvas();
  cEta[i]->Draw();
  heta1[i]->DrawNormalized();
  heta2[i]->DrawNormalized("same");
  imName= "images/"+string(heta1[i]->GetTitle())+".gif";
  cEta[i]->Print(imName.c_str());
}

for (int i=0; i<nf-1; i++){
  ceffEt[i] = new TCanvas();
  ceffEt[i]->Draw();
  effEt1[i]->Draw("e");
  effEt2[i]->Draw("esame");
  string imName= "images/"+string(effEt1[i]->GetTitle())+".gif";
  ceffEt[i]->Print(imName.c_str());

  ceffEta[i] = new TCanvas();
  ceffEta[i]->Draw();
  effEta1[i]->Draw("e");
  effEta2[i]->Draw("esame");
  imName= "images/"+string(effEta1[i]->GetTitle())+".gif";
  ceffEta[i]->Print(imName.c_str());
}

}
