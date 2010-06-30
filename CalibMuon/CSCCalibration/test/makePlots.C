void peds(){
  
  TFile *f = TFile::Open("diffPeds.root");
  TCanvas *c1 = new TCanvas("c1","",800,600);
  DiffPeds->Draw("index:diffPeds");
  //  c1->SetLogy();
  c1->Print("index_vs_diffPeds_run136071.gif");
  
  TCanvas *c2  = new TCanvas("c2","",800,600);
  c2->SetLogy();
  DiffPeds->Draw("diffPeds");
  c2->Print("diffPeds_run136071.gif");

}

void matrix(){

  TFile *f = TFile::Open("diffMatrix.root");
  TCanvas *c1 = new TCanvas("c1","",1000,1000);
  c1->Divide(3,4);
  c1->cd(1);
  DiffMatrix->Draw("index:diffElem33");
  c1->cd(2);
  DiffMatrix->Draw("index:diffElem34");
  c1->cd(3);
  DiffMatrix->Draw("index:diffElem35");

  c1->cd(4);
  DiffMatrix->Draw("index:diffElem44");
  c1->cd(5);
  DiffMatrix->Draw("index:diffElem45");
  c1->cd(6);
  DiffMatrix->Draw("index:diffElem46");

  c1->cd(7);
  DiffMatrix->Draw("index:diffElem55");
  c1->cd(8);
  DiffMatrix->Draw("index:diffElem56");
  c1->cd(9);
  DiffMatrix->Draw("index:diffElem57");

  c1->cd(10);
  DiffMatrix->Draw("index:diffElem66");
  c1->cd(11);
  DiffMatrix->Draw("index:diffElem67");
  c1->cd(12);
  DiffMatrix->Draw("index:diffElem77");

  c1->Print("index_vs_diffMatrix_run136071.gif");
  
  TCanvas *c2  = new TCanvas("c2","",1000,1000);
  c2->Divide(3,4);
  c2->cd(1);
  DiffMatrix->Draw("diffElem33");
  c2->cd(2);
  DiffMatrix->Draw("diffElem34");
  c2->cd(3);
  DiffMatrix->Draw("diffElem35");

  c2->cd(4);
  DiffMatrix->Draw("diffElem44");
  c2->cd(5);
  DiffMatrix->Draw("diffElem45");
  c2->cd(6);
  DiffMatrix->Draw("diffElem46");

  c2->cd(7);
  DiffMatrix->Draw("diffElem55");
  c2->cd(8);
  DiffMatrix->Draw("diffElem56");
  c2->cd(9);
  DiffMatrix->Draw("diffElem57");

  c2->cd(10);
  DiffMatrix->Draw("diffElem66");
  c2->cd(11);
  DiffMatrix->Draw("diffElem67");
  c2->cd(12);
  DiffMatrix->Draw("diffElem77");


  //  c2->SetLogy();
  c2->Print("diffMatrix_run136071.gif");
}

void gains(){

  TFile *f = TFile::Open("diffGains.root");
  TCanvas *c1 = new TCanvas("c1","",800,600);
  DiffGains->Draw("index:diffGains>>H");
  //  c1->SetLogy();
  c1->Print("index_vs_diffGains_run136523.gif");
  
  TCanvas *c2  = new TCanvas("c2","",800,600);
  c2->SetLogy();
  DiffGains->Draw("diffGains");
  c2->Print("diffGains_run136523.gif");
}

void xtalk() {

  TFile *f = TFile::Open("diffXtalk.root");
  TCanvas *c1 = new TCanvas("c1","",1000,1000);
  c1->Divide(2,2);
  
  c1->cd(1);
  DiffXtalk->Draw("index:diffIntL");
  c1->cd(2);
  DiffXtalk->Draw("index:diffIntR");
  c1->cd(3);
  DiffXtalk->Draw("index:diffXtalkL");
  c1->cd(4);
  DiffXtalk->Draw("index:diffXtalkR");

  c1->Print("index_vs_diffXtalk_run136524.gif");

  TCanvas *c2 = new TCanvas("c2","",1000,1000);
  c2->Divide(2,2);
  
  c2->cd(1);
  c2_1->SetLogy();
  DiffXtalk->Draw("diffIntL");
  c2->cd(2);
  c2_2->SetLogy();
  DiffXtalk->Draw("diffIntR");
  c2->cd(3);
  c2_3->SetLogy();
  DiffXtalk->Draw("diffXtalkL");
  c2->cd(4);
  c2_4->SetLogy();
  DiffXtalk->Draw("diffXtalkR");

  c2->Print("diffXtalk_run136524.gif");

}
