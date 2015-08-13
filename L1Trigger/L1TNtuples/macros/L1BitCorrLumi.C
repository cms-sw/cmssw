#include <iostream>
using namespace std;
#include "hist.C"
//#include "geom.C"
#include "L1Ntuple.h"
#include "TLine.h"
#include <TCanvas.h>

void L1BitCorrLumi(L1Ntuple* ntuple, int nlumibin = 10, int ibx=-1, int nevs=-1, int maxlumi=500, int minlumi=0);
void loop(L1Ntuple* ntuple, int i1=0, int i2=-1);
int getfirst(int ib, L1Ntuple* ntuple);

TH2F* H2BitLumi = 0;
//TH2F* H2blumi = h2d("H2blumi",200,0,200,9,-4.5, 4.5);
int maxlumi_;
int minlumi_;
int bminbias=124;
int bxc, nlumi;
int activebits = 9;
int b[10]={1000,1009,9,100 ,15, 46, 54, 55, 1041};
// BPTXAND, HF_COINC,  HFRring2Coinc, SingleFwdJet2, SingleJet6U, SingleEG2,MuHalo, MuOpen, 124
int run_;
int lumi(L1Ntuple* ntuple);
int run(L1Ntuple* ntuple);
////////////////// //////////////////////////////////////////////////////////////////////////

void L1BitCorrLumi(L1Ntuple* ntuple, int nlumibin, int ibx, int nevs, int maxlumi, int minlumi){
  maxlumi_ = maxlumi;
  minlumi_=minlumi;
  nlumi=nlumibin;
 bxc = ibx;


 H2BitLumi = new TH2F("H2blumi","L1 Bits Synchronization",(maxlumi_-minlumi_)/nlumi, minlumi_/nlumi, maxlumi_/nlumi, activebits*10,0,activebits*10);
 if(nevs) {
   hreset();
   loop(ntuple, 0,nevs);
 }
 
 //Making 1d histogram by taking Y-projection , Added by Arun
 TH1D *H2BitLumi1D = H2BitLumi->ProjectionY("H2BitLumi1D", minlumi_/nlumi, maxlumi_/nlumi);

 TCanvas* c1 = new TCanvas("c1","",900,750);
 //TString axistitle(int bit);
 //TString xtitle = axistitle(b1);
 //TString ytitle = axistitle(b2);
 //H2bb->GetXaxis()->SetTitle(xtitle);
 //H2bb->GetYaxis()->SetTitle(ytitle);
 
 H2BitLumi->GetYaxis()->SetBinLabel(5,"bptx_and  0");// - mb");
 H2BitLumi->GetYaxis()->SetBinLabel(15,"TT HF coinc  0");// - mb");
 // H2BitLumi->GetYaxis()->SetBinLabel(25,"DoubleHFr1  0");// - mb");
 H2BitLumi->GetYaxis()->SetBinLabel(25,"DoubleHFr2  0");// - mb");
 H2BitLumi->GetYaxis()->SetBinLabel(35,"SingleFwdJet2  0");// - mb");
 H2BitLumi->GetYaxis()->SetBinLabel(45,"SingleJet6  0");// - mb");
 H2BitLumi->GetYaxis()->SetBinLabel(55,"SingleEG2  0");// - mb");
 H2BitLumi->GetYaxis()->SetBinLabel(65,"MuBeamHalo  0");// - mb");
 H2BitLumi->GetYaxis()->SetBinLabel(75,"SingleMuOpen  0");// - mb");
 //H2BitLumi->GetYaxis()->SetBinLabel(85,"bscOR_and_BPTX");//"bscOR_and_BPTX  0");// - mb");
 H2BitLumi->GetYaxis()->SetBinLabel(85,"BSC TT 41 0");//"bscOR_and_BPTX  0");// - mb");
 H2BitLumi->GetYaxis()->SetBinLabel(2,"-3");
 H2BitLumi->GetYaxis()->SetBinLabel(12,"-3");
 H2BitLumi->GetYaxis()->SetBinLabel(22,"-3");
 H2BitLumi->GetYaxis()->SetBinLabel(32,"-3");
 H2BitLumi->GetYaxis()->SetBinLabel(42,"-3");
 H2BitLumi->GetYaxis()->SetBinLabel(52,"-3");
 H2BitLumi->GetYaxis()->SetBinLabel(62,"-3");
 H2BitLumi->GetYaxis()->SetBinLabel(72,"-3");
 H2BitLumi->GetYaxis()->SetBinLabel(82,"-3");
 //H2BitLumi->GetYaxis()->SetBinLabel(92,"-3");

 H2BitLumi->GetYaxis()->SetBinLabel(8,"+3");
 H2BitLumi->GetYaxis()->SetBinLabel(18,"+3");
 H2BitLumi->GetYaxis()->SetBinLabel(28,"+3");
 H2BitLumi->GetYaxis()->SetBinLabel(38,"+3");
 H2BitLumi->GetYaxis()->SetBinLabel(48,"+3");
 H2BitLumi->GetYaxis()->SetBinLabel(58,"+3");
 H2BitLumi->GetYaxis()->SetBinLabel(68,"+3");
 H2BitLumi->GetYaxis()->SetBinLabel(78,"+3");
 H2BitLumi->GetYaxis()->SetBinLabel(88,"+3");
 //H2BitLumi->GetYaxis()->SetBinLabel(98,"+3");

 H2BitLumi->GetYaxis()->LabelsOption("d");
 H2BitLumi->GetYaxis()->SetLabelFont(42);
 H2BitLumi->GetXaxis()->SetLabelFont(42);
 H2BitLumi->GetYaxis()->SetTitleFont(42);
 H2BitLumi->GetXaxis()->SetTitleFont(42);
 H2BitLumi->GetYaxis()->SetLabelOffset(0.001);
 H2BitLumi->GetYaxis()->SetLabelSize(0.03);
 H2BitLumi->GetYaxis()->SetTitleOffset(2.4);
 TString Xtitle="Lumi Section";
 if (nlumi>1) {
   Xtitle=Xtitle+" / ";
   Xtitle=Xtitle+(Long_t)nlumi;
 }
 //H2BitLumi->GetYaxis()->SetTitle("BX offset to BSC TT41");
 H2BitLumi->GetYaxis()->SetTitle("BX offset to Algo 124");
 H2BitLumi->GetXaxis()->SetTitle(Xtitle);
 H2BitLumi->GetYaxis()->SetTickLength(0);
 H2BitLumi->GetYaxis()->SetNdivisions(10*activebits);
 c1->SetGridy();
 c1->SetLogz();
 TString newtitle = "L1 Synchronization run ";
 newtitle = newtitle + (Long_t) run_;

 H2BitLumi->SetTitle(newtitle);
 H2BitLumi->Draw("colz");


 TLine *lin1 = new TLine(minlumi_/nlumi,9,maxlumi_/nlumi,9);
 lin1->Draw("same");
 TLine *lin2 = new TLine(minlumi_/nlumi,10,maxlumi_/nlumi,10);
 lin2->Draw("same");
 TLine *lin3 = new TLine(minlumi_/nlumi,19,maxlumi_/nlumi,19);
 lin3->Draw("same");
 TLine *lin4 = new TLine(minlumi_/nlumi,20,maxlumi_/nlumi,20);
 lin4->Draw("same");
 TLine *lin5 = new TLine(minlumi_/nlumi,29,maxlumi_/nlumi,29);
 lin5->Draw("same");
 TLine *lin6 = new TLine(minlumi_/nlumi,30,maxlumi_/nlumi,30);
 lin6->Draw("same");
 TLine *lin7 = new TLine(minlumi_/nlumi,39,maxlumi_/nlumi,39);
 lin7->Draw("same");
 TLine *lin8 = new TLine(minlumi_/nlumi,40,maxlumi_/nlumi,40);
 lin8->Draw("same");
 TLine *lin9 = new TLine(minlumi_/nlumi,49,maxlumi_/nlumi,49);
 lin9->Draw("same");
 TLine *lin10 = new TLine(minlumi_/nlumi,50,maxlumi_/nlumi,50);
 lin10->Draw("same");
 TLine *lin11 = new TLine(minlumi_/nlumi,59,maxlumi_/nlumi,59);
 lin11->Draw("same");
 TLine *lin12 = new TLine(minlumi_/nlumi,60,maxlumi_/nlumi,60);
 lin12->Draw("same");
 TLine *lin13 = new TLine(minlumi_/nlumi,69,maxlumi_/nlumi,69);
 lin13->Draw("same");
 TLine *lin14 = new TLine(minlumi_/nlumi,70,maxlumi_/nlumi,70);
 lin14->Draw("same");
 TLine *lin15 = new TLine(minlumi_/nlumi,79,maxlumi_/nlumi,79);
 lin15->Draw("same");
 TLine *lin16 = new TLine(minlumi_/nlumi,80,maxlumi_/nlumi,80);
 lin16->Draw("same");
 TLine *lin17 = new TLine(minlumi_/nlumi,89,maxlumi_/nlumi,89);
 lin17->Draw("same");
 TLine *lin18 = new TLine(minlumi_/nlumi,90,maxlumi_/nlumi,90);
 lin18->Draw("same");
 //TLine *lin19 = new TLine(0,99,maxlumi_/nlumi,99);
 //lin19->Draw("same");
 //TLine *lin20 = new TLine(0,100,maxlumi_/nlumi,100);
 //lin20->Draw("same");

 c1->SetLeftMargin(0.18);
 c1->SetRightMargin(0.13);

 TString filer = "L1Synchro_";
 filer = filer + (Long_t) run_;

 // TString file_root = filer + ".root";
 TString file_gif = filer + ".gif";
 cout << file_gif << endl;
 //TFile fout(file_root,"recreate");

 c1->Print(file_gif,"gif");
 // TString file_root = filer + ".root";
 //c1->Print(file_root,"root");
 //fout.Write();
 //fout.Close();
 c1->Update();

 //Making 1d histogram by taking Y-projection , Added by Arun
 TCanvas* c2 = new TCanvas("c2","",900,750);

 //TH1D *H2BitLumi1D = H2BitLumi->ProjectionY("H2BitLumi1D", (minlumi_ + 10)/nlumi, (maxlumi_ - 10)/nlumi);
  
 H2BitLumi1D->GetXaxis()->SetBinLabel(5,"bptx_and  0");// - mb");
 H2BitLumi1D->GetXaxis()->SetBinLabel(15,"TT HF coinc  0");// - mb");
 // H2BitLumi1D->GetXaxis()->SetBinLabel(25,"DoubleHFr1  0");// - mb");
 H2BitLumi1D->GetXaxis()->SetBinLabel(25,"DoubleHFr2  0");// - mb");
 H2BitLumi1D->GetXaxis()->SetBinLabel(35,"SingleFwdJet2  0");// - mb");
 H2BitLumi1D->GetXaxis()->SetBinLabel(45,"SingleJet6  0");// - mb");
 H2BitLumi1D->GetXaxis()->SetBinLabel(55,"SingleEG2  0");// - mb");
 H2BitLumi1D->GetXaxis()->SetBinLabel(65,"MuBeamHalo  0");// - mb");
 H2BitLumi1D->GetXaxis()->SetBinLabel(75,"SingleMuOpen  0");// - mb");
 //H2BitLumi1D->GetXaxis()->SetBinLabel(85,"bscOR_and_BPTX");//"bscOR_and_BPTX  0");// - mb");
 H2BitLumi1D->GetXaxis()->SetBinLabel(85,"BSC TT 41 0");//"bscOR_and_BPTX  0");// - mb");
 H2BitLumi1D->GetXaxis()->SetBinLabel(2,"-3");
 H2BitLumi1D->GetXaxis()->SetBinLabel(12,"-3");
 H2BitLumi1D->GetXaxis()->SetBinLabel(22,"-3");
 H2BitLumi1D->GetXaxis()->SetBinLabel(32,"-3");
 H2BitLumi1D->GetXaxis()->SetBinLabel(42,"-3");
 H2BitLumi1D->GetXaxis()->SetBinLabel(52,"-3");
 H2BitLumi1D->GetXaxis()->SetBinLabel(62,"-3");
 H2BitLumi1D->GetXaxis()->SetBinLabel(72,"-3");
 H2BitLumi1D->GetXaxis()->SetBinLabel(82,"-3");
 //H2BitLumi1D->GetXaxis()->SetBinLabel(92,"-3");

 H2BitLumi1D->GetXaxis()->SetBinLabel(8,"+3");
 H2BitLumi1D->GetXaxis()->SetBinLabel(18,"+3");
 H2BitLumi1D->GetXaxis()->SetBinLabel(28,"+3");
 H2BitLumi1D->GetXaxis()->SetBinLabel(38,"+3");
 H2BitLumi1D->GetXaxis()->SetBinLabel(48,"+3");
 H2BitLumi1D->GetXaxis()->SetBinLabel(58,"+3");
 H2BitLumi1D->GetXaxis()->SetBinLabel(68,"+3");
 H2BitLumi1D->GetXaxis()->SetBinLabel(78,"+3");
 H2BitLumi1D->GetXaxis()->SetBinLabel(88,"+3");
 //H2BitLumi1D->GetXaxis()->SetBinLabel(98,"+3");
 
 //H2BitLumi1D->SetMarkerStyle(20);
 //H2BitLumi1D->SetMarkerSize(1.0);
 
 H2BitLumi1D->GetYaxis()->LabelsOption("v");
 H2BitLumi1D->GetXaxis()->SetLabelFont(42);
 H2BitLumi1D->GetXaxis()->SetTitleFont(42);
 H2BitLumi1D->GetXaxis()->SetLabelOffset(0.001);
 H2BitLumi1D->GetXaxis()->SetLabelSize(0.03);
 H2BitLumi1D->GetXaxis()->SetTitleOffset(2.6);

 H2BitLumi1D->GetXaxis()->SetTitle("BX offset to Algo 124");
 H2BitLumi1D->GetXaxis()->SetTickLength(0.02);
 //H2BitLumi1D->GetXaxis()->SetNdivisions(10*activebits);
 c2->SetLogy();

 TString newtitle1D = "L1 Synchronization run ";
 newtitle1D = newtitle1D + (Long_t) run_;

 H2BitLumi1D->SetTitle(newtitle1D);
 H2BitLumi1D->SetFillColor(2);
 H2BitLumi1D->Draw();

 c2->SetBottomMargin(0.21);

 TString filer1D = "L1Synchro_1DPlot_";
 filer1D = filer1D + (Long_t) run_;

 TString file1d_gif = filer1D + ".gif";
 c2->Print(file1d_gif,"gif");
 //TString file1d_root = filer1D + ".root";
 //c2->Print(file1d_root,"root");

 bool PrintFraction = false;
 if(PrintFraction){
   TString tBitNames[9] = {"bptx_and", "TT HF coinc", "DoubleHFr2", "SingleFwdJet2", "SingleJet6", "SingleEG2", "MuBeamHalo", "SingleMuOpen", "BSC TT 41"};
   for(int it = 0; it < 9; it++){
     int jt = it*10 + 5;
     double sum = 0, sumPreFrac = 0, sumPostFrac = 0;
     for(int ibin = 0; ibin < 9; ibin++){
       sum += H2BitLumi1D->GetBinContent(jt + ibin - 4);
       if(ibin < 4)sumPreFrac += H2BitLumi1D->GetBinContent(jt + ibin - 4);
       if(ibin > 4)sumPostFrac += H2BitLumi1D->GetBinContent(jt + ibin - 4);
     }
     cout<<tBitNames[it]<<" : "<<"Pre Firing Fraction = "<<sumPreFrac/sum
	 <<" Post Firing Fraction = "<<sumPostFrac/sum<<endl;
   }
 }

}

////////////////////////////////////////////////////////////////////////////////////////////

void loop(L1Ntuple* ntuple, int i1, int i2) {
  int bpass1, bpass2, bpass3;

  
  if(i2==-1 || i2>ntuple->fChain->GetEntries()) i2=ntuple->fChain->GetEntries();
  
  cout << "Going to run on " << i2 << " events" << " for L1 synchronization check..." << endl;
  
  for(int i=i1; i<i2; i++) {
    
    if (ntuple->GetEntry(i)) {
      
      if(!(i%100000) && i) cout << "processing event " << i << "\r" << flush;
      //      if(!(i%1) && i) cout << "processing event " << i << "\r" << endl;
      
      if(bxc!=-1 && abs(ntuple->event_->bx - bxc) > 4)  continue;
      for (int ibit=0; ibit<activebits; ibit++){
	bpass1=getfirst(b[ibit],ntuple);
	bpass2=getfirst(bminbias,ntuple);
	bpass3=getfirst(1041,ntuple);
	if (bpass1!=-3 && bpass2!=-3 && bpass3!=-3){
	  H2BitLumi->Fill(lumi(ntuple)/nlumi,(10*ibit+bpass1-bpass2+4) );
	}
      }
      run_=run(ntuple);
    }
  }
  cout << "                                                                        \r" << flush;
  return;
}
int lumi(L1Ntuple* ntuple){
	return ntuple->event_->lumi;
}
int run(L1Ntuple* ntuple){
  return ntuple->event_->run;
}
int getfirst(int ib, L1Ntuple* ntuple) {
 int ir = -3;
 if(ib<64) {
   for(int i=0; i<5; i++) {
     if((ntuple->gt_->tw1[i]>>ib)&1) {
       ir=i-2;
	break;
     }
   }
 } else if(ib<128) {
   for(int i=0; i<5; i++) {
     if((ntuple->gt_->tw2[i]>>(ib-64))&1) {
       ir=i-2;
	break;
     }
   }
 } else {
   for(int i=0; i<5; i++) {
     if((ntuple->gt_->tt[i]>>(ib-1000))&1) {
       ir=i-2;
	break;
     }
   }
 }
 return ir;
}

TString axistitle(int bit){
  TString title;
  if (bit>127) {
    title = "BX of TT bit ";
    title = title + (Long_t)(bit-1000);
  } else {
    title = "BX of Algo bit ";
    title = title + (Long_t)(bit);
  } 
  return title;
}
 
