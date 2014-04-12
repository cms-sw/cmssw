
class Plots{

  public:
  Plots();

  void Legend(TString histoname1,TString histoname2,TString histoname3, TH1F *histo1, TH1F *histo2, TH1F *histo3);  void Save();
  
  float convert(float num);

};

void Plots::Plots()
{

gROOT->Reset();
gROOT->Clear();

gStyle->SetNdivisions(10);
gStyle->SetCanvasBorderMode(0); 
gStyle->SetPadBorderMode(1);
gStyle->SetOptTitle(1);
gStyle->SetStatFont(42);
gStyle->SetCanvasColor(10);
gStyle->SetPadColor(0);
gStyle->SetTitleFont(62,"xy");
gStyle->SetLabelFont(62,"xy");
gStyle->SetTitleFontSize(0.05);
gStyle->SetTitleSize(0.039,"xy");
gStyle->SetLabelSize(0.046,"xy");
// gStyle->SetTitleFillColor(0);
gStyle->SetHistFillStyle(1001);
gStyle->SetHistFillColor(0);
gStyle->SetHistLineStyle(1);
gStyle->SetHistLineWidth(2);
gStyle->SetHistLineColor(2);
gStyle->SetTitleXOffset(1.15);
gStyle->SetTitleYOffset(1.15);
gStyle->SetOptStat(1110);
gStyle->SetOptStat(kFALSE);
gStyle->SetOptFit(0111);
gStyle->SetStatH(0.1); 

TCanvas *c1 = new TCanvas("c1","c1",129,17,926,703);
c1->SetBorderSize(2);
c1->SetFrameFillColor(0);
c1->SetLogy(0);
c1->cd(); 

TFile *f[4];
TTree *MyTree[4];



f[0]= new TFile("/tmp/ndefilip/ValidationMisalignedTracker_singlemu100_merged.root");
MyTree[0]=EffTracks;
 
f[1]=new TFile("/tmp/ndefilip/ValidationMisalignedTracker_singlemu100_SurveyLASCosmics_merged.root");
MyTree[1]=EffTracks;
 
f[2]=new TFile("/tmp/ndefilip/ValidationMisalignedTracker_singlemu100_10pb_merged.root");
MyTree[2]=EffTracks;
 
f[3]=new TFile("/tmp/ndefilip/ValidationMisalignedTracker_singlemu100_100pb_merged.root");
MyTree[3]=EffTracks;


////&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS ETA ALIGNED
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
char histoname[128];
char name[128];
TH1F *pullphi[4];

for(int i=0; i<4; i++){
  sprintf(name,"pullphi[%d]",i);
  pullphi[i] = new TH1F(name,name,150,-10,10);
  sprintf(histoname,"pullphi[%d]",i);
  MyTree[i]->Project(histoname,"pullphi","eff==1 && TrackID==13");

  cout << "Entries " << pullphi[i]->GetEntries() <<endl;
  pullphi[i]->Scale(1/pullphi[i]->GetEntries());
  pullphi[i]->SetTitle("pull (#phi)");    
  pullphi[i]->SetXTitle("pull (#phi)");
  pullphi[i]->SetYTitle("arb. units");
  
  pullphi[i]->SetLineColor(i+2);
  pullphi[i]->SetLineStyle(i+1);
  pullphi[i]->SetLineWidth(i+2);
  if (i==0) pullphi[i]->Draw();
  else pullphi[i]->Draw("same");
  //  c1->WaitPrimitive();
  c1->Update();
}

Legend("pullphi[0]","pullphi[1]","pullphi[2]","pullphi[3]",pullphi[0],pullphi[1],pullphi[2],pullphi[3]);

c1->SaveAs("pullphi_mu.eps");
c1->SaveAs("pullphi_mu.gif");
gROOT->Reset();
gROOT->Clear();

delete c1;

}

void Plots::Legend(TString histoname1,TString histoname2,TString histoname3, TString histoname4, TH1F *histo1, TH1F *histo2, TH1F *histo3, TH1F *histo4)
{

TLegend *leg = new TLegend(0.45,0.89,1.,1.); 
leg->SetTextAlign(32);
leg->SetTextColor(1);
leg->SetTextSize(0.020);

char  label[128];
sprintf(label,"perfect alignment;      mean = %1.3f, RMS = %1.3f",convert(histo1->GetMean()),convert(histo1->GetRMS()));
leg->AddEntry(histoname1, label, "l");
sprintf(label,"SurveyLASCosmics alignment; mean = %1.3f, RMS = %1.3f",convert(histo2->GetMean()),convert(histo2->GetRMS()));
leg->AddEntry(histoname2, label, "l");
sprintf(label,"10 pb-1 alignment;  mean = %1.3f, RMS = %1.3f",convert(histo3->GetMean()),convert(histo3->GetRMS()));
leg->AddEntry(histoname3, label, "l");
sprintf(label,"100 pb-1 alignment;  mean = %1.3f, RMS = %1.3f",convert(histo4->GetMean()),convert(histo4->GetRMS()));
leg->AddEntry(histoname4, label, "l");
leg->Draw();

}

float Plots::convert(float num){
  int mean1 = num;
  float res = num - mean1;
  int res2 = res*1000;
  float res3 = res2*0.001;
  float mean2 = mean1 + res3;
                                                                                                                  
  float res4 = res - res3;
  int res5 = res4*10000;
                                                                                                                  
  if(res5>5)
    mean2 = mean2 + 0.001;
                                                                                                                  
  return mean2;
}
