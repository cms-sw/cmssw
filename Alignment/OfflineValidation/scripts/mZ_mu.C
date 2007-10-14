
class Plots{

  public:
  Plots();

  void Legend(TString histoname1,TString histoname2,TString histoname3,TString histoname4,TString histoname5,TH1F *histo1,TH1F *histo2,TH1F *histo3,TH1F *histo4,TH1F *histo5);  

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
gStyle->SetTitleSize(0.025,"xy");
gStyle->SetLabelSize(0.025,"xy");
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
// gStyle->SetOptFit(0111);
gStyle->SetStatH(0.1); 
gStyle->SetOptFit(1011);

TCanvas *c1 = new TCanvas("c1","c1",129,17,926,703);
c1->SetBorderSize(2);
c1->SetFrameFillColor(0);
c1->SetLogy(0);
c1->cd(); 

TFile *f[5];
TTree *MyTree[5];

f[0]= new TFile("../../Z/MisalignmentIdeal.root");  
MyTree[0]=Tracks;

f[1]=new TFile("Misalignment_SurveyLASOnlyScenario_refitter_zmumuSurveyLASCosmics.root");
MyTree[1]=Tracks;

f[2]=new TFile("../../Z/Misalignment10.root");
MyTree[2]=Tracks;

f[3]=new TFile("../../Z/Misalignment100.root");
MyTree[3]=Tracks;

f[4]=new TFile("../../Z/Misalignment_NOAPE_2.root");
MyTree[4]=Tracks;

////&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS ETA ALIGNED
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
char histoname[128];
char name[128];
TH1F *mZmu[5];

Double_t lorentzianPeak(Double_t *x, Double_t *par) {
  return (0.5*par[0]*par[1]/TMath::Pi()) /
    TMath::Max( 1.e-10,(x[0]-par[2])*(x[0]-par[2]) + .25*par[1]*par[1]);
}

TF1 *fitFcn = new TF1("fitFcn",lorentzianPeak,70,110,3);
fitFcn->SetParameters(100,3,90);
fitFcn->SetParNames("Ftop","Fwidth","Fmass");
fitFcn->SetLineWidth(2);

for(int i=0; i<5; i++){
  sprintf(name,"mZmu[%d]",i);
  mZmu[i] = new TH1F(name,name,100,70.,110.);
  sprintf(histoname,"mZmu[%d]",i);
  MyTree[i]->Project(histoname,"recmzmu","eff==1 && recmzmu>0.");

  cout << "Entries " << mZmu[i]->GetEntries() <<endl;
  mZmu[i]->Scale(1/mZmu[i]->GetEntries());
  mZmu[i]->SetTitle("Z->#mu#mu mass from Z->#mu#mu");    
  mZmu[i]->SetXTitle(" GeV/c2");
  mZmu[i]->SetYTitle("arb. units");
  
  mZmu[i]->SetLineColor(i+2);
  // mZmu[i]->SetLineStyle(i+1);
  // mZmu[i]->SetLineWidth(i+2);
  if (i==0) {
    //    mZmu[i]->Fit("fitFcn","0","",70,110);
    //const Int_t kNotDraw = 1<<9; 
    //mZmu[i]->GetFunction("fitFcn")->ResetBit(kNotDraw);
    mZmu[i]->Draw("");
  }
  else { 
    //mZmu[i]->Fit("fitFcn","0","",70,110);
    //const Int_t kNotDraw = 1<<9; 
    //mZmu[i]->GetFunction("fitFcn")->ResetBit(kNotDraw);
    mZmu[i]->Draw("same");
  }
  //  c1->WaitPrimitive();
  c1->Update();
}

Legend("mZmu[0]","mZmu[1]","mZmu[2]","mZmu[3]","mZmu[4]",mZmu[0],mZmu[1],mZmu[2],mZmu[3],mZmu[4]);

c1->SaveAs("mZ_mu_CMSSW_1_3_4_lat.gif");
c1->SaveAs("mZ_mu_CMSSW_1_3_4_lat.eps");
gROOT->Reset();
gROOT->Clear();

delete c1;

}

void Plots::Legend(TString histoname1,TString histoname2,TString histoname3,TString histoname4, TString histoname5, TH1F *histo1, TH1F *histo2, TH1F *histo3,TH1F *histo4,TH1F *histo5)
{

TLegend *leg = new TLegend(0.,0.77.,0.5,0.9); 
leg->SetTextAlign(32);
leg->SetTextColor(1);
leg->SetTextSize(0.018);

char  label[128];
sprintf(label,"No Misalignment;      mean = %1.3f, RMS = %1.3f",convert(histo1->GetMean()),convert(histo1->GetRMS()));
leg->AddEntry(histoname1, label, "l");
sprintf(label,"SurveyLASCosmics Misalignment; mean = %1.3f, RMS = %1.3f",convert(histo2->GetMean()),convert(histo2->GetRMS()));
leg->AddEntry(histoname2, label, "l");
sprintf(label,"10pb-1 Misalignment;  mean = %1.3f, RMS = %1.3f",convert(histo3->GetMean()),convert(histo3->GetRMS()));
leg->AddEntry(histoname3, label, "l");
sprintf(label,"100pb-1 Misalignment;  mean = %1.3f, RMS = %1.3f",convert(histo4->GetMean()),convert(histo4->GetRMS()));
leg->AddEntry(histoname4, label, "l");
sprintf(label,"10pb-1 Misalignment; APE not used; mean = %1.3f, RMS = %1.3f",convert(histo5->GetMean()),convert(histo5->GetRMS()));
leg->AddEntry(histoname5, label, "l");
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

