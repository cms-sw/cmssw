#include "TText.h"

void plotHist(TH1* hist, double tsize, double max1, double max2=-1.0, bool logscale=false) {

  hist->Draw();
  
  double n=0.0;
  double ntrunc1=0.0;
  double ntrunc2=0.0;
 
  for(int i=0;i<hist->GetNbinsX()+1;i++) {
    n+=hist->GetBinCenter(i)*hist->GetBinContent(i);
    if (hist->GetBinCenter(i)>max1) {
      ntrunc1+=(hist->GetBinCenter(i)-max1)*hist->GetBinContent(i);
    }
    if (hist->GetBinCenter(i)>max2) {
      ntrunc2+=(hist->GetBinCenter(i)-max2)*hist->GetBinContent(i);
    }
  }
  unsigned int ieff=1000000*ntrunc1/n;
  double eff=ieff/10000.0;
  TString effstr;
  effstr+=eff;
  effstr=effstr(0,5);
  //cout <<"ieff eff "<<ieff<<" "<<eff<<endl;
  //cout << "Loss for "<<maxlink<<" is "<<100*ntrunc/n<<"%"<<endl;
  TString name="Loss for ";
  name+=max1;
  name+=" is ";
  name+=effstr;
  name+="%";
  double ycoord=0.9;
  if (logscale) ycoord=0.8;
  TText *t1 = new TText(0.4*hist->GetBinCenter(hist->GetNbinsX()-1),
			ycoord*hist->GetMaximum(),name);
  t1->SetTextAlign(11); t1->SetTextSize(tsize);
  t1->SetTextColor(kRed);
  t1->Draw();
  TLine *l=new TLine(max1,0,max1,0.25*hist->GetMaximum());
  l->SetLineColor(kRed);
  l->Draw();
  //max2=-1;
  if (max2>0){
    unsigned int ieff=1000000*ntrunc2/n;
    double eff=ieff/10000.0;
    TString effstr;
    effstr+=eff;
    effstr=effstr(0,5);
    //cout <<"ieff eff "<<ieff<<" "<<eff<<endl;
    //cout << "Loss for "<<maxlink<<" is "<<100*ntrunc/n<<"%"<<endl;
    TString name="Loss for ";
    name+=max2;
    name+=" is ";
    name+=effstr;
    name+="%";
    double ycoord=0.82;
    if (logscale) ycoord=0.4;
    TText *t1 = new TText(0.4*hist->GetBinCenter(hist->GetNbinsX()-1),
			  ycoord*hist->GetMaximum(),name);
    t1->SetTextAlign(11); t1->SetTextSize(tsize);
    t1->SetTextColor(kBlue);
    t1->Draw();
    TLine *l=new TLine(max2,0,max2,0.25*hist->GetMaximum());
    l->SetLineColor(kBlue);
    l->Draw();
  }
  
}
