#include "TF1.h"
#include "TLegend.h"
#include <TCanvas.h>

//------------ Exemple of usage --------------
//
// In a root session:
// 	.L SFlightFuncs.C+g	//To load this program
// Then...
// To get a pointer to the SFlight function for CSV tagger Loose (L) in the eta range 0.0-0.5 for the data taking period A+B+C+D, use: 
//	TF1* SFlight = GetSFlmean("CSV","L",0.0, 0.5, "ABCD")
// To get a pointer to the SFlightmin function for CSV tagger Loose (L) in the eta range 0.0-0.5 for the data taking period A+B+C+D, use: 
//	TF1* SFlightmin = GetSFlmin("CSV","L",0.0, 0.5, "ABCD")
// To get a pointer to the SFlightmax function for CSV tagger Loose (L) in the eta range 0.0-0.5 for the data taking period A+B+C+D, use: 
//	TF1* SFlightmax = GetSFlmax("CSV","L",0.0, 0.5, "ABCD")
//
// Note:
// 1) SFlightmin and SFlightmax correspond to SFlight +- (stat+syst error).
// 2) If the specified combination of tagger/taggerstrength/etarange is not tabulated,
//    a NULL pointer is returned.
//
//-------------------------------------------
TF1* GetSFLight(TString meanminmax, TString tagger, TString TaggerStrength, Float_t Etamin, Float_t Etamax, TString DataPeriod);

TF1* GetSFlmean(TString tagger, TString TaggerStrength, float Etamin, float Etamax, TString DataPeriod)
{
  return GetSFLight("mean",tagger,TaggerStrength,Etamin,Etamax,DataPeriod);
}
TF1* GetSFlmin(TString tagger, TString TaggerStrength, float Etamin, float Etamax, TString DataPeriod)
{
  return GetSFLight("min",tagger,TaggerStrength,Etamin,Etamax,DataPeriod);
}
TF1* GetSFlmax(TString tagger, TString TaggerStrength, float Etamin, float Etamax, TString DataPeriod)
{
  return GetSFLight("max",tagger,TaggerStrength,Etamin,Etamax,DataPeriod);
}

TF1* plotmean(TString tagger, TString TaggerStrength, float Etamin, float Etamax, TString DataPeriod = "ABCD", TString opt = "" , int col = 1, float lineWidth = 1, int lineStyle = 1)
{
  TF1* f = GetSFlmean(tagger,TaggerStrength,Etamin,Etamax,DataPeriod);
  if( f != NULL )
  {
    f->SetLineColor(col);
    f->SetMinimum(0.4);
    f->SetMaximum(1.6);
    f->SetLineWidth(lineWidth);
    f->SetLineStyle(lineStyle);
    f->Draw(opt);
  }
  //else cout << "NULL pointer returned... Function seems not to exist" << endl;
  return f;
}
TF1* plotmin(TString tagger, TString TaggerStrength, float Etamin, float Etamax, TString DataPeriod, TString opt = "" , int col = 1, float lineWidth = 1, int lineStyle = 1)
{
  TF1* f = GetSFlmin(tagger,TaggerStrength,Etamin,Etamax,DataPeriod);
  if( f != NULL )
  {
    f->SetLineColor(col);
    f->SetLineWidth(lineWidth);
    f->SetLineStyle(lineStyle);
    f->Draw(opt);
  }
  //else cout << "NULL pointer returned... Function seems not to exist" << endl;
  return f;
}
TF1* plotmax(TString tagger, TString TaggerStrength, float Etamin, float Etamax, TString DataPeriod, TString opt = "" , int col = 1, float lineWidth = 1, int lineStyle = 1)
{
  TF1* f = GetSFlmax(tagger,TaggerStrength,Etamin,Etamax,DataPeriod);
  if( f != NULL )
  {
    f->SetLineColor(col);
    f->SetLineWidth(lineWidth);
    f->SetLineStyle(lineStyle);
    f->Draw(opt);
  }
  //else cout << "NULL pointer returned... Function seems not to exist" << endl;
  return f;
}
void plotmean(TCanvas *yourC, int yourzone, TString tagger, TString TaggerStrength, TString DataPeriod)
{
 TString legTitle = tagger + TaggerStrength;
 //TCanvas *cWork = new TCanvas("cWork", "plots",200,10,700,750);
 yourC->SetFillColor(10);
 yourC->SetFillStyle(4000);
 yourC->SetBorderSize(2);

 yourC->cd(yourzone);
 yourC->cd(yourzone)->SetFillColor(10);
 yourC->cd(yourzone)->SetFillStyle(4000);
 yourC->cd(yourzone)->SetBorderSize(2);
  TF1 *fmean, *fmin, *fmax;
  TF1* f[10];
  TLegend* leg= new TLegend(0.60,0.56,0.89,0.89);
    leg->SetBorderSize(0);
    leg->SetFillColor(kWhite);
    leg->SetTextFont(42);
    leg->SetHeader(legTitle);
  float etamin[10], etamax[10]; 
  int N=1;
  etamin[0] = 0.0; etamax[0] = 2.4;

  if( TaggerStrength == "L" )
  {
    N = 4;
    etamin[1] = 0.0; etamax[1] = 0.5;
    etamin[2] = 0.5; etamax[2] = 1.0;
    etamin[3] = 1.0; etamax[3] = 1.5;
    etamin[4] = 1.5; etamax[4] = 2.4;
  }
  else if( TaggerStrength == "M" )
  {
    N = 3;
    etamin[1] = 0.0; etamax[1] = 0.8;
    etamin[2] = 0.8; etamax[2] = 1.6;
    etamin[3] = 1.6; etamax[3] = 2.4;
  }
  else if( TaggerStrength == "T" )
  {
    N = 1;
    etamin[1] = 0.0; etamax[1] = 2.4;
  }

  //etamin = 0.0; etamax = 2.4;
/*
  fmean = plotmean(tagger,TaggerStrength,etamin[0], etamax[0], "", 1, 2, 1);
  leg->AddEntry(fmean,"Mean(SF)","l");
  fmin = plotmin(tagger,TaggerStrength,etamin[0], etamax[0], "same", 1, 2, 2);
  leg->AddEntry(fmin,"Min(SF)","l");
  fmax = plotmax(tagger,TaggerStrength,etamin[0], etamax[0], "same", 1, 2, 2);
  leg->AddEntry(fmax,"Max(SF)","l");
*/

  f[1] = plotmean(tagger,TaggerStrength,etamin[1], etamax[1], DataPeriod, "", 1, 1);
    //TString rangeEta = Form("Mean(SF(%1.1f #leq #eta %1.1f))",etamin[1],etamax[1]);
    TString rangeEta = Form("SF(%1.1f #leq #eta %1.1f)",etamin[1],etamax[1]);
    leg->AddEntry(f[1],rangeEta,"l");
  for( int i = 2; i <= N; ++i )
  {
    f[i] = plotmean(tagger,TaggerStrength,etamin[i], etamax[i], DataPeriod, "same", i, 1);
    //TString rangeEta = Form("Mean(SF(%1.1f #leq #eta %1.1f))",etamin[i],etamax[i]);
    TString rangeEta = Form("SF(%1.1f #leq #eta %1.1f)",etamin[i],etamax[i]);
    leg->AddEntry(f[i],rangeEta,"l");
  }
  //leg->AddEntry(gg," gluon jets","P");
  leg->Draw();
  //return cWork;
}
TCanvas *plotmean(TString tagger, TString TaggerStrength, TString DataPeriod)
{
 TCanvas *cWork = new TCanvas("cWork", "plots",200,10,700,750);
 plotmean(cWork, 0, tagger, TaggerStrength, DataPeriod);

 return cWork;
}
TCanvas *plotmean(TString selecter, TString DataPeriod)
{
 TCanvas *cWork = NULL; 
 if( selecter == "L" )
 {
   cWork = new TCanvas("cWork", "plots",200,10,700,500);
   cWork->Divide(1,2);
   plotmean(cWork, 1, "JP", selecter, DataPeriod);
   plotmean(cWork, 2, "CSV", selecter, DataPeriod);
 }
 else if( selecter == "M" )
 {
   cWork = new TCanvas("cWork", "plots",200,10,700,500);
   cWork->Divide(1,2);
   plotmean(cWork, 1, "JP", selecter, DataPeriod);
   plotmean(cWork, 2, "CSV", selecter, DataPeriod);
 }
 else if( selecter == "T" )
 {
   cWork = new TCanvas("cWork", "plots",200,10,700,750);
   cWork->Divide(1,3);
   plotmean(cWork, 1, "JP", selecter, DataPeriod);
   plotmean(cWork, 2, "CSV", selecter, DataPeriod);
   plotmean(cWork, 3, "TCHP", selecter, DataPeriod);
 }
 else if( selecter == "TCHP" )
 {
   cWork = new TCanvas("cWork", "plots",200,10,700,250);
   plotmean(cWork, 0, selecter, "T", DataPeriod);
 }
 else
 {
   cWork = new TCanvas("cWork", "plots",200,10,700,750);
   cWork->Divide(1,3);
   plotmean(cWork, 1, selecter, "L", DataPeriod);
   plotmean(cWork, 2, selecter, "M", DataPeriod);
   plotmean(cWork, 3, selecter, "T", DataPeriod);
 }

 cWork->WaitPrimitive();
 cWork->SaveAs("SFlightFunc_"+selecter+".pdf");
 return cWork;
}
TCanvas *plotmean(TString DataPeriod = "ABCD")
{
 TCanvas *cWork = new TCanvas("cWork", "plots",200,10,700,750);
 cWork->Divide(3,3, 0.002, 0.002);
 cWork->SetFillColor(10);
 cWork->SetFillStyle(4000);
 cWork->SetBorderSize(1);
 for( int i = 0; i < 3*3; ++i )
 {
   cWork->cd(i+1)->SetFillColor(10);
   cWork->cd(i+1)->SetFillStyle(4000);
   cWork->cd(i+1)->SetBorderSize(1);
 }
 plotmean(cWork, 1, "JP", "L", DataPeriod);
 plotmean(cWork, 2, "JP", "M", DataPeriod);
 plotmean(cWork, 3, "JP", "T", DataPeriod);
 plotmean(cWork, 4, "CSV", "L", DataPeriod);
 plotmean(cWork, 5, "CSV", "M", DataPeriod);
 plotmean(cWork, 6, "CSV", "T", DataPeriod);
 plotmean(cWork, 9, "TCHP", "T", DataPeriod);

 return cWork;
}


TF1* GetSFLight(TString meanminmax, TString tagger, TString TaggerStrength, Float_t Etamin, Float_t Etamax, TString DataPeriod)
{
  TF1 *tmpSFl = NULL;

  TString Atagger = tagger+TaggerStrength;
  TString sEtamin = Form("%1.1f",Etamin);
  TString sEtamax = Form("%1.1f",Etamax);
  cout << sEtamin << endl;
  cout << sEtamax << endl;

  if( (TaggerStrength == "L" || TaggerStrength == "M") && sEtamin == "0.0" && sEtamax == "2.4" )
  {
    cout << "For L and M taggers, the function is not provided integrated over eta. Only eta subranges are provided " << endl;
    return tmpSFl;
  }

  Double_t ptmax;
  if( sEtamin == "1.5" || sEtamin == "1.6" ) ptmax = 850;
  else ptmax = 1000;

// Insert function def below here =====================================

if( Atagger == "CSVL" && sEtamin == "0.0" && sEtamax == "0.5")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((1.01177+(0.0023066*x))+(-4.56052e-06*(x*x)))+(2.57917e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.977761+(0.00170704*x))+(-3.2197e-06*(x*x)))+(1.78139e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.04582+(0.00290226*x))+(-5.89124e-06*(x*x)))+(3.37128e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVL" && sEtamin == "0.5" && sEtamax == "1.0")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((0.975966+(0.00196354*x))+(-3.83768e-06*(x*x)))+(2.17466e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.945135+(0.00146006*x))+(-2.70048e-06*(x*x)))+(1.4883e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.00683+(0.00246404*x))+(-4.96729e-06*(x*x)))+(2.85697e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVL" && sEtamin == "1.0" && sEtamax == "1.5")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((0.93821+(0.00180935*x))+(-3.86937e-06*(x*x)))+(2.43222e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.911657+(0.00142008*x))+(-2.87569e-06*(x*x)))+(1.76619e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((0.964787+(0.00219574*x))+(-4.85552e-06*(x*x)))+(3.09457e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVL" && sEtamin == "1.5" && sEtamax == "2.4")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((1.00022+(0.0010998*x))+(-3.10672e-06*(x*x)))+(2.35006e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.970045+(0.000862284*x))+(-2.31714e-06*(x*x)))+(1.68866e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.03039+(0.0013358*x))+(-3.89284e-06*(x*x)))+(3.01155e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVM" && sEtamin == "0.0" && sEtamax == "0.8")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((1.07541+(0.00231827*x))+(-4.74249e-06*(x*x)))+(2.70862e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.964527+(0.00149055*x))+(-2.78338e-06*(x*x)))+(1.51771e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.18638+(0.00314148*x))+(-6.68993e-06*(x*x)))+(3.89288e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVM" && sEtamin == "0.8" && sEtamax == "1.6")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((1.05613+(0.00114031*x))+(-2.56066e-06*(x*x)))+(1.67792e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.946051+(0.000759584*x))+(-1.52491e-06*(x*x)))+(9.65822e-10*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.16624+(0.00151884*x))+(-3.59041e-06*(x*x)))+(2.38681e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVM" && sEtamin == "1.6" && sEtamax == "2.4")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((1.05625+(0.000487231*x))+(-2.22792e-06*(x*x)))+(1.70262e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.956736+(0.000280197*x))+(-1.42739e-06*(x*x)))+(1.0085e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.15575+(0.000693344*x))+(-3.02661e-06*(x*x)))+(2.39752e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVT" && sEtamin == "0.0" && sEtamax == "2.4")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((1.00462+(0.00325971*x))+(-7.79184e-06*(x*x)))+(5.22506e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.845757+(0.00186422*x))+(-4.6133e-06*(x*x)))+(3.21723e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.16361+(0.00464695*x))+(-1.09467e-05*(x*x)))+(7.21896e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVV1L" && sEtamin == "0.0" && sEtamax == "0.5")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((1.03599+(0.00187708*x))+(-3.73001e-06*(x*x)))+(2.09649e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.995735+(0.00146811*x))+(-2.83906e-06*(x*x)))+(1.5717e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.0763+(0.00228243*x))+(-4.61169e-06*(x*x)))+(2.61601e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVV1L" && sEtamin == "0.5" && sEtamax == "1.0")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((0.987393+(0.00162718*x))+(-3.21869e-06*(x*x)))+(1.84615e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.947416+(0.00130297*x))+(-2.50427e-06*(x*x)))+(1.41682e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.02741+(0.00194855*x))+(-3.92587e-06*(x*x)))+(2.27149e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVV1L" && sEtamin == "1.0" && sEtamax == "1.5")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((0.950146+(0.00150932*x))+(-3.28136e-06*(x*x)))+(2.06196e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.91407+(0.00123525*x))+(-2.61966e-06*(x*x)))+(1.63016e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((0.986259+(0.00178067*x))+(-3.93596e-06*(x*x)))+(2.49014e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVV1L" && sEtamin == "1.5" && sEtamax == "2.4")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((1.01923+(0.000898874*x))+(-2.57986e-06*(x*x)))+(1.8149e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.979782+(0.000743807*x))+(-2.14927e-06*(x*x)))+(1.49486e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.05868+(0.00105264*x))+(-3.00767e-06*(x*x)))+(2.13498e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVV1M" && sEtamin == "0.0" && sEtamax == "0.8")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((1.06383+(0.00279657*x))+(-5.75405e-06*(x*x)))+(3.4302e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.971686+(0.00195242*x))+(-3.98756e-06*(x*x)))+(2.38991e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.15605+(0.00363538*x))+(-7.50634e-06*(x*x)))+(4.4624e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVV1M" && sEtamin == "0.8" && sEtamax == "1.6")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((1.03709+(0.00169762*x))+(-3.52511e-06*(x*x)))+(2.25975e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.947328+(0.00117422*x))+(-2.32363e-06*(x*x)))+(1.46136e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.12687+(0.00221834*x))+(-4.71949e-06*(x*x)))+(3.05456e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVV1M" && sEtamin == "1.6" && sEtamax == "2.4")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((1.01679+(0.00211998*x))+(-6.26097e-06*(x*x)))+(4.53843e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.922527+(0.00176245*x))+(-5.14169e-06*(x*x)))+(3.61532e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.11102+(0.00247531*x))+(-7.37745e-06*(x*x)))+(5.46589e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVV1T" && sEtamin == "0.0" && sEtamax == "2.4")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((1.15047+(0.00220948*x))+(-5.17912e-06*(x*x)))+(3.39216e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.936862+(0.00149618*x))+(-3.64924e-06*(x*x)))+(2.43883e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.36418+(0.00291794*x))+(-6.6956e-06*(x*x)))+(4.33793e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVSLV1L" && sEtamin == "0.0" && sEtamax == "0.5")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((1.06344+(0.0014539*x))+(-2.72328e-06*(x*x)))+(1.47643e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((1.01168+(0.000950951*x))+(-1.58947e-06*(x*x)))+(7.96543e-10*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.11523+(0.00195443*x))+(-3.85115e-06*(x*x)))+(2.15307e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVSLV1L" && sEtamin == "0.5" && sEtamax == "1.0")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((1.0123+(0.00151734*x))+(-2.99087e-06*(x*x)))+(1.73428e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.960377+(0.00109821*x))+(-2.01652e-06*(x*x)))+(1.13076e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.06426+(0.0019339*x))+(-3.95863e-06*(x*x)))+(2.3342e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVSLV1L" && sEtamin == "1.0" && sEtamax == "1.5")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((0.975277+(0.00146932*x))+(-3.17563e-06*(x*x)))+(2.03698e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.931687+(0.00110971*x))+(-2.29681e-06*(x*x)))+(1.45867e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.0189+(0.00182641*x))+(-4.04782e-06*(x*x)))+(2.61199e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVSLV1L" && sEtamin == "1.5" && sEtamax == "2.4")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((1.04201+(0.000827388*x))+(-2.31261e-06*(x*x)))+(1.62629e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.992838+(0.000660673*x))+(-1.84971e-06*(x*x)))+(1.2758e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.09118+(0.000992959*x))+(-2.77313e-06*(x*x)))+(1.9769e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVSLV1M" && sEtamin == "0.0" && sEtamax == "0.8")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((1.06212+(0.00223614*x))+(-4.25167e-06*(x*x)))+(2.42728e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.903956+(0.00121678*x))+(-2.04383e-06*(x*x)))+(1.10727e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.22035+(0.00325183*x))+(-6.45023e-06*(x*x)))+(3.74225e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVSLV1M" && sEtamin == "0.8" && sEtamax == "1.6")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((1.04547+(0.00216995*x))+(-4.579e-06*(x*x)))+(2.91791e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.900637+(0.00120088*x))+(-2.27069e-06*(x*x)))+(1.40609e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.19034+(0.00313562*x))+(-6.87854e-06*(x*x)))+(4.42546e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVSLV1M" && sEtamin == "1.6" && sEtamax == "2.4")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((0.991865+(0.00324957*x))+(-9.65897e-06*(x*x)))+(7.13694e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.868875+(0.00222761*x))+(-6.44897e-06*(x*x)))+(4.53261e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.11481+(0.00426745*x))+(-1.28612e-05*(x*x)))+(9.74425e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "CSVSLV1T" && sEtamin == "0.0" && sEtamax == "2.4")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((1.09494+(0.00193966*x))+(-4.35021e-06*(x*x)))+(2.8973e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.813331+(0.00139561*x))+(-3.15313e-06*(x*x)))+(2.12173e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.37663+(0.00247963*x))+(-5.53583e-06*(x*x)))+(3.66635e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "JPL" && sEtamin == "0.0" && sEtamax == "0.5")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((0.991991+(0.000898777*x))+(-1.88002e-06*(x*x)))+(1.11276e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.930838+(0.000687929*x))+(-1.36976e-06*(x*x)))+(7.94486e-10*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.05319+(0.00110776*x))+(-2.38542e-06*(x*x)))+(1.42826e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "JPL" && sEtamin == "0.5" && sEtamax == "1.0")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((0.96633+(0.000419215*x))+(-9.8654e-07*(x*x)))+(6.30396e-10*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.904781+(0.000324913*x))+(-7.2229e-07*(x*x)))+(4.52185e-10*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.0279+(0.00051255*x))+(-1.24815e-06*(x*x)))+(8.07098e-10*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "JPL" && sEtamin == "1.0" && sEtamax == "1.5")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((0.968008+(0.000482491*x))+(-1.2496e-06*(x*x)))+(9.02736e-10*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.914619+(0.000330357*x))+(-8.41216e-07*(x*x)))+(6.14504e-10*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.02142+(0.000633484*x))+(-1.6547e-06*(x*x)))+(1.18921e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "JPL" && sEtamin == "1.5" && sEtamax == "2.4")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((0.991448+(0.000765746*x))+(-2.26144e-06*(x*x)))+(1.65233e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.933947+(0.000668609*x))+(-1.94474e-06*(x*x)))+(1.39774e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.04894+(0.000861785*x))+(-2.57573e-06*(x*x)))+(1.90702e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "JPM" && sEtamin == "0.0" && sEtamax == "0.8")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((0.991457+(0.00130778*x))+(-2.98875e-06*(x*x)))+(1.81499e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.822012+(0.000908344*x))+(-1.89516e-06*(x*x)))+(1.1163e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.16098+(0.00170403*x))+(-4.07382e-06*(x*x)))+(2.50873e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "JPM" && sEtamin == "0.8" && sEtamax == "1.6")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((1.00576+(0.00121353*x))+(-3.20601e-06*(x*x)))+(2.15905e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.845597+(0.000734909*x))+(-1.76311e-06*(x*x)))+(1.16104e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.16598+(0.00168902*x))+(-4.64013e-06*(x*x)))+(3.15214e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "JPM" && sEtamin == "1.6" && sEtamax == "2.4")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((0.939038+(0.00226026*x))+(-7.38544e-06*(x*x)))+(5.77162e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.803867+(0.00165886*x))+(-5.19532e-06*(x*x)))+(3.88441e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.07417+(0.00285862*x))+(-9.56945e-06*(x*x)))+(7.66167e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "JPT" && sEtamin == "0.0" && sEtamax == "2.4")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((0.953235+(0.00206692*x))+(-5.21754e-06*(x*x)))+(3.44893e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.642947+(0.00180129*x))+(-4.16373e-06*(x*x)))+(2.68061e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.26372+(0.0023265*x))+(-6.2548e-06*(x*x)))+(4.20761e-09*(x*(x*x)))", 20.,ptmax);
}
if( Atagger == "TCHPT" && sEtamin == "0.0" && sEtamax == "2.4")
{
if( meanminmax == "mean" ) tmpSFl = new TF1("SFlight","((1.20175+(0.000858187*x))+(-1.98726e-06*(x*x)))+(1.31057e-09*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "min" ) tmpSFl = new TF1("SFlightMin","((0.968557+(0.000586877*x))+(-1.34624e-06*(x*x)))+(9.09724e-10*(x*(x*x)))", 20.,ptmax);
if( meanminmax == "max" ) tmpSFl = new TF1("SFlightMax","((1.43508+(0.00112666*x))+(-2.62078e-06*(x*x)))+(1.70697e-09*(x*(x*x)))", 20.,ptmax);
}

// Insert function def above here =====================================
  
  if( tmpSFl == NULL ) cout << "NULL pointer returned... Function seems not to exist" << endl;

  return tmpSFl;
}

