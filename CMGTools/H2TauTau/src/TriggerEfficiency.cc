#include "CMGTools/H2TauTau/interface/TriggerEfficiency.h"
#include <TFile.h>
#include "Math/Minimizer.h"
#include "Math/Factory.h"
#include "Math/Functor.h"
#include <TGraph.h>


double TriggerEfficiency::efficiency(double m, double m0, double sigma, double alpha,double n, double norm) const {
  if(m<1. || 1000.<m)return 0.;//safety check

  const double sqrtPiOver2 = 1.2533141373;
  const double sqrt2 = 1.4142135624;
  double sig = fabs((double) sigma);
  double t = (m - m0)/sig;
  if(alpha < 0)
    t = -t;
  double absAlpha = fabs(alpha/sig);
  double a = TMath::Power(n/absAlpha,n)*exp(-0.5*absAlpha*absAlpha);
  double b = absAlpha - n/absAlpha;
  double ApproxErf;
  double arg = absAlpha / sqrt2;
  if (arg > 5.) ApproxErf = 1;
  else if (arg < -5.) ApproxErf = -1;
  else ApproxErf = TMath::Erf(arg);
  double leftArea = (1 + ApproxErf) * sqrtPiOver2;
  double rightArea = ( a * 1/TMath::Power(absAlpha - b,n-1)) / (n - 1);
  double area = leftArea + rightArea;
  if( t <= absAlpha ){
    arg = t / sqrt2;
    if(arg > 5.) ApproxErf = 1;
    else if (arg < -5.) ApproxErf = -1;
    else ApproxErf = TMath::Erf(arg);
    return norm * (1 + ApproxErf) * sqrtPiOver2 / area;
  }
  else{
    return norm * (leftArea + a * (1/TMath::Power(t-b,n-1) -  1/TMath::Power(absAlpha - b,n-1)) / (1 - n)) / area;
  }
}



double TriggerEfficiency::operator()(const double *xx) const {
  if(!chi2FunctorHisto)return 0.;

  double chi2=0.;
  for(int b=1;b<=chi2FunctorHisto->GetNbinsX();b++){
    if(chi2FunctorHisto->GetBinError(b)>0 
       && chi2FunctorHisto->GetBinContent(b)>0
       && (chi2FunctorHisto->GetBinCenter(b)>xmin_ || xmin_==0.)
       && (chi2FunctorHisto->GetBinCenter(b)<xmax_ || xmax_==0.)
       ){//skip over the bins with no content and bins out of desired fit range

      double diff = chi2FunctorHisto->GetBinContent(b) - efficiency(chi2FunctorHisto->GetBinCenter(b),xx[0],xx[1],xx[2],xx[3],xx[4]);
      chi2 += diff*diff/(chi2FunctorHisto->GetBinError(b)*chi2FunctorHisto->GetBinError(b));
    }
  }

  return chi2;
}

bool TriggerEfficiency::fitEfficiency(const char* filename,float xmin,float xmax){

  TFile File(filename,"read");
  chi2FunctorHisto=(TH1*)File.Get("efficiency");
  if(!chi2FunctorHisto)return 0;
  xmin_=xmin;
  xmax_=xmax;

  ROOT::Math::Minimizer* min = ROOT::Math::Factory::CreateMinimizer("Minuit2","");
  min->SetMaxFunctionCalls(200000); // for Minuit/Minuit2 
  min->SetMaxIterations(100000);  // for GSL 
  min->SetTolerance(0.0001);
  min->SetPrintLevel(0);
  
  ROOT::Math::Functor chi2(*this,5); 
  min->SetFunction(chi2); 

  double step[5] = {0.001,0.001,0.001,0.001,0.001};
  //double variable[5] = {18.80484409,0.19082817,0.19983010,1.81979820,0.93270649};
  double variable[5] = {18,0.19,0.19,1.8,0.9};
  min->SetVariable(0,"v0",variable[0], step[0]);
  min->SetVariable(1,"v1",variable[1], step[1]);
  min->SetVariable(2,"v2",variable[2], step[2]);
  min->SetVariable(3,"v3",variable[3], step[3]);
  min->SetVariable(4,"v4",variable[4], step[4]);
  
  min->Minimize(); 
  if(min->MinValue()>20) min->Minimize(); 
 
  const double *xx = min->X();
//   std::cout<<"Initial parameters:"<<std::endl;
//   std::cout<<variable[0]<<" , "<<variable[1]<<" , "<<variable[2]<<" , "<<variable[3]<<" , "<<variable[4]<<std::endl;
//   std::cout<<"Final parameters:"<<std::endl;
//   std::cout<<xx[0]<<" , "<<xx[1]<<" , "<<xx[2]<<" , "<<xx[3]<<" , "<<xx[4]<<std::endl;
//   std::cout<<"Chi2 = "<<min->MinValue()  << std::endl;
 

  //Save the result in a root file
  TFile FResults(TString("")+filename+"_Fit.root","recreate");
  TGraph Fit;
  for(Int_t p=0;p<100;p++){
    float x=chi2FunctorHisto->GetXaxis()->GetXmin() + p*(chi2FunctorHisto->GetXaxis()->GetXmax()-chi2FunctorHisto->GetXaxis()->GetXmin())/100.;
    Fit.SetPoint(p,x,efficiency(x,xx[0],xx[1],xx[2],xx[3],xx[4]));
  }
  Fit.SetLineColor(2);
  Fit.SetName("Fit");
  Fit.SetTitle("Fit Curve");
  Fit.Write();

  TH1F FitParameters("FitParameters","Parameters",5,0,5);
  FitParameters.SetBinContent(0,min->MinValue());
  FitParameters.SetBinContent(1,xx[0]);
  FitParameters.SetBinContent(2,xx[1]);
  FitParameters.SetBinContent(3,xx[2]);
  FitParameters.SetBinContent(4,xx[3]);
  FitParameters.SetBinContent(5,xx[4]);
  FitParameters.Write();

  chi2FunctorHisto->Write();

  //FResults.ls();
  FResults.Close();  


  File.Close();

  return 1;
}
