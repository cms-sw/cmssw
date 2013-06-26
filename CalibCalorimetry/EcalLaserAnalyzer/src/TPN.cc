/* 
 *  \class TPN
 *
 *  $Date: 2012/02/09 10:08:10 $
 *  \author: Julie Malcles - CEA/Saclay
 */

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMom.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TPN.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMarkov.h>
#include <TMath.h>

using namespace std;

//ClassImp(TPN)


// Default Constructor...
TPN::TPN(int iPN)
{
  init();
  _nPN=iPN;
}


// Destructor
TPN::~TPN()
{
}

void TPN::init()
{

  for(int j=0;j<nOutVar;j++){
    cuts[0][j]=0.0;
    cuts[1][j]=10.0e9;
    mom[j]=new TMom();
  }
}

void TPN::addEntry(double pn, double pn0, double pn1)
{

  double val[nOutVar];
 
  if(_nPN==0) val[iPN]=pn0;
  else val[iPN]=pn1;

  if(pn!=0) val[iPNoPN]=val[iPN]/pn;
  else val[iPNoPN]=0;

  if(pn0!=0) val[iPNoPN0]=val[iPN]/pn0;
  else val[iPNoPN0]=0;

  if(pn1!=0) val[iPNoPN1]=val[iPN]/pn1;
  else val[iPNoPN1]=0;
  
  for(int ivar=0;ivar<nOutVar;ivar++){
    mom[ivar]->addEntry(val[ivar]);
  }
  
}
  
void  TPN::setCut(int ivar, double mean, double sig){

  cuts[0][ivar]=mean-2.0*sig;
  cuts[1][ivar]=mean+2.0*sig;
  if(cuts[0][ivar]<0)cuts[0][ivar]=0.0 ;

  mom[ivar]->setCut(cuts[0][ivar],cuts[1][ivar]);
}

void  TPN::setPNCut(double mean, double sig){setCut(TPN::iPN,mean,sig);}
void  TPN::setPNoPNCut(double mean, double sig){setCut(TPN::iPNoPN,mean,sig);}
void  TPN::setPNoPN0Cut(double mean, double sig){setCut(TPN::iPNoPN0,mean,sig);}
void  TPN::setPNoPN1Cut(double mean, double sig){setCut(TPN::iPNoPN1,mean,sig);}


std::vector<double> TPN::get(int ivar){ 
  
  std::vector<double> res;
  
  if(ivar<nOutVar){
    
    res.push_back(mom[ivar]->getMean());
    res.push_back(mom[ivar]->getRMS());
    res.push_back(mom[ivar]->getM3());
    res.push_back(mom[ivar]->getNevt());
    res.push_back(mom[ivar]->getMin());
    res.push_back(mom[ivar]->getMax());
  }
  
  return res;
  
}

std::vector<double>   TPN::getPN(){vector <double> x= get(TPN::iPN); return x;}
std::vector<double>   TPN::getPNoPN(){vector <double> x= get(TPN::iPNoPN); return x;}
std::vector<double>   TPN::getPNoPN0(){vector <double> x= get(TPN::iPNoPN0); return x;}
std::vector<double>   TPN::getPNoPN1(){vector <double> x= get(TPN::iPNoPN1); return x;}

