/* 
 *  \class TAPD
 *
 *  $Date: 2010/04/12 14:17:11 $
 *  \author: Julie Malcles  - CEA/Saclay
 */

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMom.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TAPD.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMarkov.h>
#include <TMath.h>

using namespace std;
#include <iostream>
#include <cassert>
//ClassImp(TAPD)


// Default Constructor...
TAPD::TAPD()
{
  init();
}


// Destructor
TAPD::~TAPD()
{
}

void TAPD::init()
{

  
  default_max[iAPD]=10000.0;
  default_max[iAPDoPN]=20.0;
  default_max[iAPDoPN0]=20.0;
  default_max[iAPDoPN1]=20.0;
  default_max[iTime]=10.0;
  default_max[iAPDoAPD]=20.0;
  default_max[iAPDoAPD0]=20.0;
  default_max[iAPDoAPD1]=20.0;
  default_max[iAPDoPNCor]=20.0;
  default_max[iAPDoPN0Cor]=20.0;
  default_max[iAPDoPN1Cor]=20.0;


  for(int j=0;j<nOutVar;j++){

    _apdcuts[0][j].clear();
    _apdcuts[1][j].clear();
    _cutvars[j].clear();

    _apdcuts[0][j].push_back(0.0);
    _apdcuts[1][j].push_back(10.0e6);

    _cutvars[j].push_back(j);
    
    mom[j]=new TMom();
  }
}

void TAPD::addEntry(double apd, double pn0, double pn1, double time)
{
  addEntry(apd, pn0, pn1, time, 0.0, 0.0, 0.0, 0.0);
}

void TAPD::addEntry(double apd, double pn0, double pn1, double time, double apda, double apdb)
{
  addEntry(apd, pn0, pn1, time, apda , apdb, 0.0, 0.0 );

}
void TAPD::addEntry(double apd, double pn0, double pn1, double time, double apd0, double apd1, double pncor0, double pncor1 )
{
  double pn, pncor;
  double apdref;
  
  if (pn0<50 && pn1>50) {
    pn=pn1;
  }else if (pn1<50 && pn0>50){
    pn=pn0;
  }else pn=0.5*(pn0+pn1);

  if (pncor0<50 && pncor1>50) {
    pncor=pncor1;
  }else if (pncor1<50 && pncor0>50){
    pncor=pncor0;
  }else pncor=0.5*(pncor0+pncor1);

  
  if (apd0<50 && apd1>50) {
    apdref=apd1;
  }else if (apd1<50 && apd0>50){
    apdref=apd0;
  }else apdref=0.5*(apd0+apd1);
  
  double val[nOutVar];
  std::vector <double> valcuts[nOutVar];

  val[iAPD]=apd;
  
  if(pn!=0) val[iAPDoPN]=apd/pn;
  else val[iAPDoPN]=0.0;
  if(pn0!=0) val[iAPDoPN0]=apd/pn0;
  else val[iAPDoPN0]=0.0;
  if(pn1!=0) val[iAPDoPN1]=apd/pn1;
  else val[iAPDoPN1]=0.0;
  val[iTime]=time;
  
  if(apd0!=0.)val[iAPDoAPD0]=apd/apd0;
  else val[iAPDoAPD0]=0.0;
  if(apd1!=0.)val[iAPDoAPD1]=apd/apd1;
  else val[iAPDoAPD1]=0.0;
  if(apdref!=0.)val[iAPDoAPD]=apd/apdref;
  else val[iAPDoAPD]=0.0;
  
  if(pncor!=0) val[iAPDoPNCor]=apd/pncor;
  else val[iAPDoPNCor]=0.0;
  if(pncor0!=0) val[iAPDoPN0Cor]=apd/pncor0;
  else val[iAPDoPN0Cor]=0.0;
  if(pncor1!=0) val[iAPDoPN1Cor]=apd/pncor1;
  else val[iAPDoPN1Cor]=0.0;
  
  for(int ivar=0;ivar<nOutVar;ivar++){
    int dimcut=_cutvars[ivar].size();
    for(int ic=0;ic<dimcut;ic++){
      assert(_cutvars[ivar][ic]<nOutVar);
      valcuts[ivar].push_back(val[_cutvars[ivar][ic]]);
    }
  }

  for(int ivar=0;ivar<nOutVar;ivar++){
    mom[ivar]->addEntry(val[ivar],valcuts[ivar]); 
  }
  
}
  
void TAPD::setCut(int ivar, double mean, double sig){
  
  assert(ivar<nOutVar);
  
  std::vector <int>cutvar;
  cutvar.push_back(ivar);
  
  std::vector <double>lowcut;
  std::vector <double>highcut;

  double low=mean-2.0*sig;
  if (low<0) low=0.0;
  double high=mean+2.0*sig;

  lowcut.push_back(low);
  highcut.push_back(high);

  setCut(ivar,cutvar,lowcut,highcut);  

}

void TAPD::setCut(int ivar, std::vector<int> cutVars, std::vector<double> lowCut, std::vector<double> highCut){
  
  assert(ivar<nOutVar);
  unsigned int cutdim=cutVars.size();
  assert(cutdim<nOutVar);
  assert(cutdim==lowCut.size());
  assert(cutdim==highCut.size());
  
  _apdcuts[0][ivar].clear();
  _apdcuts[1][ivar].clear();
  _cutvars[ivar].clear();
  
  for (unsigned int ic=0;ic<cutdim;ic++){
    
    // FINISH THIS
    if(lowCut[ic]>0){
      _apdcuts[0][ivar].push_back(lowCut[ic]);
    }else _apdcuts[0][ivar].push_back(0.0);
    
    _apdcuts[1][ivar].push_back(highCut[ic]);
    _cutvars[ivar].push_back(cutVars[ic]);

  }
 
  mom[ivar]->setCut(_apdcuts[0][ivar],_apdcuts[1][ivar]);
}

// Simple 1D cuts on main variable at 2 sigmas
// ===========================================

void  TAPD::setAPDCut(double mean, double sig){setCut(TAPD::iAPD,mean,sig);}
void  TAPD::setAPDoPNCut(double mean, double sig){setCut(TAPD::iAPDoPN,mean,sig);}
void  TAPD::setAPDoPN0Cut(double mean, double sig){setCut(TAPD::iAPDoPN0,mean,sig);}
void  TAPD::setAPDoPN1Cut(double mean, double sig){setCut(TAPD::iAPDoPN1,mean,sig);}

void  TAPD::setAPDoPNCorCut(double mean, double sig){setCut(TAPD::iAPDoPNCor,mean,sig);}
void  TAPD::setAPDoPN0CorCut(double mean, double sig){setCut(TAPD::iAPDoPN0Cor,mean,sig);}
void  TAPD::setAPDoPN1CorCut(double mean, double sig){setCut(TAPD::iAPDoPN1Cor,mean,sig);}

void  TAPD::setTimeCut(double mean, double sig){setCut(TAPD::iTime,mean,sig);}

// More complicated 2D cuts
// ========================= 

// Cut on main var and Time:
void  TAPD::set2DCut(int ivar, std::vector<double> lowCut,std::vector<double> highCut){
  
  assert (lowCut.size()==2);
  assert (highCut.size()==2);
  std::vector<int> cutVars;
  cutVars.push_back(ivar);
  cutVars.push_back(TAPD::iTime); 
  setCut(ivar, cutVars, lowCut, highCut);
  
}


void  TAPD::set2DAPDCut(std::vector<double> lowCut,std::vector<double> highCut){
  set2DCut(TAPD::iAPD, lowCut, highCut);
}
void  TAPD::set2DAPDoPNCut(std::vector<double> lowCut,std::vector<double> highCut){
  set2DCut(TAPD::iAPDoPN, lowCut, highCut);
}
void  TAPD::set2DAPDoPN0Cut(std::vector<double> lowCut,std::vector<double> highCut){
  set2DCut(TAPD::iAPDoPN0, lowCut, highCut);
}
void  TAPD::set2DAPDoPN1Cut(std::vector<double> lowCut,std::vector<double> highCut){
  set2DCut(TAPD::iAPDoPN1, lowCut, highCut);
}


void  TAPD::set2DAPDoAPD0Cut(std::vector<double> lowCut,std::vector<double> highCut){

  assert (lowCut.size()==2);
  assert (highCut.size()==2);
  std::vector<int> cutVars;
  cutVars.push_back(TAPD::iAPD);
  cutVars.push_back(TAPD::iTime); 
  setCut(TAPD::iAPDoAPD0, cutVars, lowCut, highCut);
}
void  TAPD::set2DAPDoAPD1Cut(std::vector<double> lowCut,std::vector<double> highCut){

  assert (lowCut.size()==2);
  assert (highCut.size()==2);
  std::vector<int> cutVars;
  cutVars.push_back(TAPD::iAPD);
  cutVars.push_back(TAPD::iTime); 
  setCut(TAPD::iAPDoAPD1, cutVars, lowCut, highCut);
}
void  TAPD::set2DAPDoAPDCut(std::vector<double> lowCut,std::vector<double> highCut){

  assert (lowCut.size()==2);
  assert (highCut.size()==2);
  std::vector<int> cutVars;
  cutVars.push_back(TAPD::iAPD);
  cutVars.push_back(TAPD::iTime); 
  setCut(TAPD::iAPDoAPD, cutVars, lowCut, highCut);
}

void  TAPD::set2DTimeCut(std::vector<double> lowCut,std::vector<double> highCut){

  assert (lowCut.size()==2);
  assert (highCut.size()==2);
  std::vector<int> cutVars;
  cutVars.push_back(TAPD::iAPD);
  cutVars.push_back(TAPD::iTime); 
  setCut(TAPD::iTime, cutVars, lowCut, highCut);
}



vector<double> TAPD::get(int ivar){ 

  vector<double> res;
  
  if(ivar<nOutVar){

    res.push_back(mom[ivar]->getMean());
    res.push_back(mom[ivar]->getRMS());
    res.push_back(mom[ivar]->getM3());
    res.push_back(mom[ivar]->getNevt());
    res.push_back(mom[ivar]->getMin());
    res.push_back(mom[ivar]->getMax());

  }

  //  cout << "In get: ivar="<< ivar << ", mean="<< mom[ivar]->getMean()<<" res size="<< res.size()<< endl;

  return res;

}

vector<double>   TAPD::getAPD(){vector<double> x=get(TAPD::iAPD); return x;}
vector<double>   TAPD::getAPDoPN(){vector<double> x=get(TAPD::iAPDoPN); return x;}
vector<double>   TAPD::getAPDoPN0(){vector<double> x=get(TAPD::iAPDoPN0); return x;}
vector<double>   TAPD::getAPDoPN1(){vector<double> x=get(TAPD::iAPDoPN1); return x;}
vector<double>   TAPD::getTime(){vector<double> x=get(TAPD::iTime); return x;}
vector<double>   TAPD::getAPDoAPD0(){vector<double> x=get(TAPD::iAPDoAPD0); return x;}
vector<double>   TAPD::getAPDoAPD1(){vector<double> x=get(TAPD::iAPDoAPD1); return x;}
vector<double>   TAPD::getAPDoAPD(){vector<double> x=get(TAPD::iAPDoAPD); return x;}

vector<double>   TAPD::getAPDoPNCor(){vector<double> x=get(TAPD::iAPDoPNCor); return x;}
vector<double>   TAPD::getAPDoPN0Cor(){vector<double> x=get(TAPD::iAPDoPN0Cor); return x;}
vector<double>   TAPD::getAPDoPN1Cor(){vector<double> x=get(TAPD::iAPDoPN1Cor); return x;}


// double*  TAPD::getAPD2(){
  
//   double* res=get2(TAPD::iAPD);
//   return res;

// }
// double* TAPD::get2(int ivar){ 
  
//   static double res[VARSIZE];

//   res[0]=mom[ivar]->getMean();
//   res[1]=mom[ivar]->getRMS();
//   res[2]=mom[ivar]->getM3();
//   res[3]=mom[ivar]->getNevt();
//   res[4]=mom[ivar]->getMin();
//   res[5]=mom[ivar]->getMax();

//   return res;

// }
