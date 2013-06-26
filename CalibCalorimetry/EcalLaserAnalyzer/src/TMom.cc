/* 
 *  \class TMom
 *
 *  $Date: 2013/04/19 22:19:23 $
 *  \author: Julie Malcles - CEA/Saclay
 */

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMom.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMarkov.h>
#include <TMath.h>

#include <cassert>

using namespace std;

//ClassImp(TMom)


// Default Constructor...
TMom::TMom()
{
  init(0.0,10.0e9);
}

// Constructor...
TMom::TMom(double cutlow, double cuthigh)
{
  init(cutlow,cuthigh);
}
TMom::TMom(const std::vector<double>& cutlow, const std::vector<double>& cuthigh)
{
  init(cutlow,cuthigh);
}

// Destructor
TMom::~TMom()
{
}

void TMom::init(double cutlow, double cuthigh)
{

  nevt=0;
  mean=0;
  mean2=0;
  mean3=0;
  sum=0;
  sum2=0;
  sum3=0;
  rms=0;
  M3=0;
  peak=0;
  min=10.0e9;
  max=0.;
  _cutLow.clear();
  _cutHigh.clear();
  _dimCut=1;
  _cutLow.push_back(cutlow);
  _cutHigh.push_back(cuthigh);
  for(int i=0;i<101;i++){
    bing[i]=0;
  }
  
}
void TMom::init(const std::vector<double>& cutlow, const std::vector<double>& cuthigh)
{

  nevt=0;
  mean=0;
  mean2=0;
  mean3=0;
  sum=0;
  sum2=0;
  sum3=0;
  rms=0;
  M3=0;
  peak=0;
  min=10.0e9;
  max=0.;
  assert(cutlow.size()==cuthigh.size());  
  _cutLow.clear();
  _cutHigh.clear();
  _dimCut=cutlow.size();
  _cutLow=cutlow;
  _cutHigh=cuthigh;
  for(int i=0;i<101;i++){
    bing[i]=0;
  }
  
}
void TMom::setCut(double cutlow, double cuthigh){

  _cutLow.clear();
  _cutHigh.clear();
  _dimCut=1;
  _cutLow.push_back(cutlow);
  _cutHigh.push_back(cuthigh);

}
void TMom::setCut(const std::vector<double>& cutlow ,const std::vector<double>& cuthigh){
  
  assert(cutlow.size( )== cuthigh.size());
  _cutLow.clear();
  _cutHigh.clear();
  _dimCut=cutlow.size();
  _cutLow=cutlow;
  _cutHigh=cuthigh;
  
}

void TMom::addEntry(double val)
{
  std::vector<double> dumb;
  dumb.push_back(val);
  addEntry(val,dumb);
}
  
void TMom::addEntry(double val, const std::vector<double>& valcut)
{
  
  int passingAllCuts=1;
  
  for (int iCut=0;iCut<_dimCut;iCut++){
    int passing;
    if( valcut.at(iCut)>_cutLow.at(iCut) && valcut.at(iCut) <=_cutHigh.at(iCut) ){
      passing=1;
    }else passing=0;
    passingAllCuts*=passing; 
  }
  
  if( passingAllCuts == 1 ){
    
    nevt+=1;
    sum+=val;
    sum2+=val*val;
    sum3+=val*val*val;
    if(val>max) max=val;
    if(val<min) min=val;
    
    // for peak stuff 
    _ampl.push_back(val);
  }
  
}



double  TMom::getMean(){
  if(nevt!=0) mean=sum/nevt;
  else mean=0.0;
  return mean;
}

double  TMom::getMean2(){
  if(nevt!=0) mean2=sum2/nevt;
  else mean2=0.0;
  return mean2;
}
double  TMom::getMean3(){
  if(nevt!=0) mean3=sum3/nevt;
  else mean3=0.0;
  return mean3;
}

int  TMom::getNevt(){ return nevt;}

double  TMom::getRMS(){
  double m=getMean(); 
  double m2=getMean2(); 
  if(nevt!=0) rms=TMath::Sqrt( m2 - m*m );
  else rms=0.0;
  return rms;
}

double TMom::getM3(){

  double m=getMean(); 
  double m2=getMean2();
  double m3=getMean3();
  double sig = getRMS();
  
  if(nevt!=0 && sig!=0) M3=( m3 - 3.0*m*(m2-m*m) - m*m*m )/(sig*sig*sig); 
  else M3=0.0;
  return M3;
}

double TMom::getMin(){return min;}
double  TMom::getMax(){return max;}

std::vector<double> TMom::getPeak(){
  
  std::vector<double> p;
  double wbin=(max-min)/100.;
  int bung;
  
  for(unsigned int i=0;i<_ampl.size();i++){
    if(wbin <= 0.0)
      bung=1;
    else
      bung= (int) ((_ampl.at(i)-min)/wbin)+1;
    if(1 <= bung && bung <= 100)
      bing[bung]++;
  }
  
  TMarkov *peakM = new TMarkov();

  int popmax=0;
  
  for(int k=1;k<101;k++) {
    if(bing[k] > popmax) {
      popmax=bing[k];
    }
  }
  
  peakM -> peakFinder(&bing[0]);
  p.push_back(peakM -> getPeakValue(0));
  p.push_back(peakM -> getPeakValue(1));
        
  return p;
}

