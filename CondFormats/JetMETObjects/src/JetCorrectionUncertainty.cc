#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"
#include "CondFormats/JetMETObjects/interface/SimpleJetCorrectorParameters.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <vector>
#include <string>

#include "Math/PtEtaPhiE4D.h"
#include "Math/LorentzVector.h"
typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > PtEtaPhiELorentzVectorD;
typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > XYZTLorentzVectorD;
using namespace std;
using namespace edm;

/////////////////////////////////////////////////////////////////////////
JetCorrectionUncertainty::JetCorrectionUncertainty () 
{
  mParameters = new SimpleJetCorrectorParameters();
}
/////////////////////////////////////////////////////////////////////////
JetCorrectionUncertainty::JetCorrectionUncertainty (const std::string& fDataFile)  
{
  std::string tmp = "CondFormats/JetMETObjects/data/"+fDataFile+".txt";
  edm::FileInPath f1("CondFormats/JetMETObjects/data/"+fDataFile+".txt");
  mParameters = new SimpleJetCorrectorParameters(f1.fullPath());
}
/////////////////////////////////////////////////////////////////////////
JetCorrectionUncertainty::~JetCorrectionUncertainty () 
{
  delete mParameters;
}
/////////////////////////////////////////////////////////////////////////
void JetCorrectionUncertainty::setParameters (const std::string& fDataFile) 
{
  //---- delete the mParameters pointer before setting the new address ---
  delete mParameters; 
  std::string tmp = "CondFormats/JetMETObjects/data/"+fDataFile+".txt";
  edm::FileInPath f1("CondFormats/JetMETObjects/data/"+fDataFile+".txt");
  mParameters = new SimpleJetCorrectorParameters(f1.fullPath());
}
/////////////////////////////////////////////////////////////////////////
double JetCorrectionUncertainty::uncertaintyPtEta (double fPt, double fEta, std::string fDirection) const 
{
  double result = 1.;
  int band = mParameters->bandIndex(fEta);
  if (band<0) 
    band = fEta<0 ? 0 : mParameters->size()-1;
  else if (band==0 || band==int(mParameters->size())-1)
    result = uncertaintyBandPtEta (band, fPt, fEta, fDirection);
  else
    { 
      double etaMiddle[3];
      double etaValue[3];
      for(int i=0; i<3; i++)
        {  
          etaMiddle[i] = mParameters->record (band+i-1).etaMiddle();
          etaValue[i]  = uncertaintyBandPtEta (band+i-1, fPt, fEta, fDirection);
          //std::cout<<etaMiddle[i]<<" "<<etaValue[i]<<std::endl;
        }
      result = quadraticInterpolation(fEta,etaMiddle,etaValue);  
    }
  return result;
}
/////////////////////////////////////////////////////////////////////////
double JetCorrectionUncertainty::uncertaintyXYZT (double fPx, double fPy, double fPz, double fE, std::string fDirection) const 
{
  XYZTLorentzVectorD p4 (fPx, fPy, fPz, fE);
  return uncertaintyPtEta (p4.Pt(), p4.Eta(), fDirection);
}
/////////////////////////////////////////////////////////////////////////
double JetCorrectionUncertainty::uncertaintyEtEtaPhiP (double fEt, double fEta, double fPhi, double fP, std::string fDirection) const 
{
  double costhetainv = cosh (fEta);
  return uncertaintyPtEta (fP/costhetainv, fEta, fDirection);
}
/////////////////////////////////////////////////////////////////////////
double JetCorrectionUncertainty::uncertaintyBandPtEta (unsigned fBand, double fPt, double fEta, std::string fDirection) const 
{
  if (fBand >= mParameters->size()) 
    throw cms::Exception ("JetCorrectionUncertainty")<<"wrong band: "<<fBand<<": only "<<mParameters->size()<<" is available"<<", eta = "<<fEta;
  const std::vector<float>& p = mParameters->record (fBand).parameters ();
  if ((p.size() % 3) != 0)
    throw cms::Exception ("JetCorrectionUncertainty")<<"wrong # of parameters: multiple of 3 expected, "<<p.size()<< " got";
  if ((fDirection != "UP") && (fDirection != "DOWN"))
    throw cms::Exception ("JetCorrectionUncertainty")<<"wrong error direction: "<<fDirection<< ". Choose \"UP\" or \"DOWN\"";
  std::vector<double> ptMiddle,uncertaintyUP,uncertaintyDOWN;
  unsigned int N = p.size()/3;
  unsigned int i,ind;
  int bin;
  double resultUP,resultDOWN,result;
  for(i=0;i<N;i++)
    {
      ind = 3*i;
      ptMiddle.push_back(p[ind]);
      uncertaintyDOWN.push_back(p[ind+1]);
      uncertaintyUP.push_back(p[ind+2]); 
    }
  if (fPt<=ptMiddle[0])
    {
      resultUP = uncertaintyUP[0];
      resultDOWN = uncertaintyDOWN[0];
    }  
  else if (fPt>=ptMiddle[N-1])
    {
      resultUP = uncertaintyUP[N-1];
      resultDOWN = uncertaintyDOWN[N-1];
    } 
  else
    {
      bin = findPtBin(ptMiddle,fPt); 
      resultUP = linearInterpolation(fPt,ptMiddle[bin],ptMiddle[bin+1],uncertaintyUP[bin],uncertaintyUP[bin+1]);
      resultDOWN = linearInterpolation(fPt,ptMiddle[bin],ptMiddle[bin+1],uncertaintyDOWN[bin],uncertaintyDOWN[bin+1]);
    }
  if (fDirection=="UP")
    result = resultUP;
  else
    result = resultDOWN;
  return result;
}
/////////////////////////////////////////////////////////////////////////
double JetCorrectionUncertainty::quadraticInterpolation(double z, const double x[3], const double y[3]) const
{
  // Quadratic interpolation through the points (x[i],y[i]). First find the parabola that
  // is defined by the points and then calculate the y(z).
  double D[4],a[3];
  D[0] = x[0]*x[1]*(x[0]-x[1])+x[1]*x[2]*(x[1]-x[2])+x[2]*x[0]*(x[2]-x[0]);
  D[3] = y[0]*(x[1]-x[2])+y[1]*(x[2]-x[0])+y[2]*(x[0]-x[1]);
  D[2] = y[0]*(pow(x[2],2)-pow(x[1],2))+y[1]*(pow(x[0],2)-pow(x[2],2))+y[2]*(pow(x[1],2)-pow(x[0],2));
  D[1] = y[0]*x[1]*x[2]*(x[1]-x[2])+y[1]*x[0]*x[2]*(x[2]-x[0])+y[2]*x[0]*x[1]*(x[0]-x[1]);
  if (D[0] != 0)
    {
      a[0] = D[1]/D[0];
      a[1] = D[2]/D[0];
      a[2] = D[3]/D[0];
    }
  else
    {
      a[0] = 0;
      a[1] = 0;
      a[2] = 0;
    }
  double r = a[0]+z*(a[1]+z*a[2]);
  return r;
}
/////////////////////////////////////////////////////////////////////////
double JetCorrectionUncertainty::linearInterpolation(double z, const double x1, const double x2, const double y1, const double y2) const
{
  // Linear interpolation through the points (x[i],y[i]). First find the line that
  // is defined by the points and then calculate the y(z).
  double a,b,r;
  r = 0;
  if (x1 == x2)
    {
      if (y1 == y2)
        r = y1;
      else
        std::cout<<"ERROR!!!"<<std::endl;
    } 
  else   
    {
      a = (y2-y1)/(x2-x1);
      b = (y1*x2-y2*x1)/(x2-x1);
      r = a*z+b;
    }
  return r;
}
/////////////////////////////////////////////////////////////////////////
int JetCorrectionUncertainty::findPtBin(std::vector<double> v, double x) const
{
  int i;
  int n = v.size()-1;
  if (n<=0) return -1;
  if (x<v[0] || x>=v[n])
    return -1;
  for(i=0;i<n;i++)
   {
     if (x>=v[i] && x<v[i+1])
       return i;
   }
  return 0; 
}


