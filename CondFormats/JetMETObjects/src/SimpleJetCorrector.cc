#include "CondFormats/JetMETObjects/interface/SimpleJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include <iostream>
#include <vector>
#include <cmath>
#include "FWCore/Utilities/interface/Exception.h"

//------------------------------------------------------------------------ 
//--- Default SimpleJetCorrector constructor -----------------------------
//------------------------------------------------------------------------
SimpleJetCorrector::SimpleJetCorrector() 
{ 
  mFunc            = new TFormula(); 
  mParameters      = new JetCorrectorParameters();
  mDoInterpolation = false;
}
//------------------------------------------------------------------------ 
//--- SimpleJetCorrector constructor -------------------------------------
//--- reads arguments from a file ----------------------------------------
//------------------------------------------------------------------------
SimpleJetCorrector::SimpleJetCorrector(const std::string& fDataFile, const std::string& fOption) 
{
  mParameters      = new JetCorrectorParameters(fDataFile,fOption);
  mFunc            = new TFormula("function",((mParameters->definitions()).formula()).c_str());
  mDoInterpolation = false;
}
//------------------------------------------------------------------------ 
//--- SimpleJetCorrector destructor --------------------------------------
//------------------------------------------------------------------------
SimpleJetCorrector::~SimpleJetCorrector() 
{
  delete mFunc;
  delete mParameters;
}
//------------------------------------------------------------------------ 
//--- calculates the correction ------------------------------------------
//------------------------------------------------------------------------
double SimpleJetCorrector::correction(const std::vector<float>& fX,const std::vector<float>& fY) const 
{
  double result = 1.;
  double tmp    = 0.0;
  double cor    = 0.0;
  int bin = mParameters->binIndex(fX);
  if (bin<0) 
    throw cms::Exception("SimpleJetCorrector")<<" bin variables out of range";
  if (!mDoInterpolation)
    result = correctionBin(bin,fY);
  else
    { 
      for(unsigned i=0;i<mParameters->definitions().nBinVar();i++)
        { 
          double xMiddle[3];
          double xValue[3];
          int prevBin = mParameters->neighbourBin((unsigned)bin,i,false);
          int nextBin = mParameters->neighbourBin((unsigned)bin,i,true);
          if (prevBin>=0 && nextBin>=0)
            { 
              xMiddle[0] = mParameters->record(prevBin).xMiddle(i);
              xMiddle[1] = mParameters->record(bin).xMiddle(i);
              xMiddle[2] = mParameters->record(nextBin).xMiddle(i);
              xValue[0]  = correctionBin(prevBin,fY);
              xValue[1]  = correctionBin(bin,fY);
              xValue[2]  = correctionBin(nextBin,fY);
              cor = quadraticInterpolation(fX[i],xMiddle,xValue);
              tmp+=cor;
            }
          else
            {
              cor = correctionBin(bin,fY);
              tmp+=cor;
            }
          //std::cout<<"Bin: "<<bin<<", Previous Bin: "<<prevBin<<", Next Bin: "<<nextBin<<std::endl;
          //std::cout<<"Interpolation var"<<i<<" "<<cor<<" "<<tmp<<std::endl;
        }
      result = tmp/mParameters->definitions().nBinVar();        
    }
  return result;
}
//------------------------------------------------------------------------ 
//--- calculates the correction for a specific bin -----------------------
//------------------------------------------------------------------------
double SimpleJetCorrector::correctionBin(unsigned fBin,const std::vector<float>& fY) const 
{
  if (fBin >= mParameters->size()) 
    throw cms::Exception("SimpleJetCorrector")<<" wrong bin: "<<fBin<<": only "<<mParameters->size()<<" are available";
  double result = -1;
  const std::vector<float>& par = mParameters->record(fBin).parameters();
  unsigned N = fY.size();
  for(unsigned int i=2*N;i<par.size();i++)
    mFunc->SetParameter(i-2*N,par[i]);
  if (N==1)
    {
      double x = (fY[0] < par[0]) ? par[0] : (fY[0] > par[1]) ? par[1] : fY[0];
      result = mFunc->Eval(x);
    } 
  if (N==2)
    {
      double x = (fY[0] < par[0]) ? par[0] : (fY[0] > par[1]) ? par[1] : fY[0];
      double y = (fY[1] < par[2]) ? par[2] : (fY[1] > par[3]) ? par[3] : fY[1];
      result = mFunc->Eval(x,y);
    }
  if (N==3)
    {
      double x = (fY[0] < par[0]) ? par[0] : (fY[0] > par[1]) ? par[1] : fY[0];
      double y = (fY[1] < par[2]) ? par[2] : (fY[1] > par[3]) ? par[3] : fY[1];
      double z = (fY[2] < par[4]) ? par[4] : (fY[2] > par[5]) ? par[5] : fY[2]; 
      result = mFunc->Eval(x,y,z);
    }
  if (N==4)
    {
      double x = (fY[0] < par[0]) ? par[0] : (fY[0] > par[1]) ? par[1] : fY[0];
      double y = (fY[1] < par[2]) ? par[2] : (fY[1] > par[3]) ? par[3] : fY[1];
      double z = (fY[2] < par[4]) ? par[4] : (fY[2] > par[5]) ? par[5] : fY[2];
      double t = (fY[3] < par[6]) ? par[6] : (fY[3] > par[7]) ? par[7] : fY[3]; 
      result = mFunc->Eval(x,y,z,t);
    }
  return result;
}
//------------------------------------------------------------------------ 
//--- quadratic interpolation --------------------------------------------
//------------------------------------------------------------------------
double SimpleJetCorrector::quadraticInterpolation(double fZ, const double fX[3], const double fY[3]) const
{
  // Quadratic interpolation through the points (x[i],y[i]). First find the parabola that
  // is defined by the points and then calculate the y(z).
  double D[4],a[3];
  D[0] = fX[0]*fX[1]*(fX[0]-fX[1])+fX[1]*fX[2]*(fX[1]-fX[2])+fX[2]*fX[0]*(fX[2]-fX[0]);
  D[3] = fY[0]*(fX[1]-fX[2])+fY[1]*(fX[2]-fX[0])+fY[2]*(fX[0]-fX[1]);
  D[2] = fY[0]*(pow(fX[2],2)-pow(fX[1],2))+fY[1]*(pow(fX[0],2)-pow(fX[2],2))+fY[2]*(pow(fX[1],2)-pow(fX[0],2));
  D[1] = fY[0]*fX[1]*fX[2]*(fX[1]-fX[2])+fY[1]*fX[0]*fX[2]*(fX[2]-fX[0])+fY[2]*fX[0]*fX[1]*(fX[0]-fX[1]);
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
  double r = a[0]+fZ*(a[1]+fZ*a[2]);
  return r;
}
