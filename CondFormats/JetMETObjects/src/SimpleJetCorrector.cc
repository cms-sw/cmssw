#include "CondFormats/JetMETObjects/interface/SimpleJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include <iostream>
#include <vector>
#include <cmath>

#ifdef STANDALONE
#include <sstream>
#include <stdexcept>
#else
#include "FWCore/Utilities/interface/Exception.h"
#endif

//------------------------------------------------------------------------ 
//--- Default SimpleJetCorrector constructor -----------------------------
//------------------------------------------------------------------------
SimpleJetCorrector::SimpleJetCorrector() 
{ 
  mFunc            = new TFormula(); 
  mParameters      = new JetCorrectorParameters();
  mDoInterpolation = false;
  mInvertVar       = 9999;
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
  mInvertVar       = 9999; 
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
float SimpleJetCorrector::correction(const std::vector<float>& fX,const std::vector<float>& fY) const 
{
  float result = 1.;
  float tmp    = 0.0;
  float cor    = 0.0;
  int bin = mParameters->binIndex(fX);
  if (bin<0) 
    {
      #ifdef STANDALONE
         std::stringstream sserr; 
         sserr<<"SimpleJetCorrector ERROR: bin variables out of range!";
         throw std::runtime_error(sserr.str());
      #else
         throw cms::Exception("SimpleJetCorrector")<<" bin variables out of range";
      #endif
    }
  if (!mDoInterpolation)
    result = correctionBin(bin,fY);
  else
    { 
      for(unsigned i=0;i<mParameters->definitions().nBinVar();i++)
        { 
          float xMiddle[3];
          float xValue[3];
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
        }
      result = tmp/mParameters->definitions().nBinVar();        
    }
  return result;
}
//------------------------------------------------------------------------ 
//--- calculates the correction for a specific bin -----------------------
//------------------------------------------------------------------------
float SimpleJetCorrector::correctionBin(unsigned fBin,const std::vector<float>& fY) const 
{
  if (fBin >= mParameters->size()) 
    {
      #ifdef STANDALONE
         std::stringstream sserr; 
         sserr<<"SimpleJetCorrector ERROR: wrong bin: "<<fBin<<": only "<<mParameters->size()<<" available!";
         throw std::runtime_error(sserr.str());
      #else
         throw cms::Exception("SimpleJetCorrector")<<" wrong bin: "<<fBin<<": only "<<mParameters->size()<<" are available";
      #endif
    }
  unsigned N = fY.size();
  if (N > 4)
    {
      #ifdef STANDALONE 
         std::stringstream sserr;
         sserr<<"SimpleJetCorrector ERROR: two many variables: "<<N<<" maximum is 4";
         throw std::runtime_error(sserr.str());
      #else
         throw cms::Exception("SimpleJetCorrector")<<" two many variables: "<<N<<" maximum is 4";
      #endif
    } 
  float result = -1;
  const std::vector<float>& par = mParameters->record(fBin).parameters();
  for(unsigned int i=2*N;i<par.size();i++)
    mFunc->SetParameter(i-2*N,par[i]);
  float x[4];
  std::vector<float> tmp;
  for(unsigned i=0;i<N;i++)
    {
      x[i] = (fY[i] < par[2*i]) ? par[2*i] : (fY[i] > par[2*i+1]) ? par[2*i+1] : fY[i];
      tmp.push_back(x[i]);
    }
  if (mParameters->definitions().isResponse())
    result = invert(tmp);
  else
    result = mFunc->Eval(x[0],x[1],x[2],x[3]);  
  return result;
}
//------------------------------------------------------------------------ 
//--- set inversion ------------------------------------------------------
//------------------------------------------------------------------------
void SimpleJetCorrector::doInversion(unsigned fVar)
{
  if (mParameters->definitions().isResponse())
    mInvertVar = fVar;
  else
    {
      #ifdef STANDALONE 
        std::stringstream sserr;
        sserr<<"SimpleJetCorrector ERROR: inversion is applicable only when response is given";
        throw std::runtime_error(sserr.str());
      #else
        throw cms::Exception("SimpleJetCorrector")<<" inversion is applicable only when response is given";
      #endif
    } 
}
//------------------------------------------------------------------------ 
//--- inversion ----------------------------------------------------------
//------------------------------------------------------------------------
float SimpleJetCorrector::invert(std::vector<float> fX) const
{
  unsigned nMax = 50;
  unsigned N = fX.size();
  if (mInvertVar > N-1) 
    { 
      #ifdef STANDALONE 
         std::stringstream sserr;
         sserr<<"SimpleJetCorrector ERROR: inversion variable: "<<mInvertVar<<" greater than maximum "<<N-1;
         throw std::runtime_error(sserr.str());
      #else
         throw cms::Exception("SimpleJetCorrector")<<" inversion variable: "<<mInvertVar<<" greater than maximum "<<N-1;
      #endif
    }
  float precision = 0.0001;
  float rsp = 1.0;
  float e = 1.0;
  float x[4] = {0.0,0.0,0.0,0.0};
  for(unsigned i=0;i<N;i++)
    x[i] = fX[i]; 
  unsigned nLoop=0;
  while(e > precision && nLoop < nMax) 
    {
      rsp = mFunc->Eval(x[0],x[1],x[2],x[3]);
      float tmp = x[mInvertVar] * rsp;
      e = fabs(tmp - fX[mInvertVar])/fX[mInvertVar];
      x[mInvertVar] = fX[mInvertVar]/rsp;
      nLoop++;
    }
  return 1./rsp;
}
//------------------------------------------------------------------------ 
//--- quadratic interpolation --------------------------------------------
//------------------------------------------------------------------------
float SimpleJetCorrector::quadraticInterpolation(float fZ, const float fX[3], const float fY[3]) const
{
  // Quadratic interpolation through the points (x[i],y[i]). First find the parabola that
  // is defined by the points and then calculate the y(z).
  float D[4],a[3];
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
  float r = a[0]+fZ*(a[1]+fZ*a[2]);
  return r;
}
