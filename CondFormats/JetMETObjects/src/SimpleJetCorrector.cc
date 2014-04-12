#include "CondFormats/JetMETObjects/interface/SimpleJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/src/Utilities.cc"
#include <iostream>
#include <sstream>
#include <cmath>

//------------------------------------------------------------------------
//--- Default SimpleJetCorrector constructor -----------------------------
//------------------------------------------------------------------------
SimpleJetCorrector::SimpleJetCorrector()
{
  mDoInterpolation = false;
  mInvertVar       = 9999;
}
//------------------------------------------------------------------------
//--- SimpleJetCorrector constructor -------------------------------------
//--- reads arguments from a file ----------------------------------------
//------------------------------------------------------------------------
SimpleJetCorrector::SimpleJetCorrector(const std::string& fDataFile, const std::string& fOption):
  mParameters(fDataFile,fOption),
  mFunc("function",((mParameters.definitions()).formula()).c_str())
{
  mDoInterpolation = false;
  if (mParameters.definitions().isResponse())
    mInvertVar = findInvertVar();
}
//------------------------------------------------------------------------
//--- SimpleJetCorrector constructor -------------------------------------
//--- reads arguments from a file ----------------------------------------
//------------------------------------------------------------------------
SimpleJetCorrector::SimpleJetCorrector(const JetCorrectorParameters& fParameters):
  mParameters(fParameters),
  mFunc("function",((mParameters.definitions()).formula()).c_str())
{
  mDoInterpolation = false;
  if (mParameters.definitions().isResponse())
    mInvertVar = findInvertVar();
}
//------------------------------------------------------------------------
//--- SimpleJetCorrector destructor --------------------------------------
//------------------------------------------------------------------------
SimpleJetCorrector::~SimpleJetCorrector()
{
}

//------------------------------------------------------------------------
//--- calculates the correction ------------------------------------------
//------------------------------------------------------------------------
float SimpleJetCorrector::correction(const std::vector<float>& fX,const std::vector<float>& fY) const
{
  float result = 1.;
  float tmp    = 0.0;
  float cor    = 0.0;
  int bin = mParameters.binIndex(fX);
  if (bin<0)
    return result;
  if (!mDoInterpolation)
    result = correctionBin(bin,fY);
  else
    {
      for(unsigned i=0;i<mParameters.definitions().nBinVar();i++)
        {
          float xMiddle[3];
          float xValue[3];
          int prevBin = mParameters.neighbourBin((unsigned)bin,i,false);
          int nextBin = mParameters.neighbourBin((unsigned)bin,i,true);
          if (prevBin>=0 && nextBin>=0)
            {
              xMiddle[0] = mParameters.record(prevBin).xMiddle(i);
              xMiddle[1] = mParameters.record(bin).xMiddle(i);
              xMiddle[2] = mParameters.record(nextBin).xMiddle(i);
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
      result = tmp/mParameters.definitions().nBinVar();
    }
  return result;
}
//------------------------------------------------------------------------
//--- calculates the correction for a specific bin -----------------------
//------------------------------------------------------------------------
float SimpleJetCorrector::correctionBin(unsigned fBin,const std::vector<float>& fY) const
{
  if (fBin >= mParameters.size())
    {
      std::stringstream sserr;
      sserr<<"wrong bin: "<<fBin<<": only "<<mParameters.size()<<" available!";
      handleError("SimpleJetCorrector",sserr.str());
    }
  unsigned N = fY.size();
  if (N > 4)
    {
      std::stringstream sserr;
      sserr<<"two many variables: "<<N<<" maximum is 4";
      handleError("SimpleJetCorrector",sserr.str());
    }
  float result = -1;
  //Have to do calculation using a temporary TFormula to avoid
  // thread safety issues
  TFormula tFunc(mFunc);

  const std::vector<float>& par = mParameters.record(fBin).parameters();
  for(unsigned int i=2*N;i<par.size();i++)
    tFunc.SetParameter(i-2*N,par[i]);
  float x[4] = {};
  std::vector<float> tmp;
  for(unsigned i=0;i<N;i++)
    {
      x[i] = (fY[i] < par[2*i]) ? par[2*i] : (fY[i] > par[2*i+1]) ? par[2*i+1] : fY[i];
      tmp.push_back(x[i]);
    }
  if (mParameters.definitions().isResponse())
    result = invert(tmp,tFunc);
  else
    result = tFunc.Eval(x[0],x[1],x[2],x[3]);
  return result;
}
//------------------------------------------------------------------------
//--- find invertion variable (JetPt) ------------------------------------
//------------------------------------------------------------------------
unsigned SimpleJetCorrector::findInvertVar()
{
  unsigned result = 9999;
  std::vector<std::string> vv = mParameters.definitions().parVar();
  for(unsigned i=0;i<vv.size();i++)
    if (vv[i]=="JetPt")
      {
        result = i;
        break;
      }
  if (result >= vv.size())
    handleError("SimpleJetCorrector","Response inversion is required but JetPt is not specified as parameter");
  return result;
}
//------------------------------------------------------------------------
//--- inversion ----------------------------------------------------------
//------------------------------------------------------------------------
float SimpleJetCorrector::invert(const std::vector<float>& fX, TFormula& tFunc) const
{
  unsigned nMax = 50;
  unsigned N = fX.size();
  float precision = 0.0001;
  float rsp = 1.0;
  float e = 1.0;
  float x[4] = {0.0,0.0,0.0,0.0};
  for(unsigned i=0;i<N;i++)
    x[i] = fX[i];
  unsigned nLoop=0;
  while(e > precision && nLoop < nMax)
    {
      rsp = tFunc.Eval(x[0],x[1],x[2],x[3]);
      float tmp = x[mInvertVar] * rsp;
      e = fabs(tmp - fX[mInvertVar])/fX[mInvertVar];
      x[mInvertVar] = fX[mInvertVar]/rsp;
      nLoop++;
    }
  return 1./rsp;
}








