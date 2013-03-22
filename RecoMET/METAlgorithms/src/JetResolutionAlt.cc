// -*- C++ -*-
// $Id: METProducer.cc,v 1.51 2012/08/14 13:11:37 eulisse Exp $

//____________________________________________________________________________||
#include "RecoMET/METAlgorithms/interface/JetResolutionAlt.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TMath.h>
#include <TF1.h>


#include <iostream>
#include <sstream>
#include <cassert>

//____________________________________________________________________________||
namespace jetresolutionaltfuncs
{
  double fnc_dscb(double*xx,double*pp);
  double fnc_gaussalpha(double*xx,double*pp);
  double fnc_gaussalpha1alpha2(double*xx,double*pp);
}

//____________________________________________________________________________||
JetResolutionAlt::JetResolutionAlt(const std::string& fileName,bool doGaussian)
{
  initialize(fileName, doGaussian);
}

//____________________________________________________________________________||
JetResolutionAlt::~JetResolutionAlt()
{
  for (std::vector<TF1*>::size_type i = 0; i < parameterFncs_.size(); ++i)
    delete parameterFncs_[i];

  for (std::vector<JetCorrectorParameters*>::size_type i = 0; i < parameters_.size(); ++i)
    delete parameters_[i];
}

//____________________________________________________________________________||
void JetResolutionAlt::initialize(const std::string& fileName, bool doGaussian)
{

  JetCorrectorParameters resolutionPars(fileName, "resolution");
  std::string formulaName = resolutionPars.definitions().formula();

  // e.g. "Spring10_PtResolution_AK5PF"
  std::string resolutionName = mkResolutionNameFrom(fileName);

  TF1* resolutionFunction = mkResolutionFunction(doGaussian, formulaName, resolutionName);

  int nPars(1);
  
  std::vector<std::string> levelNames = readLevelNames(resolutionPars);

  for(std::vector<std::string>::iterator levelName = levelNames.begin(); levelName != levelNames.end(); ++levelName)
    {
      if ( !(!doGaussian || *levelName == "mean" || *levelName == "sigma") ) continue;

      parameters_.push_back(new JetCorrectorParameters(fileName,*levelName));
      formulaName = parameters_.back()->definitions().formula();

      parameterFncs_.push_back(new TF1(("f" + *levelName + "_" + resolutionName).c_str(),
				       formulaName.c_str(),
				       parameters_.back()->record(0).parameters()[0],
				       parameters_.back()->record(0).parameters()[1])
			       );

      resolutionFunction->SetParName(nPars,parameters_.back()->definitions().level().c_str());
      nPars++;
    }

  if ( nPars != resolutionFunction->GetNpar() )
    edm::LogError("ParameterNotFound") << "JetResolutionAlt::parameterEtaEval(): incorrect number of parameters: "
				       << nPars << std::endl;

  if ( !(!doGaussian || nPars == 3) )
    edm::LogError("ParameterNotFound") << "JetResolutionAlt::parameterEtaEval(): incorrect number of parameters: "
				       << nPars << std::endl;

}
  

//____________________________________________________________________________||
double JetResolutionAlt::parameterEtaEval(const std::string& parameterName, float eta, float pt)
{
  TF1* func(0);
  JetCorrectorParameters* params(0);
  for (std::vector<TF1*>::size_type ifunc = 0; ifunc < parameterFncs_.size(); ++ifunc)
    {
      std::string fncname = parameterFncs_[ifunc]->GetName();
      if ( !(fncname.find("f"+parameterName) == 0) ) continue;
      params = parameters_[ifunc];
      func = (TF1*)parameterFncs_[ifunc];
      break;
    }

  if (!func)
    edm::LogError("ParameterNotFound") << "JetResolutionAlt::parameterEtaEval(): no parameter \""
				  << parameterName << "\" found" << std::endl;

  std::vector<float> etas; etas.push_back(eta);
  int bin = params->binIndex(etas);

  if ( !(0 <= bin && bin < (int)params->size() ) )
    edm::LogError("ParameterNotFound") << "JetResolutionAlt::parameterEtaEval(): bin out of range: "
				       << bin << std::endl;

  const std::vector<float>& pars = params->record(bin).parameters();

  int N = params->definitions().nParVar();
  for (unsigned ii = 2*N; ii < pars.size(); ++ii)
    {
      func->SetParameter(ii-2*N, pars[ii]); 
    }
  
  return func->Eval(pt);
}

//____________________________________________________________________________||
std::string JetResolutionAlt::mkResolutionNameFrom(const std::string& fileName)
{
  std::string ret = fileName;
  ret = ret.substr(0, ret.find_last_of('.'));
  ret = ret.substr(ret.find_last_of('/') + 1);
  return ret; // e.g. "Spring10_PtResolution_AK5PF"
}

//____________________________________________________________________________||
TF1* JetResolutionAlt::mkResolutionFunction(bool doGaussian, const std::string& formulaName, const std::string& resolutionName)
{
  TF1* ret;

  std::string functionName = "fResolution_" + resolutionName;

  if (doGaussian)
    ret = new TF1(functionName.c_str(),"gaus",0.,5.);
  else if (formulaName=="DSCB")
    ret = new TF1(functionName.c_str(), jetresolutionaltfuncs::fnc_dscb, 0., 5., 7);
  else if (formulaName=="GaussAlpha1Alpha2")
    ret = new TF1(functionName.c_str(), jetresolutionaltfuncs::fnc_gaussalpha1alpha2, -5., 5., 5);
  else if (formulaName=="GaussAlpha")
    ret = new TF1(functionName.c_str(), jetresolutionaltfuncs::fnc_gaussalpha, -5., 5., 4);
  else 
    ret = new TF1(functionName.c_str(),formulaName.c_str(),0.,5.);

  ret->SetNpx(200);
  ret->SetParName(0, "N");
  ret->SetParameter(0, 1.0);

  return ret;
}

//____________________________________________________________________________||
std::vector<std::string> JetResolutionAlt::readLevelNames(const JetCorrectorParameters& resolutionPars)
{
  std::vector<std::string> ret;

  // e.g. "mean:sigma:aone:pone:atwo:ptwo"
  std::string levelNamesByColon = resolutionPars.definitions().level();

  while (!levelNamesByColon.empty())
    {
      size_t posColon = levelNamesByColon.find(':');
      std::string levelName = levelNamesByColon.substr(0, posColon);
      ret.push_back(levelName);
      levelNamesByColon = (posColon == std::string::npos) ? "" : levelNamesByColon.substr(posColon + 1);
    }

  return ret;
}

//____________________________________________________________________________||
double jetresolutionaltfuncs::fnc_dscb(double*xx,double*pp)
{
  double x   = xx[0];
  double N   = pp[0];
  double mu  = pp[1];
  double sig = pp[2];
  double a1  = pp[3];
  double p1  = pp[4];
  double a2  = pp[5];
  double p2  = pp[6];
  
  double u   = (x-mu)/sig;
  double A1  = TMath::Power(p1/TMath::Abs(a1),p1)*TMath::Exp(-a1*a1/2);
  double A2  = TMath::Power(p2/TMath::Abs(a2),p2)*TMath::Exp(-a2*a2/2);
  double B1  = p1/TMath::Abs(a1) - TMath::Abs(a1);
  double B2  = p2/TMath::Abs(a2) - TMath::Abs(a2);

  double result(N);
  if      (u<-a1) result *= A1*TMath::Power(B1-u,-p1);
  else if (u<a2)  result *= TMath::Exp(-u*u/2);
  else            result *= A2*TMath::Power(B2+u,-p2);
  return result;
}


//______________________________________________________________________________
double jetresolutionaltfuncs::fnc_gaussalpha(double *v, double *par)
{
    double N    =par[0];
    double mean =par[1];
    double sigma=par[2];
    double alpha=par[3];
    double t    =TMath::Abs((v[0]-mean)/sigma);
    double cut  = 1.0;
    return (t<=cut) ? N*TMath::Exp(-0.5*t*t) : N*TMath::Exp(-0.5*(alpha*(t-cut)+cut*cut));
}


//______________________________________________________________________________
double jetresolutionaltfuncs::fnc_gaussalpha1alpha2(double *v, double *par)
{
    double N     =par[0];
    double mean  =par[1];
    double sigma =par[2];
    double alpha1=par[3];
    double alpha2=par[4];
    double t     =TMath::Abs((v[0]-mean)/sigma);
    double cut = 1.0;
    return
      (t<=cut) ? N*TMath::Exp(-0.5*t*t) :
      ((v[0]-mean)>=0) ? N*TMath::Exp(-0.5*(alpha1*(t-cut)+cut*cut)) :
      N*TMath::Exp(-0.5*(alpha2*(t-cut)+cut*cut));
}

//______________________________________________________________________________
