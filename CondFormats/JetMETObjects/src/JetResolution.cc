////////////////////////////////////////////////////////////////////////////////
//
// JetResolution
// -------------
//
//            11/05/2010 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "CondFormats/JetMETObjects/interface/JetResolution.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TMath.h>


#include <iostream>
#include <sstream>
#include <cassert>


using namespace std;


////////////////////////////////////////////////////////////////////////////////
// GLOBAL FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
double fnc_dscb(double*xx,double*pp);
double fnc_gaussalpha(double*xx,double*pp);
double fnc_gaussalpha1alpha2(double*xx,double*pp);


////////////////////////////////////////////////////////////////////////////////
// CONSTRUCTION / DESTRUCTION
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
JetResolution::JetResolution()
  : resolutionFnc_(0)
{
  resolutionFnc_ = new TF1();
}


//______________________________________________________________________________
JetResolution::JetResolution(const string& fileName,bool doGaussian)
  : resolutionFnc_(0)
{
  initialize(fileName,doGaussian);
}


//______________________________________________________________________________
JetResolution::~JetResolution()
{
  delete resolutionFnc_;
  for (unsigned i=0;i<parameterFncs_.size();i++) delete parameterFncs_[i];
  for (unsigned i=0;i<parameters_.size();i++)    delete parameters_[i];
}


////////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION OF MEMBER FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void JetResolution::initialize(const string& fileName,bool doGaussian)
{
  size_t pos;

  name_ = fileName;
  pos = name_.find_last_of('.'); name_ = name_.substr(0,pos);
  pos = name_.find_last_of('/'); name_ = name_.substr(pos+1);

  JetCorrectorParameters resolutionPars(fileName,"resolution");
  string fncname = "fResolution_" + name_;
  string formula = resolutionPars.definitions().formula();
  if      (doGaussian)                   resolutionFnc_=new TF1(fncname.c_str(),"gaus",0.,5.);
  else if (formula=="DSCB")              resolutionFnc_=new TF1(fncname.c_str(),fnc_dscb,0.,5.,7);
  else if (formula=="GaussAlpha1Alpha2") resolutionFnc_=new TF1(fncname.c_str(),fnc_gaussalpha1alpha2,-5.,5.,5);
  else if (formula=="GaussAlpha")        resolutionFnc_=new TF1(fncname.c_str(),fnc_gaussalpha,-5.,5.,4);
  else                                   resolutionFnc_=new TF1(fncname.c_str(),formula.c_str(),0.,5.);

  resolutionFnc_->SetNpx(200);
  resolutionFnc_->SetParName(0,"N");
  resolutionFnc_->SetParameter(0,1.0);
  unsigned nPar(1);

  string tmp = resolutionPars.definitions().level();
  pos = tmp.find(':');
  while (!tmp.empty()) {
    string paramAsStr = tmp.substr(0,pos);
    if (!doGaussian||paramAsStr=="mean"||paramAsStr=="sigma") {
      parameters_.push_back(new JetCorrectorParameters(fileName,paramAsStr));
      formula = parameters_.back()->definitions().formula();
      parameterFncs_.push_back(new TF1(("f"+paramAsStr+"_"+name()).c_str(),formula.c_str(),
				       parameters_.back()->record(0).parameters()[0],
				       parameters_.back()->record(0).parameters()[1]));
      resolutionFnc_->SetParName(nPar,parameters_.back()->definitions().level().c_str());
      nPar++;
    }
    tmp = (pos==string::npos) ? "" : tmp.substr(pos+1);
    pos = tmp.find(':');
  }

  assert(nPar==(unsigned)resolutionFnc_->GetNpar());
  assert(!doGaussian||nPar==3);
}


//______________________________________________________________________________
TF1* JetResolution::resolutionEtaPt(float eta, float pt) const
{
  vector<float> x; x.push_back(eta);
  vector<float> y; y.push_back(pt);
  return resolution(x,y);
}


//______________________________________________________________________________
TF1* JetResolution::resolution(const vector<float>& x,
			       const vector<float>& y) const
{
  unsigned N(y.size());
  for (unsigned iPar=0;iPar<parameters_.size();iPar++) {
    int bin = parameters_[iPar]->binIndex(x);
    assert(bin>=0);
    assert(bin<(int)parameters_[iPar]->size());
    const std::vector<float>& pars = parameters_[iPar]->record(bin).parameters();
    for (unsigned i=2*N;i<pars.size();i++)
      parameterFncs_[iPar]->SetParameter(i-2*N,pars[i]);
    float yy[4] = {};
    for (unsigned i=0;i<N;i++)
      yy[i] = (y[i] < pars[2*i]) ? pars[2*i] : (y[i] > pars[2*i+1]) ? pars[2*i+1] : y[i];
    resolutionFnc_->SetParameter(iPar+1,
				 parameterFncs_[iPar]->Eval(yy[0],yy[1],yy[2],yy[3]));
  }
  return resolutionFnc_;
}


//______________________________________________________________________________
TF1* JetResolution::parameterEta(const string& parameterName, float eta)
{
  vector<float> x; x.push_back(eta);
  return parameter(parameterName,x);
}


//______________________________________________________________________________
TF1* JetResolution::parameter(const string& parameterName,const vector<float>& x)
{
  TF1* result(0);
  for (unsigned i=0;i<parameterFncs_.size()&&result==0;i++) {
    string fncname = parameterFncs_[i]->GetName();
    if (fncname.find("f"+parameterName)==0) {
      stringstream ssname; ssname<<parameterFncs_[i]->GetName();
      for (unsigned ii=0;ii<x.size();ii++)
	ssname<<"_"<<parameters_[i]->definitions().binVar(ii)<<x[ii];
      result = (TF1*)parameterFncs_[i]->Clone();
      result->SetName(ssname.str().c_str());
      int N = parameters_[i]->definitions().nParVar();
      int bin = parameters_[i]->binIndex(x);
      assert(bin>=0);
      assert(bin<(int)parameters_[i]->size());
      const std::vector<float>& pars = parameters_[i]->record(bin).parameters();
      for (unsigned ii=2*N;ii<pars.size();ii++) result->SetParameter(ii-2*N,pars[ii]);
    }
  }

  if (0==result) cerr<<"JetResolution::parameter() ERROR: no parameter "
		     <<parameterName<<" found."<<endl;

  return result;
}


//______________________________________________________________________________
double JetResolution::parameterEtaEval(const std::string& parameterName, float eta, float pt)
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
    edm::LogError("ParameterNotFound") << "JetResolution::parameterEtaEval(): no parameter \""
				  << parameterName << "\" found" << std::endl;

  std::vector<float> etas; etas.push_back(eta);
  int bin = params->binIndex(etas);

  if ( !(0 <= bin && bin < (int)params->size() ) )
    edm::LogError("ParameterNotFound") << "JetResolution::parameterEtaEval(): bin out of range: "
				       << bin << std::endl;

  const std::vector<float>& pars = params->record(bin).parameters();

  int N = params->definitions().nParVar();
  for (unsigned ii = 2*N; ii < pars.size(); ++ii)
    {
      func->SetParameter(ii-2*N, pars[ii]);
    }

  return func->Eval(pt);
}


////////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION OF GLOBAL FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
double fnc_dscb(double*xx,double*pp)
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
double fnc_gaussalpha(double *v, double *par)
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
double fnc_gaussalpha1alpha2(double *v, double *par)
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

