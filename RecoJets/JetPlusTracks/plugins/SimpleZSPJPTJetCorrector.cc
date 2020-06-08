//
// Original Author:  Fedor Ratnikov Dec 27, 2006
//
// ZSP Jet Corrector
//
#include "SimpleZSPJPTJetCorrector.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Math/PtEtaPhiE4D.h"
#include "Math/LorentzVector.h"
typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > PtEtaPhiELorentzVectorD;
typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > XYZTLorentzVectorD;

using namespace std;


SimpleZSPJPTJetCorrector::SimpleZSPJPTJetCorrector ()
{ debug_=false;}

SimpleZSPJPTJetCorrector::SimpleZSPJPTJetCorrector (const std::string& fDataFile) 
{
  debug_=false;
  mParameters = new JetCorrectorParameters (fDataFile,"");

  //std::cout<<" Formula "<<((mParameters->definitions()).formula()).c_str()<<std::endl;

  mFunc            = new TFormula("function",((mParameters->definitions()).formula()).c_str());

// Read parameters
 if (debug_) {
  std::cout<<" Size of parameters as read by SimpleJetCorrectorParameters "<<mParameters->size()<<std::endl;
  for(unsigned int i = 0; i<mParameters->size(); i++){
   const std::vector<float> p = mParameters->record (i).parameters ();
    for(std::vector<float>::const_iterator j=p.begin(); j<p.end(); j++) {
         std::cout<<" Parameter number "<<mParameters->record (i).xMin(0)<<" "<<mParameters->record (i).xMax(0)<<" "<<(*j)<<std::endl;
     }
  }
 } // debug
}

SimpleZSPJPTJetCorrector::~SimpleZSPJPTJetCorrector () {
} 

double SimpleZSPJPTJetCorrector::correctionPtEtaPhiE (double fPt, double fEta, double fPhi, double fE) const {
  double costhetainv = cosh (fEta);
  return correctionEtEtaPhiP (fE/costhetainv, fEta, fPhi, fPt*costhetainv);
}

double SimpleZSPJPTJetCorrector::correctionEtEtaPhiP (double fEt, double fEta, double fPhi, double fP) const {
  
  double et=fEt;
  double eta=fabs (fEta);
  double factor = 1.;

// Define Eta bin for parametrization 
  std::vector<float> xx; xx.push_back(eta);
  int band = mParameters->binIndex(xx);

  if(band < 0) return factor;

  const std::vector<float> p = mParameters->record (band).parameters ();

 // Set parameters 
  for(unsigned int i=0; i<p.size();i++) {
     mFunc->SetParameter(i,p[i]);
  }

  if (debug_) {
     cout<<" Et and eta of jet "<<et<<" "<<eta<<" bin "<<band<<" "<<p[1]<<" "<<p[2]<<" "<<p[3]<<
     " "<<p[4]<<" "<<p[5]<<endl;
  } 

//  double koef = 1. - p[2]*exp(p[3]*et)-p[4]*exp(p[5]*et);
  
  double koef = 1. - mFunc->Eval(et);

// If et calojet less then some value - use correction on the boundary
 
  if( et < p[0] ) koef = 1. - mFunc->Eval(p[0]);

//
  if(koef <= 0.000001) 
  {
   if (debug_)  std::cout<<"SimpleZSPJPTJetCorrector::Problem with ZSP corrections "<<koef<<std::endl; 
  koef = 1.;
  }

  double etnew = et/koef;

  if (debug_) cout<<" The new energy found "<<etnew<<" "<<et<<" "<<koef<<endl;
  
  return etnew/et;
}

double SimpleZSPJPTJetCorrector::correctionPUEtEtaPhiP (double fEt, double fEta, double fPhi, double fP) const {

  double et=fEt;
  double eta=fabs (fEta);
  double factor = 1.;

// Define Eta bin for parametrization 
  std::vector<float> xx; xx.push_back(eta);
  int band = mParameters->binIndex(xx);

  if(band < 0) return factor;

  const std::vector<float> p = mParameters->record (band).parameters ();

  if (debug_) {
     cout<<" Et and eta of jet "<<et<<" "<<eta<<" bin "<<band<<std::endl;
  }

  double koef = (et-p[2])/et;
  double etnew = et/koef;

  if (debug_) cout<<" The new energy found "<<etnew<<" "<<et<<endl;

  return etnew/et;
}



