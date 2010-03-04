//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: SimpleZSPJetCorrector.cc,v 1.4 2009/11/24 13:45:19 kodolova Exp $
//
// ZSP Jet Corrector
//
#include "RecoJets/JetPlusTracks/interface/SimpleZSPJPTJetCorrector.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Math/PtEtaPhiE4D.h"
#include "Math/LorentzVector.h"
typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > PtEtaPhiELorentzVectorD;
typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > XYZTLorentzVectorD;

using namespace std;


namespace zspjpt {

  bool debug = false;
  
} // namespace

SimpleZSPJPTJetCorrector::SimpleZSPJPTJetCorrector ()
{}

SimpleZSPJPTJetCorrector::SimpleZSPJPTJetCorrector (const std::string& fDataFile) 
{
  mParameters = new SimpleJetCorrectorParameters (fDataFile);
// Read parameters
 if (zspjpt::debug) {
  std::cout<<" Size of parameters as read by SimpleJetCorrectorParameters "<<mParameters->size()<<std::endl;
  for(int i = 0; i<mParameters->size(); i++){
   const std::vector<float> p = mParameters->record (i).parameters ();
    for(std::vector<float>::const_iterator j=p.begin(); j<p.end(); j++) {
         std::cout<<" Parameter number "<<mParameters->record (i).etaMin()<<" "<<mParameters->record (i).etaMax()<<" "<<(*j)<<std::endl;
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
  int band = mParameters->bandIndex(eta);

  if(band < 0) return factor;

  const std::vector<float> p = mParameters->record (band).parameters ();

  if (zspjpt::debug) {
     cout<<" Et and eta of jet "<<et<<" "<<eta<<" bin "<<band<<" "<<p[1]<<" "<<p[2]<<" "<<p[3]<<
     " "<<p[4]<<" "<<p[5]<<endl;
  } 

  double koef = 1. - p[2]*exp(p[3]*et)-p[4]*exp(p[5]*et);
  double etnew = et/koef;

  if (zspjpt::debug) cout<<" The new energy found "<<etnew<<" "<<et<<endl;
  
  return etnew/et;
}
