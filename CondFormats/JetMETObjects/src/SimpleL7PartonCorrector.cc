//
// Original Author: Attilio Santocchia Feb. 28, 2008
//
// Jet Parton dependent corrections 
//
#include "CondFormats/JetMETObjects/interface/SimpleL7PartonCorrector.h"

#include "CondFormats/JetMETObjects/interface/SimpleJetCorrectorParameters.h"

#include <vector>

#include "FWCore/Utilities/interface/Exception.h"

#include "Math/PtEtaPhiE4D.h"
#include "Math/LorentzVector.h"
typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > PtEtaPhiELorentzVectorD;
typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > XYZTLorentzVectorD;

namespace {
  const unsigned nParameters = 5;
}

SimpleL7PartonCorrector::SimpleL7PartonCorrector () 
  : mParameters (0) 
{}

SimpleL7PartonCorrector::SimpleL7PartonCorrector (const std::string& fDataFile, const std::string& fSection) 
  : mParameters (new SimpleJetCorrectorParameters (fDataFile, fSection)) 
{}

SimpleL7PartonCorrector::~SimpleL7PartonCorrector () {
  delete mParameters;
}

double SimpleL7PartonCorrector::correctionPtEta (double fPt, double fEta) const {

  //needed to avoid anomalous infinite while loop
  int nMax = 50;

  double precision = 0.001;
  double theResp = 1;

  double relDIFF = 1.0;
  double sxPtParton = fPt*0.25;
  double dxPtParton = fPt*4.0;
  if(fPt<sxPtParton || fPt>dxPtParton) {
    std::cout << "WRONG STARTING LIMITS" << std::endl;
    return 1.0;
  }
  double cnPtParton = (dxPtParton+sxPtParton)/2;
  int nLoop=0;
  while(relDIFF>precision && nLoop<nMax) {
    cnPtParton = (dxPtParton+sxPtParton)/2;
    theResp  = theResponseFunction(fEta,cnPtParton);
    float tmpPtJet = cnPtParton * theResp;
    relDIFF = fabs(tmpPtJet - fPt)/fPt;
    //std::cout << sxPtParton << ":" << dxPtParton 
    //          << " - PtPar:" << cnPtParton << " Rsp:" << theResp << " tmpPtJet:" << tmpPtJet << std::endl;
    if(fPt<tmpPtJet) dxPtParton=cnPtParton;
    else            sxPtParton=cnPtParton;
    nLoop++;
  }
  //theResp  = theResponseFunction(fEta,cnPtParton);
  //std::cout << cnPtParton << " " <<  relDIFF << std::endl;
  return 1./theResp;

}

double SimpleL7PartonCorrector::correctionXYZT (double fPx, double fPy, double fPz, double fE) const {
  XYZTLorentzVectorD p4 (fPx, fPy, fPz, fE);
  return correctionPtEta (p4.Pt(), p4.Eta());
}

double SimpleL7PartonCorrector::correctionEtEtaPhiP (double fEt, double fEta, double fPhi, double fP) const {
  double costhetainv = cosh (fEta);
  return correctionPtEta (fP/costhetainv, fEta);
}

double SimpleL7PartonCorrector::theResponseFunction(double fEta,double fPt) const {
  const std::vector<float>& p = mParameters->record (0).parameters ();
/*
  std::cout << p[0] << " "
            << p[1] << " "
            << p[2] << " "
            << p[3] << " "
            << p[4] << " "
            << p[5] << " "
            << p[6] << " "
            << p[7] << " "
            << p[8] << " "
            << p[9] << " "
            << fEta << " " << fPt << std::endl;
*/
  if( fPt < p[0] ) return 1.;
  float aPar = 1/(p[2]*fPt+p[3]) + p[4];
  float bPar = p[5]+p[6]*log(fPt)+p[7]*log(fPt)*log(fPt);
  float cPar = p[8]+p[9]*fPt;
  float response = aPar+bPar*fabs(fEta)+cPar*fEta*fEta;
  return response;
}

