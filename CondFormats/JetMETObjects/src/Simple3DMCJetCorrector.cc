//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: Simple3DMCJetCorrector.cc,v 1.1 2007/11/01 21:50:30 fedor Exp $
//
// MC Jet Corrector
//
#include "CondFormats/JetMETObjects/interface/Simple3DMCJetCorrector.h"

#include "CondFormats/JetMETObjects/interface/SimpleJetCorrectorParameters.h"

#include <vector>

#include "FWCore/Utilities/interface/Exception.h"

#include "Math/PtEtaPhiE4D.h"
#include "Math/LorentzVector.h"
typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > PtEtaPhiELorentzVectorD;
typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > XYZTLorentzVectorD;

namespace {
  const unsigned nParameters = 20;
}

Simple3DMCJetCorrector::Simple3DMCJetCorrector () 
  : mParameters (0) 
{}

Simple3DMCJetCorrector::Simple3DMCJetCorrector (const std::string& fDataFile) 
  : mParameters (new SimpleJetCorrectorParameters (fDataFile)) 
{}

Simple3DMCJetCorrector::~Simple3DMCJetCorrector () {
  delete mParameters;
}

double Simple3DMCJetCorrector::correctionXYZTEmfraction (double fPx, double fPy, double fPz, double fE, double fEmFraction) const {
  XYZTLorentzVectorD p4 (fPx, fPy, fPz, fE);
  return correctionPtEtaEmfraction (p4.Pt(), p4.Eta(), fEmFraction);
}

double Simple3DMCJetCorrector::correctionEtEtaPhiPEmfraction (double fEt, double fEta, double fPhi, double fP, double fEmFraction) const {
  double costhetainv = cosh (fEta);
  return correctionPtEtaEmfraction (fP/costhetainv, fEta, fEmFraction);
}

double Simple3DMCJetCorrector::correctionPtEtaEmfraction (double fPt, double fEta, double fEmFraction) const {
  unsigned band = mParameters->bandIndex (fEta);
  if (band >= mParameters->size()) {
    throw cms::Exception ("Simple3DMCJetCorrector") 
      << "wrong band: " << band << ": only " <<  mParameters->size() << " is available";
  }
  const std::vector<float>& p = mParameters->record (band).parameters ();
  if (p.size() != nParameters) {
    throw cms::Exception ("Simple3DMCJetCorrector") 
      << "wrong # of parameters: " << nParameters << " expected, " << p.size() << " got";
  }
  double aeta = fabs(fEta);
  double aeta2 = aeta*aeta;
  double aeta3 = aeta2*aeta;
  double aeta4 = aeta3*aeta;
  double emf = fEmFraction;
  double emf2 = emf*emf;
  double emf3 = emf2*emf;
  double emf4 = emf3*emf;

  double p0 = p[0]*emf + 
    p[1]*aeta2 + p[2]*emf*aeta + p[3]*emf2 + 
    p[4]*emf3;
  double p1 = p[5] + 
    p[6]*aeta + p[7]*emf + 
    p[8]*aeta2 + p[9]*aeta*emf + p[10]*emf2 + 
    p[11]*aeta3 + p[12]*aeta2*emf + p[13]*aeta*emf2 + p[14]*emf3 + 
    p[15]*aeta4 + p[16]*aeta3*emf + p[17]*aeta2*emf2 + p[18]*aeta*emf3 + p[19]*emf4;
  double offset = p0 * (pow (fPt, p1) - 1.);
  double result = 1;
  // corrections valid in limited area only;
  return (fPt > 0 && fPt < 300) ? result = 1. + offset/fPt : 1.;
}

