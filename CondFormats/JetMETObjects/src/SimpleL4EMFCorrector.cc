//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: SimpleL4EMFCorrector.cc,v 1.1.2.2 2008/02/15 22:35:39 fedor Exp $
//
// MC Jet Corrector
//
#include "CondFormats/JetMETObjects/interface/SimpleL4EMFCorrector.h"

#include "CondFormats/JetMETObjects/interface/SimpleJetCorrectorParameters.h"

#include <vector>

#include "FWCore/Utilities/interface/Exception.h"

#include "Math/PtEtaPhiE4D.h"
#include "Math/LorentzVector.h"
typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > PtEtaPhiELorentzVectorD;
typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > XYZTLorentzVectorD;

namespace {
  const unsigned nParameters = 32;
  const double PT_CUTOFF = 30.;
}

SimpleL4EMFCorrector::SimpleL4EMFCorrector () 
  : mParameters (0) 
{}

SimpleL4EMFCorrector::SimpleL4EMFCorrector (const std::string& fDataFile) 
  : mParameters (new SimpleJetCorrectorParameters (fDataFile)) 
{}

SimpleL4EMFCorrector::~SimpleL4EMFCorrector () {
  delete mParameters;
}

double SimpleL4EMFCorrector::correctionPtEtaEmfraction (double fPt, double fEta, double fEmFraction) const {
  unsigned band = mParameters->bandIndex (fEta);
  const std::vector<float>& p = mParameters->record (band).parameters ();
  if (p.size() == nParameters) { // defined band: get params
    return correctionBandPtEtaEmfraction (band, fPt, fEta, fEmFraction);
  }
  else { // interpolation
    double eta1 =  mParameters->record (band).etaMin();
    double eta2 =  mParameters->record (band).etaMax();
    double correction1 = correctionBandPtEtaEmfraction (band-1, fPt, fEta, fEmFraction);
    double correction2 = correctionBandPtEtaEmfraction (band+1, fPt, fEta, fEmFraction);
    return (correction1*(eta2-fEta)+correction2*(fEta-eta1))/(eta2-eta1);
  }
}

double SimpleL4EMFCorrector::correctionXYZTEmfraction (double fPx, double fPy, double fPz, double fE, double fEmFraction) const {
  XYZTLorentzVectorD p4 (fPx, fPy, fPz, fE);
  return correctionPtEtaEmfraction (p4.Pt(), p4.Eta(), fEmFraction);
}

double SimpleL4EMFCorrector::correctionEtEtaPhiPEmfraction (double fEt, double fEta, double fPhi, double fP, double fEmFraction) const {
  double costhetainv = cosh (fEta);
  return correctionPtEtaEmfraction (fP/costhetainv, fEta, fEmFraction);
}

double SimpleL4EMFCorrector::correctionBandPtEtaEmfraction (unsigned fBand, double fPt, double fEta, double fEmFraction) const {
  if (fBand >= mParameters->size()) {
    throw cms::Exception ("SimpleL4EMFCorrector") 
      << "wrong band: " << fBand << ": only " <<  mParameters->size() << " is available";
  }
  const std::vector<float>& p = mParameters->record (fBand).parameters ();
  if (p.size() != nParameters) {
    throw cms::Exception ("SimpleL4EMFCorrector") 
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
  double p0 = p[0] +
    p[1]*aeta + p[2]*emf +
    p[3]*aeta2 + p[4]*aeta*emf + p[5]*emf2 +
    p[6]*aeta3 + p[7]*aeta2*emf + p[8]*aeta*emf2 + p[9]*emf3 +
    p[10]*aeta4 + p[11]*aeta3*emf + p[12]*aeta2*emf2 + p[13]*aeta*emf3 + p[14]*emf4 +
    p[15]*aeta3*emf2;
  double p1 = p[16] +
    p[17]*aeta + p[18]*emf +
    p[19]*aeta2 + p[20]*aeta*emf + p[21]*emf2 +
    p[22]*aeta3 + p[23]*aeta2*emf + p[24]*aeta*emf2 + p[25]*emf3 +
    p[26]*aeta4 + p[27]*aeta3*emf + p[28]*aeta2*emf2 + p[29]*aeta*emf3 + p[30]*emf4 +
    p[31]*aeta3*emf2;
  double effectivePt = fPt > PT_CUTOFF ? fPt : PT_CUTOFF;
  double logPt = log (effectivePt);
  double deltaPt = p0*logPt + p1*logPt*logPt*logPt*sqrt(effectivePt);
  return 1+deltaPt/effectivePt;
}

